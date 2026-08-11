"""Training targets and strict Top-8 membership loss for REC queries."""

import copy
import hashlib
import json
import math
import numbers
import struct
import weakref

import torch
import torch.nn.functional as F

from .rec_reranker import compute_query_ious, select_candidate_indices


_TOPK = 8
_THRESHOLDS = (0.25, 0.50)
_THRESHOLD_WEIGHTS = (2.0, 1.0)

SOURCE_GATE_TRAINABLE_PREFIX = (
    "prediction_heads.5.sem_cls_scores_head."
)
_SOURCE_GATE_TRAINABLE_MODULE = SOURCE_GATE_TRAINABLE_PREFIX[:-1]
_SOURCE_GATE_GROUP_NAME = "source_gate_semantic_classifier"
_SOURCE_GATE_LR = 1e-4
_SOURCE_GATE_WEIGHT_DECAY = 1e-4
_SOURCE_GATE_MAX_NORM = 1.0
_SOURCE_GATE_CONTRACT_SEAL = object()


class _RecSourceGateParameterContract:
    __slots__ = (
        "_names", "_parameters", "_mcln", "_parent", "_geometry", "_seal",
        "__weakref__",
    )

    def __init__(self, names, parameters, mcln, parent, geometry, seal):
        if seal is not _SOURCE_GATE_CONTRACT_SEAL:
            raise ValueError("source-gate contracts must come from configure")
        object.__setattr__(self, "_names", names)
        object.__setattr__(self, "_parameters", parameters)
        object.__setattr__(self, "_mcln", mcln)
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_geometry", geometry)
        object.__setattr__(self, "_seal", seal)

    def __setattr__(self, _name, _value):
        raise AttributeError("source-gate parameter contracts are immutable")

    @property
    def names(self):
        return self._names

    @property
    def parameters(self):
        return self._parameters

    @property
    def mcln(self):
        return self._mcln

    @property
    def parent(self):
        return self._parent

    @property
    def geometry(self):
        return self._geometry


_SOURCE_GATE_CONTRACT_PROVENANCE = weakref.WeakKeyDictionary()


def attach_full_query_targets(full_state, end_points, root_only=True):
    """Return detached IoUs for every full-state query."""
    gt_boxes = torch.cat([
        end_points["center_label"][..., :3].float(),
        end_points["size_gts"].float(),
    ], dim=-1)
    gt_mask = end_points["box_label_mask"]
    if root_only:
        gt_boxes = gt_boxes[:, :1]
        gt_mask = gt_mask[:, :1]
    return compute_query_ious(
        full_state["boxes"], gt_boxes, gt_mask
    ).detach()


def _is_finite_real(value):
    return (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_exact_tuple(name, value, expected):
    if type(value) is not tuple or len(value) != len(expected):
        raise ValueError("{} must be exactly {}".format(name, expected))
    for actual, required in zip(value, expected):
        if not _is_finite_real(actual) or float(actual) != required:
            raise ValueError("{} must be exactly {}".format(name, expected))


def _validate_production_contract(topk, thresholds, threshold_weights,
                                  margin, temperature):
    if type(topk) is not int or topk != _TOPK:
        raise ValueError("topk must be exactly 8")
    _validate_exact_tuple("thresholds", thresholds, _THRESHOLDS)
    _validate_exact_tuple(
        "threshold_weights", threshold_weights, _THRESHOLD_WEIGHTS
    )
    if not _is_finite_real(margin) or float(margin) != 0.0:
        raise ValueError("margin must be exactly 0.0")
    if not _is_finite_real(temperature) or float(temperature) != 1.0:
        raise ValueError("temperature must be exactly 1.0")


def _validate_loss_tensors(default_scores, query_ious, query_valid):
    tensors = (default_scores, query_ious, query_valid)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise ValueError("source-gate inputs must be tensors")
    if any(value.dim() != 2 for value in tensors):
        raise ValueError("source-gate inputs must have shape [B,Q]")
    if (query_ious.shape != default_scores.shape
            or query_valid.shape != default_scores.shape):
        raise ValueError("source-gate input shapes must match")
    if default_scores.shape[0] == 0 or default_scores.shape[1] == 0:
        raise ValueError("source-gate input axes cannot be empty")
    if (not torch.is_floating_point(default_scores)
            or not torch.is_floating_point(query_ious)):
        raise ValueError("scores and IoUs must have floating dtype")
    if default_scores.dtype != query_ious.dtype:
        raise ValueError("scores and IoUs must have the same dtype")
    if query_valid.dtype != torch.bool:
        raise ValueError("query_valid must have bool dtype")
    if (query_ious.device != default_scores.device
            or query_valid.device != default_scores.device):
        raise ValueError("source-gate inputs must share a device")
    if not bool(query_valid.any(dim=1).all().item()):
        raise ValueError("every row must contain a valid query")

    valid_scores = default_scores.masked_select(query_valid)
    valid_ious = query_ious.masked_select(query_valid)
    if not bool(torch.isfinite(valid_scores).all().item()):
        raise ValueError("valid default scores must be finite")
    if not bool(torch.isfinite(valid_ious).all().item()):
        raise ValueError("valid query IoUs must be finite")
    if bool(((valid_ious < 0.0) | (valid_ious > 1.0)).any().item()):
        raise ValueError("valid query IoUs must lie in [0,1]")


def _finite_float64(value):
    finfo = torch.finfo(torch.float64)
    return torch.nan_to_num(
        value,
        nan=0.0,
        posinf=float(finfo.max),
        neginf=float(finfo.min),
    )


def _cast_saturated(value, target_dtype):
    finfo = torch.finfo(target_dtype)
    lower = float(finfo.min)
    upper = float(finfo.max)
    return _finite_float64(value).clamp(
        min=lower, max=upper
    ).to(target_dtype)


def _threshold_membership_loss(default_scores, query_ious, query_valid,
                               threshold, graph_zero):
    positive = query_valid & (query_ious > threshold)
    negative = query_valid & ~positive
    positive_per_row = positive.sum(dim=1)
    negative_per_row = negative.sum(dim=1)
    no_positive = positive_per_row == 0
    too_few_negative = (~no_positive) & (negative_per_row < _TOPK)
    informative = (~no_positive) & (negative_per_row >= _TOPK)

    if bool(informative.any().item()):
        working_scores = default_scores.to(torch.float64)
        positive_scores = working_scores.masked_fill(
            ~positive, -float("inf")
        )
        negative_scores = working_scores.masked_fill(
            ~negative, -float("inf")
        )
        best_positive = positive_scores.max(dim=1).values
        eighth_negative = negative_scores.topk(
            _TOPK, dim=1
        ).values[:, -1]
        informative_gaps = _finite_float64(
            best_positive[informative] - eighth_negative[informative],
        )
        row_losses = _finite_float64(
            F.softplus(_finite_float64(-informative_gaps))
        )
        informative_count = informative.sum().to(torch.float64)
        loss_working = _finite_float64(
            (row_losses / informative_count).sum(),
        )
        loss = _cast_saturated(loss_working, default_scores.dtype)
        active_violations = (
            best_positive[informative] <= eighth_negative[informative]
        ).sum()
        mean_gap = _cast_saturated(_finite_float64(
            (informative_gaps / informative_count).sum(),
        ), default_scores.dtype)
    else:
        loss = graph_zero
        loss_working = graph_zero.to(torch.float64)
        active_violations = torch.zeros(
            (), dtype=torch.long, device=default_scores.device
        )
        mean_gap = default_scores.new_zeros(())

    stats = {
        "informative_rows": informative.sum().detach(),
        "active_violations": active_violations.detach(),
        "no_positive_rows": no_positive.sum().detach(),
        "too_few_negative_rows": too_few_negative.sum().detach(),
        "positive_count": positive_per_row.sum().detach(),
        "mean_positive_cutoff_gap": mean_gap.detach(),
    }
    return loss, loss_working, stats


def compute_rec_source_gate_loss(
        default_scores, query_ious, query_valid, topk=8,
        thresholds=(0.25, 0.50), threshold_weights=(2.0, 1.0),
        margin=0.0, temperature=1.0):
    """Compute the fixed two-threshold strict Top-8 membership loss."""
    _validate_production_contract(
        topk, thresholds, threshold_weights, margin, temperature
    )
    _validate_loss_tensors(default_scores, query_ious, query_valid)

    graph_zero = default_scores.masked_select(query_valid)[0] * 0.0
    loss025, loss025_working, stats025 = _threshold_membership_loss(
        default_scores, query_ious, query_valid, 0.25, graph_zero
    )
    loss050, loss050_working, stats050 = _threshold_membership_loss(
        default_scores, query_ious, query_valid, 0.50, graph_zero
    )
    weighted025 = _finite_float64(loss025_working * 2.0)
    loss_total = _cast_saturated(
        _finite_float64(weighted025 + loss050_working),
        default_scores.dtype,
    )
    stats = {
        "loss025": loss025.detach(),
        "loss050": loss050.detach(),
        "loss_total": loss_total.detach(),
        "threshold025": stats025,
        "threshold050": stats050,
    }
    return loss_total, stats


_CALIBRATION_SCHEMA = "rec-source-gate-calibration-v1"
_CALIBRATION_DIGEST_FORMAT = (
    "rec-source-gate-calibration-float32-sha256-v1"
)
_CALIBRATION_OBSERVATION_FIELDS = frozenset((
    "full_query_ious",
    "default_scores",
    "contrastive_scores",
    "compact_query_indices",
    "compact_valid_mask",
    "parent_candidate_ious",
    "parent_valid_mask",
    "parent_top1_positions",
    "geometry_candidate_ious",
    "geometry_valid_mask",
    "geometry_selected_ious",
))
_CALIBRATION_FLOAT_FIELDS = (
    "full_query_ious",
    "default_scores",
    "contrastive_scores",
    "parent_candidate_ious",
    "geometry_candidate_ious",
    "geometry_selected_ious",
)
_CALIBRATION_IOU_FIELDS = (
    "full_query_ious",
    "parent_candidate_ious",
    "geometry_candidate_ious",
    "geometry_selected_ious",
)
_CALIBRATION_BRANCH_NAMES = (
    "raw_query",
    "default_top8",
    "contrastive_top8",
    "union_query",
    "parent_candidate",
    "geometry_candidate",
    "default_top1",
    "parent_top1",
    "geometry_top1",
)
_CALIBRATION_REPORT_GROUPS = (
    (
        "membership",
        (
            ("default_top8", "default_top8"),
            ("contrastive_top8", "contrastive_top8"),
            ("union_top16", "union_query"),
        ),
    ),
    (
        "candidate_oracle",
        (
            ("raw_query", "raw_query"),
            ("union_query", "union_query"),
            ("parent_candidate", "parent_candidate"),
            ("geometry_candidate", "geometry_candidate"),
        ),
    ),
    (
        "top1",
        (
            ("default", "default_top1"),
            ("parent", "parent_top1"),
            ("geometry", "geometry_top1"),
        ),
    ),
)
_CALIBRATION_INDEX_DTYPES = frozenset((
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
))


def _ordered_calibration_indices(indices, name):
    if isinstance(indices, torch.Tensor):
        if (indices.dim() != 1
                or indices.dtype not in _CALIBRATION_INDEX_DTYPES
                or indices.device.type == "meta"):
            raise ValueError(
                "{} must be a one-dimensional integer sequence".format(name)
            )
        values = tuple(indices.detach().cpu().tolist())
    elif type(indices) in (list, tuple):
        values = tuple(indices)
    else:
        raise ValueError("{} must be an ordered index sequence".format(name))
    if not values:
        raise ValueError("{} cannot be empty".format(name))
    if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for value in values):
        raise ValueError(
            "{} contains an invalid dataset index".format(name)
        )
    if len(set(values)) != len(values):
        raise ValueError(
            "{} contains duplicate dataset indices".format(name)
        )
    return values


def _validate_calibration_tensor_contract(observation, batch_size, layout):
    if (type(observation) is not dict
            or set(observation) != _CALIBRATION_OBSERVATION_FIELDS):
        raise ValueError(
            "source-gate calibration observation fields do not match schema"
        )
    if any(
            not isinstance(value, torch.Tensor)
            for value in observation.values()):
        raise ValueError(
            "source-gate calibration observation values must be tensors"
        )

    full_ious = observation["full_query_ious"]
    default_scores = observation["default_scores"]
    contrastive_scores = observation["contrastive_scores"]
    compact_indices = observation["compact_query_indices"]
    compact_valid = observation["compact_valid_mask"]
    parent_ious = observation["parent_candidate_ious"]
    parent_valid = observation["parent_valid_mask"]
    parent_top1 = observation["parent_top1_positions"]
    geometry_ious = observation["geometry_candidate_ious"]
    geometry_valid = observation["geometry_valid_mask"]
    geometry_selected = observation["geometry_selected_ious"]

    if (full_ious.dim() != 2
            or full_ious.shape[0] != batch_size
            or full_ious.shape[1] == 0):
        raise ValueError("full_query_ious must have non-empty shape [B,Q]")
    if (default_scores.shape != full_ious.shape
            or default_scores.dim() != 2
            or contrastive_scores.shape != full_ious.shape
            or contrastive_scores.dim() != 2):
        raise ValueError(
            "source score shapes must match full_query_ious [B,Q]"
        )
    if (compact_indices.dim() != 2
            or compact_indices.shape != (batch_size, 16)
            or compact_valid.shape != compact_indices.shape
            or compact_valid.dim() != 2):
        raise ValueError(
            "compact query tensors must have shape [B,16]"
        )
    if (parent_ious.dim() != 2
            or parent_ious.shape != compact_indices.shape
            or parent_valid.dim() != 2
            or parent_valid.shape != compact_indices.shape):
        raise ValueError(
            "parent candidate tensors must match compact shape [B,16]"
        )
    if parent_top1.dim() != 1 or parent_top1.shape[0] != batch_size:
        raise ValueError("parent_top1_positions must have shape [B]")
    if (geometry_ious.dim() != 2
            or geometry_ious.shape[0] != batch_size
            or geometry_ious.shape[1] == 0
            or geometry_valid.dim() != 2
            or geometry_valid.shape != geometry_ious.shape):
        raise ValueError(
            "geometry candidate tensors must have non-empty shape [B,G]"
        )
    if (geometry_selected.dim() != 1
            or geometry_selected.shape[0] != batch_size):
        raise ValueError("geometry_selected_ious must have shape [B]")

    if any(
            observation[name].dtype != torch.float32
            for name in _CALIBRATION_FLOAT_FIELDS):
        raise ValueError(
            "source-gate calibration floating tensors must use float32"
        )
    if compact_indices.dtype != torch.long:
        raise ValueError("compact_query_indices must use long dtype")
    if parent_top1.dtype != torch.long:
        raise ValueError("parent_top1_positions must use long dtype")
    for name in (
            "compact_valid_mask", "parent_valid_mask",
            "geometry_valid_mask"):
        if observation[name].dtype != torch.bool:
            raise ValueError("{} must use bool dtype".format(name))

    devices = {value.device for value in observation.values()}
    if len(devices) != 1:
        raise ValueError(
            "source-gate calibration tensors must share a device"
        )
    device = next(iter(devices))
    if device.type == "meta":
        raise ValueError("source-gate calibration tensors need real storage")

    current_layout = (
        int(full_ious.shape[1]),
        int(compact_indices.shape[1]),
        int(geometry_ious.shape[1]),
        device.type,
        device.index,
    )
    if layout is not None and current_layout != layout:
        raise ValueError(
            "source-gate calibration tensor layout changed between batches"
        )

    with torch.no_grad():
        for name in _CALIBRATION_FLOAT_FIELDS:
            if not bool(torch.isfinite(observation[name]).all().item()):
                raise ValueError("{} must be finite".format(name))
        for name in _CALIBRATION_IOU_FIELDS:
            values = observation[name]
            if bool(((values < 0.0) | (values > 1.0)).any().item()):
                raise ValueError("{} must lie in [0, 1]".format(name))
        if not bool(compact_valid.any(dim=1).all().item()):
            raise ValueError("every compact row needs a valid candidate")
        if not torch.equal(parent_valid, compact_valid):
            raise ValueError(
                "parent_valid_mask must equal compact_valid_mask"
            )
        if not bool(geometry_valid.any(dim=1).all().item()):
            raise ValueError("every geometry row needs a valid candidate")

        query_count = full_ious.shape[1]
        if bool((
                (compact_indices < 0)
                | (compact_indices >= query_count)
        ).any().item()):
            raise ValueError("compact_query_indices contains an invalid index")
        candidate_count = compact_indices.shape[1]
        if bool((
                (parent_top1 < 0)
                | (parent_top1 >= candidate_count)
        ).any().item()):
            raise ValueError("parent_top1_positions contains an invalid index")
        parent_top1_valid = parent_valid.gather(
            1, parent_top1.unsqueeze(1)
        ).squeeze(1)
        if not bool(parent_top1_valid.all().item()):
            raise ValueError(
                "parent_top1_positions must point to valid candidates"
            )

    return current_layout


def _masked_calibration_max(values, valid_mask):
    return values.masked_fill(~valid_mask, -float("inf")).max(
        dim=1
    ).values


def _calibration_query_membership_ious(full_ious, indices, valid_mask):
    gathered = full_ious.gather(1, indices)
    return _masked_calibration_max(gathered, valid_mask)


def _stable_calibration_top8(scores):
    return select_candidate_indices(
        scores,
        scores,
        topk_per_source=8,
        max_candidates=8,
    )


def _calibration_float32_bytes(value):
    bits = value.detach().to(device="cpu").contiguous().view(
        torch.int32
    ).reshape(-1).tolist()
    return b"".join(
        struct.pack("<I", int(item) & 0xFFFFFFFF) for item in bits
    )


def _update_calibration_digest_frame(digest, name, payload):
    name = name if isinstance(name, bytes) else name.encode("ascii")
    digest.update(struct.pack("<Q", len(name)))
    digest.update(name)
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


def _new_calibration_digest(field):
    digest = hashlib.sha256()
    _update_calibration_digest_frame(
        digest, "schema", _CALIBRATION_DIGEST_FORMAT.encode("ascii")
    )
    _update_calibration_digest_frame(
        digest, "field", field.encode("ascii")
    )
    return digest


def _calibration_digest_record(value):
    shape = json.dumps(
        list(value.shape), separators=(",", ":")
    ).encode("ascii")
    return shape, _calibration_float32_bytes(value)


def _append_calibration_digest_record(
        digest, dataset_index, shape, payload):
    _update_calibration_digest_frame(digest, "sample", b"")
    _update_calibration_digest_frame(
        digest, "dataset_index", str(dataset_index).encode("ascii")
    )
    _update_calibration_digest_frame(digest, "shape", shape)
    _update_calibration_digest_frame(digest, "dtype", b"float32")
    _update_calibration_digest_frame(digest, "bytes", payload)


def _prepare_calibration_observation(
        observation, batch_size, layout):
    current_layout = _validate_calibration_tensor_contract(
        observation, batch_size, layout
    )
    values = {
        name: tensor.detach() for name, tensor in observation.items()
    }
    with torch.no_grad():
        expected_indices, expected_valid = select_candidate_indices(
            values["default_scores"],
            values["contrastive_scores"],
            topk_per_source=8,
            max_candidates=16,
        )
        if (not torch.equal(
                values["compact_query_indices"], expected_indices
        ) or not torch.equal(
                values["compact_valid_mask"], expected_valid
        )):
            raise ValueError(
                "compact candidates do not match stable source selection"
            )

        default_top8_indices, default_top8_valid = (
            _stable_calibration_top8(values["default_scores"])
        )
        contrastive_top8_indices, contrastive_top8_valid = (
            _stable_calibration_top8(values["contrastive_scores"])
        )
        full_ious = values["full_query_ious"]
        compact_indices = values["compact_query_indices"]
        compact_valid = values["compact_valid_mask"]
        parent_ious = values["parent_candidate_ious"]
        parent_valid = values["parent_valid_mask"]
        parent_top1 = values["parent_top1_positions"]
        geometry_ious = values["geometry_candidate_ious"]
        geometry_valid = values["geometry_valid_mask"]

        branches = {
            "raw_query": full_ious.max(dim=1).values,
            "default_top8": _calibration_query_membership_ious(
                full_ious, default_top8_indices, default_top8_valid
            ),
            "contrastive_top8": _calibration_query_membership_ious(
                full_ious,
                contrastive_top8_indices,
                contrastive_top8_valid,
            ),
            "union_query": _calibration_query_membership_ious(
                full_ious, compact_indices, compact_valid
            ),
            "parent_candidate": _masked_calibration_max(
                parent_ious, parent_valid
            ),
            "geometry_candidate": _masked_calibration_max(
                geometry_ious, geometry_valid
            ),
            "default_top1": full_ious.gather(
                1, default_top8_indices[:, :1]
            ).squeeze(1),
            "parent_top1": parent_ious.gather(
                1, parent_top1.unsqueeze(1)
            ).squeeze(1),
            "geometry_top1": values["geometry_selected_ious"],
        }
        hit_bits = {
            name: {
                "025": tuple(
                    bool(item)
                    for item in (branch > 0.25).cpu().tolist()
                ),
                "050": tuple(
                    bool(item)
                    for item in (branch > 0.50).cpu().tolist()
                ),
            }
            for name, branch in branches.items()
        }
        raw_records = tuple(
            _calibration_digest_record(full_ious[index])
            for index in range(batch_size)
        )
        selected = values["geometry_selected_ious"]
        selected_records = tuple(
            _calibration_digest_record(selected[index])
            for index in range(batch_size)
        )
    return current_layout, hit_bits, raw_records, selected_records


def _calibration_metric(bits025, bits050):
    sample_count = len(bits025)
    hits025 = sum(bits025)
    hits050 = sum(bits050)
    return {
        "hits025": hits025,
        "hits050": hits050,
        "acc025": hits025 / float(sample_count),
        "acc050": hits050 / float(sample_count),
    }


def _calibration_transition(current, baseline):
    if baseline is None:
        return {
            "gained025": 0,
            "lost025": 0,
            "gained050": 0,
            "lost050": 0,
        }
    result = {}
    for suffix in ("025", "050"):
        previous_bits = baseline[suffix]
        current_bits = current[suffix]
        result["gained" + suffix] = sum(
            (not previous) and now
            for previous, now in zip(previous_bits, current_bits)
        )
        result["lost" + suffix] = sum(
            previous and (not now)
            for previous, now in zip(previous_bits, current_bits)
        )
    return result


class RecSourceGateCalibrationAccumulator:
    """Accumulate fixed-order source-gate calibration diagnostics."""

    def __init__(self, expected_indices, baseline=None):
        self._expected_indices = _ordered_calibration_indices(
            expected_indices, "expected_indices"
        )
        self._baseline_hit_bits = None
        if baseline is not None:
            if not isinstance(
                    baseline, RecSourceGateCalibrationAccumulator):
                raise ValueError(
                    "baseline must be a source-gate calibration accumulator"
                )
            if not baseline._finalized:
                raise ValueError("baseline must be finalized")
            if baseline._expected_indices != self._expected_indices:
                raise ValueError(
                    "baseline expected_indices must exactly match"
                )
            self._baseline_hit_bits = {
                name: {
                    suffix: tuple(baseline._hit_bits[name][suffix])
                    for suffix in ("025", "050")
                }
                for name in _CALIBRATION_BRANCH_NAMES
            }

        self._cursor = 0
        self._layout = None
        self._hit_bits = {
            name: {"025": [], "050": []}
            for name in _CALIBRATION_BRANCH_NAMES
        }
        self._raw_digest = _new_calibration_digest("raw_query_ious")
        self._selected_digest = _new_calibration_digest(
            "geometry_selected_ious"
        )
        self._finalized = False
        self._report = None

    @property
    def expected_indices(self):
        return self._expected_indices

    def update(self, dataset_indices, observation):
        if self._finalized:
            raise RuntimeError("source-gate calibration is finalized")
        batch_indices = _ordered_calibration_indices(
            dataset_indices, "dataset_indices"
        )
        expected = self._expected_indices[
            self._cursor:self._cursor + len(batch_indices)
        ]
        if batch_indices != expected:
            raise ValueError(
                "source-gate calibration dataset indices are out of order"
            )

        (current_layout, hit_bits, raw_records, selected_records) = (
            _prepare_calibration_observation(
                observation, len(batch_indices), self._layout
            )
        )
        for name in _CALIBRATION_BRANCH_NAMES:
            for suffix in ("025", "050"):
                self._hit_bits[name][suffix].extend(
                    hit_bits[name][suffix]
                )
        for offset, dataset_index in enumerate(batch_indices):
            raw_shape, raw_payload = raw_records[offset]
            _append_calibration_digest_record(
                self._raw_digest,
                dataset_index,
                raw_shape,
                raw_payload,
            )
            selected_shape, selected_payload = selected_records[offset]
            _append_calibration_digest_record(
                self._selected_digest,
                dataset_index,
                selected_shape,
                selected_payload,
            )
        self._layout = current_layout
        self._cursor += len(batch_indices)

    def finalize(self, expected_sample_count):
        if (type(expected_sample_count) is not int
                or expected_sample_count <= 0
                or expected_sample_count != len(self._expected_indices)):
            raise ValueError(
                "expected_sample_count must be the exact positive count"
            )
        if self._cursor != expected_sample_count:
            raise ValueError("source-gate calibration pass is incomplete")
        if self._finalized:
            return copy.deepcopy(self._report)

        frozen_bits = {
            name: {
                suffix: tuple(self._hit_bits[name][suffix])
                for suffix in ("025", "050")
            }
            for name in _CALIBRATION_BRANCH_NAMES
        }
        metrics = {}
        transitions = {}
        for group_name, branch_specs in _CALIBRATION_REPORT_GROUPS:
            metrics[group_name] = {}
            transitions[group_name] = {}
            for public_name, internal_name in branch_specs:
                current = frozen_bits[internal_name]
                metrics[group_name][public_name] = _calibration_metric(
                    current["025"], current["050"]
                )
                baseline = None
                if self._baseline_hit_bits is not None:
                    baseline = self._baseline_hit_bits[internal_name]
                transitions[group_name][public_name] = (
                    _calibration_transition(current, baseline)
                )

        self._hit_bits = frozen_bits
        self._report = {
            "schema": _CALIBRATION_SCHEMA,
            "sample_count": expected_sample_count,
            "baseline_present": self._baseline_hit_bits is not None,
            "metrics": metrics,
            "transitions": transitions,
            "digests": {
                "canonical_format": _CALIBRATION_DIGEST_FORMAT,
                "raw_query_ious_sha256": self._raw_digest.hexdigest(),
                "geometry_selected_ious_sha256": (
                    self._selected_digest.hexdigest()
                ),
            },
        }
        self._finalized = True
        return copy.deepcopy(self._report)


def _require_source_gate_models(mcln, parent, geometry):
    for name, model in (
            ("mcln", mcln),
            ("parent", parent),
            ("geometry", geometry)):
        if not isinstance(model, torch.nn.Module):
            raise ValueError("{} must be a Module".format(name))


def _freeze_source_gate_models(mcln, parent, geometry):
    seen_modules = set()
    seen_parameters = set()
    stack = [
        model for model in (mcln, parent, geometry)
        if isinstance(model, torch.nn.Module)
    ]
    while stack:
        module = stack.pop()
        if id(module) in seen_modules:
            continue
        seen_modules.add(id(module))
        for parameter in module._parameters.values():
            if parameter is None or id(parameter) in seen_parameters:
                continue
            seen_parameters.add(id(parameter))
            parameter.requires_grad_(False)
        stack.extend(
            child for child in module._modules.values()
            if child is not None
        )


def _eval_source_gate_models(mcln, parent, geometry):
    seen = set()
    stack = [
        model for model in (mcln, parent, geometry)
        if isinstance(model, torch.nn.Module)
    ]
    while stack:
        module = stack.pop()
        if id(module) in seen:
            continue
        seen.add(id(module))
        module.training = False
        stack.extend(
            child for child in module._modules.values()
            if child is not None
        )


def _close_source_gate_models(mcln, parent, geometry):
    errors = []
    for action in (_freeze_source_gate_models, _eval_source_gate_models):
        try:
            action(mcln, parent, geometry)
        except Exception as error:
            errors.append(error)
    if errors:
        raise RuntimeError("source-gate model cleanup failed") from errors[0]


def _source_gate_registered_paths(root):
    """Return every registered module, parameter, and buffer path."""
    module_paths = []
    parameter_paths = []
    buffer_paths = []
    stack = [(root, "", ())]
    while stack:
        module, prefix, ancestor_ids = stack.pop()
        if id(module) in ancestor_ids:
            raise ValueError(
                "cyclic module registration at {}".format(
                    prefix or "<root>"
                )
            )
        module_paths.append((prefix, module))
        next_ancestors = ancestor_ids + (id(module),)
        for name, parameter in module._parameters.items():
            if parameter is not None:
                full_name = "{}.{}".format(prefix, name) if prefix else name
                parameter_paths.append((full_name, parameter))
        for name, buffer in module._buffers.items():
            if buffer is not None:
                full_name = "{}.{}".format(prefix, name) if prefix else name
                buffer_paths.append((full_name, buffer))
        children = tuple(module._modules.items())
        for name, child in reversed(children):
            if child is None:
                continue
            full_name = "{}.{}".format(prefix, name) if prefix else name
            stack.append((child, full_name, next_ancestors))

    return (
        tuple(module_paths),
        tuple(parameter_paths),
        tuple(buffer_paths),
    )


def _validate_source_gate_model_graphs(mcln, parent, geometry):
    paths = {}
    for model_name, model in (
            ("mcln", mcln),
            ("parent", parent),
            ("geometry", geometry)):
        try:
            paths[model_name] = _source_gate_registered_paths(model)
        except ValueError as error:
            raise ValueError("{}: {}".format(model_name, error))
    return paths


def _source_gate_storage_key(name, tensor):
    if tensor.numel() == 0:
        return None
    if tensor.layout != torch.strided:
        raise ValueError(
            "cannot verify storage for non-strided tensor {}".format(name)
        )
    try:
        pointer = int(tensor.storage().data_ptr())
    except Exception as error:
        raise ValueError(
            "cannot verify storage for tensor {}".format(name)
        ) from error
    if pointer == 0:
        raise ValueError(
            "cannot verify storage for tensor {}".format(name)
        )
    return (tensor.device.type, tensor.device.index, pointer)


def _validate_source_gate_tensor_isolation(
        mcln, parent, geometry, selected):
    selected_by_id = {
        id(parameter): name for name, parameter in selected
    }
    selected_storage = {}
    for name, parameter in selected:
        storage_key = _source_gate_storage_key(name, parameter)
        if storage_key is None:
            continue
        if storage_key in selected_storage:
            raise ValueError(
                "selected parameter storage overlap: {}, {}".format(
                    selected_storage[storage_key], name
                )
            )
        selected_storage[storage_key] = name

    graph_paths = _validate_source_gate_model_graphs(
        mcln, parent, geometry
    )
    for model_name, (_modules, parameter_paths, buffer_paths) in (
            graph_paths.items()):
        for kind, tensor_paths in (
                ("parameter", parameter_paths),
                ("buffer", buffer_paths)):
            for name, tensor in tensor_paths:
                qualified_name = "{}.{}".format(model_name, name)
                legitimate_selected = (
                    model_name == "mcln"
                    and kind == "parameter"
                    and _is_source_gate_parameter_name(name)
                    and id(tensor) in selected_by_id
                )
                if legitimate_selected:
                    continue
                if id(tensor) in selected_by_id:
                    raise ValueError(
                        "source-gate tensor identity overlap: {} and {}".format(
                            selected_by_id[id(tensor)], qualified_name
                        )
                    )
                storage_key = _source_gate_storage_key(
                    qualified_name, tensor
                )
                if (storage_key is not None
                        and storage_key in selected_storage):
                    raise ValueError(
                        "source-gate storage overlap: {} and {}".format(
                            selected_storage[storage_key], qualified_name
                        )
                    )


def _is_source_gate_parameter_name(name):
    return name.startswith(SOURCE_GATE_TRAINABLE_PREFIX)


def _is_source_gate_module_name(name):
    return (
        name == _SOURCE_GATE_TRAINABLE_MODULE
        or name.startswith(_SOURCE_GATE_TRAINABLE_MODULE + ".")
    )


def _reject_source_gate_cross_boundary_aliases(
        paths, is_allowed, kind, boundary, error_type=ValueError):
    names_by_identity = {}
    for name, value in paths:
        names_by_identity.setdefault(id(value), []).append(name)
    for names in names_by_identity.values():
        if len({is_allowed(name) for name in names}) > 1:
            display_names = tuple(name or "<root>" for name in names)
            raise error_type(
                "{} aliases cross the source-gate {} boundary: {}".format(
                    kind, boundary, ", ".join(display_names)
                )
            )


def _selected_source_gate_parameters(mcln):
    return tuple(
        (name, parameter)
        for name, parameter in mcln.named_parameters()
        if _is_source_gate_parameter_name(name)
    )


def _validate_source_gate_aliases(mcln, module_error_type=ValueError):
    module_paths, parameter_paths, _buffer_paths = (
        _source_gate_registered_paths(mcln)
    )
    _reject_source_gate_cross_boundary_aliases(
        module_paths,
        _is_source_gate_module_name,
        "module",
        "train/eval",
        error_type=module_error_type,
    )
    _reject_source_gate_cross_boundary_aliases(
        parameter_paths,
        _is_source_gate_parameter_name,
        "parameter",
        "allowlist",
    )
    selected_names_by_identity = {}
    for name, parameter in parameter_paths:
        if _is_source_gate_parameter_name(name):
            selected_names_by_identity.setdefault(id(parameter), []).append(
                name
            )
    for names in selected_names_by_identity.values():
        if len(names) > 1:
            raise ValueError(
                "source-gate selected parameter has duplicate registrations: "
                + ", ".join(names)
            )
    return module_paths, parameter_paths


def _validate_source_gate_trainability(mcln, parent, geometry):
    selected = _selected_source_gate_parameters(mcln)
    if not selected:
        raise ValueError(
            "source-gate trainable prefix matches no parameters: {}".format(
                SOURCE_GATE_TRAINABLE_PREFIX
            )
        )
    for name, parameter in mcln.named_parameters():
        expected = _is_source_gate_parameter_name(name)
        if parameter.requires_grad is not expected:
            raise RuntimeError(
                "source-gate trainability does not match the allowlist"
            )
    if any(
            parameter.requires_grad
            for model in (parent, geometry)
            for parameter in model.parameters()):
        raise RuntimeError(
            "source-gate reranker trainability does not match the allowlist"
        )
    return selected


def _reject_frozen_source_gate_gradients(mcln, parent, geometry):
    for name, parameter in mcln.named_parameters():
        if (not _is_source_gate_parameter_name(name)
                and parameter.grad is not None):
            raise RuntimeError(
                "frozen MCLN parameter {} has a gradient".format(name)
            )
    for model_name, model in (("parent", parent), ("geometry", geometry)):
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                raise RuntimeError(
                    "frozen {} parameter {} has a gradient".format(
                        model_name, name
                    )
                )


def _validated_source_gate_contract(parameters):
    if (type(parameters) is not _RecSourceGateParameterContract
            or parameters._seal is not _SOURCE_GATE_CONTRACT_SEAL):
        raise ValueError("source-gate parameter contract schema is invalid")

    try:
        provenance = _SOURCE_GATE_CONTRACT_PROVENANCE[parameters]
    except KeyError:
        raise ValueError("source-gate parameter contract has no provenance")

    original_names, original_parameters, original_models = provenance
    if (parameters.names is not original_names
            or parameters.parameters is not original_parameters
            or parameters.mcln is not original_models[0]
            or parameters.parent is not original_models[1]
            or parameters.geometry is not original_models[2]):
        raise ValueError("source-gate parameter contract provenance changed")

    names = original_names
    selected_parameters = original_parameters
    mcln, parent, geometry = original_models
    _require_source_gate_models(mcln, parent, geometry)

    if type(names) is not tuple or not names:
        raise ValueError("source-gate names must be a non-empty tuple")
    if type(selected_parameters) is not tuple or not selected_parameters:
        raise ValueError("source-gate parameters must be a non-empty tuple")
    if len(names) != len(selected_parameters):
        raise ValueError("source-gate names and parameters must align")
    if not all(isinstance(name, str) for name in names):
        raise ValueError("source-gate names must contain strings")
    if not all(
            isinstance(parameter, torch.nn.Parameter)
            for parameter in selected_parameters):
        raise ValueError("source-gate parameters must contain Parameters")
    if len(set(names)) != len(names):
        raise ValueError("source-gate selected names contain duplicates")
    selected_ids = tuple(id(parameter) for parameter in selected_parameters)
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("source-gate selected parameters contain duplicates")

    _validate_source_gate_aliases(mcln)
    expected = _selected_source_gate_parameters(mcln)
    expected_names = tuple(name for name, _parameter in expected)
    expected_ids = tuple(id(parameter) for _name, parameter in expected)
    if names != expected_names or selected_ids != expected_ids:
        raise ValueError(
            "source-gate selected group does not exactly match the allowlist"
        )
    _validate_source_gate_tensor_isolation(
        mcln, parent, geometry, expected
    )
    _validate_source_gate_trainability(mcln, parent, geometry)
    _reject_frozen_source_gate_gradients(mcln, parent, geometry)
    return tuple(parameter for _name, parameter in expected)


def configure_rec_source_gate_trainability(mcln, parent, geometry):
    """Enable only the final semantic classifier and return its contract."""
    try:
        _require_source_gate_models(mcln, parent, geometry)
        _validate_source_gate_model_graphs(mcln, parent, geometry)
        _close_source_gate_models(mcln, parent, geometry)
        _validate_source_gate_aliases(mcln)
        selected = _selected_source_gate_parameters(mcln)
        if not selected:
            raise ValueError(
                "source-gate trainable prefix matches no parameters: {}".format(
                    SOURCE_GATE_TRAINABLE_PREFIX
                )
            )
        selected_ids = tuple(id(parameter) for _name, parameter in selected)
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError(
                "source-gate selected parameters contain duplicates"
            )
        _validate_source_gate_tensor_isolation(
            mcln, parent, geometry, selected
        )
        _reject_frozen_source_gate_gradients(mcln, parent, geometry)
        for _name, parameter in selected:
            parameter.requires_grad_(True)
        _validate_source_gate_trainability(mcln, parent, geometry)
    except Exception:
        _close_source_gate_models(mcln, parent, geometry)
        raise

    contract = _RecSourceGateParameterContract(
        tuple(name for name, _parameter in selected),
        tuple(parameter for _name, parameter in selected),
        mcln,
        parent,
        geometry,
        _SOURCE_GATE_CONTRACT_SEAL,
    )
    _SOURCE_GATE_CONTRACT_PROVENANCE[contract] = (
        contract.names,
        contract.parameters,
        (mcln, parent, geometry),
    )
    return contract


def set_rec_source_gate_train_mode(mcln, parent, geometry):
    """Keep all roots in eval and train only the allowed classifier subtree."""
    try:
        _require_source_gate_models(mcln, parent, geometry)
        graph_paths = _validate_source_gate_model_graphs(
            mcln, parent, geometry
        )
        _eval_source_gate_models(mcln, parent, geometry)
        module_paths, _parameter_paths = _validate_source_gate_aliases(
            mcln, module_error_type=AssertionError
        )
        selected = _validate_source_gate_trainability(
            mcln, parent, geometry
        )
        _validate_source_gate_tensor_isolation(
            mcln, parent, geometry, selected
        )
        matching_modules = tuple(
            module for name, module in module_paths
            if name == _SOURCE_GATE_TRAINABLE_MODULE
        )
        if len(matching_modules) != 1:
            raise ValueError(
                "MCLN is missing source-gate trainable module {}".format(
                    _SOURCE_GATE_TRAINABLE_MODULE
                )
            )
        matching_modules[0].train()
        for name, module in module_paths:
            expected = _is_source_gate_module_name(name)
            if module.training is not expected:
                raise AssertionError(
                    "source-gate module {} has an invalid mode".format(
                        name or "<root>"
                    )
                )
        if any(
                module.training
                for model_name in ("parent", "geometry")
                for _name, module in graph_paths[model_name][0]):
            raise AssertionError("source-gate rerankers entered train mode")
    except Exception:
        _close_source_gate_models(mcln, parent, geometry)
        raise


def set_rec_source_gate_eval_mode(mcln, parent, geometry):
    """Put the MCLN and both frozen rerankers fully in evaluation mode."""
    try:
        _require_source_gate_models(mcln, parent, geometry)
        _validate_source_gate_model_graphs(mcln, parent, geometry)
        _eval_source_gate_models(mcln, parent, geometry)
    except Exception:
        _close_source_gate_models(mcln, parent, geometry)
        raise


def _validate_exact_positive_hyperparameter(name, value, expected):
    if (not _is_finite_real(value)
            or float(value) <= 0.0
            or float(value) != expected):
        raise ValueError("{} must be exactly {}".format(name, expected))


def build_rec_source_gate_optimizer(
        parameters, lr=1e-4, weight_decay=1e-4):
    """Build a fresh one-group constant-rate AdamW for the allowlist."""
    _validate_exact_positive_hyperparameter("lr", lr, _SOURCE_GATE_LR)
    _validate_exact_positive_hyperparameter(
        "weight_decay", weight_decay, _SOURCE_GATE_WEIGHT_DECAY
    )
    selected_parameters = _validated_source_gate_contract(parameters)
    group = {
        "name": _SOURCE_GATE_GROUP_NAME,
        "params": selected_parameters,
        "lr": _SOURCE_GATE_LR,
        "weight_decay": _SOURCE_GATE_WEIGHT_DECAY,
    }
    return torch.optim.AdamW(
        [group],
        lr=_SOURCE_GATE_LR,
        weight_decay=_SOURCE_GATE_WEIGHT_DECAY,
    )


def _reject_nonfinite_selected_gradients(parameters):
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        values = gradient.coalesce().values() if gradient.is_sparse else gradient
        if not bool(torch.isfinite(values).all().item()):
            raise FloatingPointError(
                "source-gate selected gradient is non-finite"
            )


def clip_rec_source_gate_gradients(parameters, max_norm=1.0):
    """Validate and clip only the exact source-gate parameter allowlist."""
    _validate_exact_positive_hyperparameter(
        "max_norm", max_norm, _SOURCE_GATE_MAX_NORM
    )
    selected_parameters = _validated_source_gate_contract(parameters)
    _reject_nonfinite_selected_gradients(selected_parameters)
    total_norm = torch.nn.utils.clip_grad_norm_(
        selected_parameters, _SOURCE_GATE_MAX_NORM
    )
    norm = float(total_norm)
    if not math.isfinite(norm):
        raise FloatingPointError(
            "source-gate gradient norm is non-finite"
        )
    _reject_nonfinite_selected_gradients(selected_parameters)
    return {_SOURCE_GATE_GROUP_NAME: norm}
