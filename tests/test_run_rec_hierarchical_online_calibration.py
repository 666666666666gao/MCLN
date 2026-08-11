import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models import rec_source_gate
import scripts.run_rec_hierarchical_online_calibration as online


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _evidence(hits025, hits050):
    return {
        "sample_count": 3625,
        "hits025": hits025,
        "hits050": hits050,
        "oracle_hits025": 3606,
        "oracle_hits050": 3588,
        "raw_query_hits025": 3615,
        "raw_query_hits050": 3559,
        "candidate_iou_sha256": "1" * 64,
        "raw_query_iou_sha256": (
            "7a75f033a7afb2b1871e971b2797e544b"
            "ac8bb59e6a20aa68735eb23842d5751"
        ),
        "row_materialization_sha256": "2" * 64,
        "selected_iou_sha256": online.ONLINE_BASELINE_SELECTED_IOU_SHA256,
    }


def _snapshot(path, sha256):
    return {
        "path": str(Path(path).absolute()),
        "device": 1,
        "inode": 2,
        "mode": 0o444,
        "size": 3,
        "mtime_ns": 4,
        "ctime_ns": 5,
        "sha256": sha256,
    }


def _provenance(staged_path, staged_sha256):
    protected = {
        "backbone": _snapshot("/frozen/backbone.pth", "a" * 64),
        "parent": _snapshot("/frozen/parent.pth", "b" * 64),
        "geometry": _snapshot("/frozen/geometry.pth", "c" * 64),
        "staged_hierarchical": _snapshot(staged_path, staged_sha256),
        "staged_result_receipt": _snapshot(
            "/frozen/staged-result-receipt.json", "9" * 64
        ),
        "source_gate_baseline": _snapshot(
            "/frozen/source-gate.json", "d" * 64
        ),
    }
    files = {
        "online_runner": {
            "path": str(Path(online.__file__).absolute()),
            "sha256": "e" * 64,
        },
        "hierarchical_trainer": {
            "path": str(Path(online.__file__).absolute()),
            "sha256": "f" * 64,
        },
    }
    code_sha = hashlib.sha256(json.dumps(
        files, sort_keys=True, separators=(",", ":")
    ).encode("ascii")).hexdigest()
    return {
        "command": ["python", str(Path(online.__file__).absolute())],
        "environment": {
            "device": "cuda:0",
            "CUDA_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": "1",
            "PYTHONPATH": "/repo:/repo/pointnet2",
            "python": "/python",
        },
        "code": {"files": files, "sha256": code_sha},
        "protected_before": protected,
        "protected_after": copy.deepcopy(protected),
        "source_gate_baseline_receipt": {
            "path": protected["source_gate_baseline"]["path"],
            "sha256": protected["source_gate_baseline"]["sha256"],
        },
        "staged_result_receipt": {
            "path": protected["staged_result_receipt"]["path"],
            "sha256": protected["staged_result_receipt"]["sha256"],
        },
    }


def _record(staged_path, staged_sha256, candidate_hits025=3524,
            candidate_hits050=3316):
    baseline = _evidence(3461, 3316)
    candidate = _evidence(candidate_hits025, candidate_hits050)
    candidate["selected_iou_sha256"] = "4" * 64
    return {
        "schema": online.ONLINE_CALIBRATION_SCHEMA,
        "version": online.ONLINE_CALIBRATION_VERSION,
        "sample_count": 3625,
        "baseline": baseline,
        "candidate": candidate,
        "staged_artifact": {
            "path": str(Path(staged_path).absolute()),
            "sha256": staged_sha256,
            "deployable": False,
        },
        "deployed_artifact": None,
        "gate": {
            "passed": True,
            "failures": [],
            "required_hits025": 3524,
            "required_hits050": 3316,
        },
        "provenance": _provenance(staged_path, staged_sha256),
        "validation_data_accessed": False,
        "inference_uses_ground_truth": False,
    }


def test_online_gate_requires_exact_baseline_candidate_and_invariants():
    baseline = _evidence(3461, 3316)
    candidate = _evidence(3524, 3316)
    candidate["selected_iou_sha256"] = "4" * 64

    passed = online.online_calibration_gate(baseline, candidate)

    assert passed.passed is True
    assert passed.failures == ()
    assert passed.required_hits025 == 3524
    assert passed.required_hits050 == 3316

    mutations = {
        "sample_count": 3624,
        "hits025": 3523,
        "hits050": 3315,
        "oracle_hits025": 3605,
        "oracle_hits050": 3587,
        "raw_query_hits025": 3614,
        "raw_query_hits050": 3558,
        "candidate_iou_sha256": "f" * 64,
        "raw_query_iou_sha256": "e" * 64,
        "row_materialization_sha256": "d" * 64,
    }
    for field, value in mutations.items():
        changed = copy.deepcopy(candidate)
        changed[field] = value
        failed = online.online_calibration_gate(baseline, changed)
        assert failed.passed is False
        assert field in failed.failures

    changed_baseline = copy.deepcopy(baseline)
    changed_baseline["hits050"] = 3315
    failed = online.online_calibration_gate(changed_baseline, candidate)
    assert failed.passed is False
    assert "baseline_hits050" in failed.failures


def test_online_accumulator_uses_strict_thresholds_and_ordered_digests():
    accumulator = online.OnlineCalibrationAccumulator((8, 3, 5))
    accumulator.update(
        torch.tensor([8, 3]),
        {
            "baseline_selected_ious": torch.tensor([0.25, 0.5001]),
            "candidate_selected_ious": torch.tensor([0.2501, 0.50]),
            "candidate_ious": torch.tensor([
                [0.25, 0.50, 0.9],
                [0.2501, 0.5001, 0.0],
            ]),
            "candidate_valid": torch.tensor([
                [True, True, True],
                [True, True, False],
            ]),
            "raw_query_ious": torch.tensor([
                [0.25, 0.50, 0.75],
                [0.2501, 0.5001, 0.0],
            ]),
            "row_materialization": torch.tensor([
                [1.0, 2.0],
                [3.0, 4.0],
            ]),
        },
    )
    accumulator.update(
        torch.tensor([5]),
        {
            "baseline_selected_ious": torch.tensor([1.0]),
            "candidate_selected_ious": torch.tensor([0.0]),
            "candidate_ious": torch.tensor([[0.1, 0.2, 0.3]]),
            "candidate_valid": torch.tensor([[True, True, True]]),
            "raw_query_ious": torch.tensor([[0.1, 0.2, 0.3]]),
            "row_materialization": torch.tensor([[5.0, 6.0]]),
        },
    )

    report = accumulator.finalize(3)

    raw_digest = rec_source_gate._new_calibration_digest("raw_query_ious")
    baseline_digest = rec_source_gate._new_calibration_digest(
        "geometry_selected_ious"
    )
    candidate_digest = rec_source_gate._new_calibration_digest(
        "geometry_selected_ious"
    )
    raw_rows = (
        torch.tensor([0.25, 0.50, 0.75]),
        torch.tensor([0.2501, 0.5001, 0.0]),
        torch.tensor([0.1, 0.2, 0.3]),
    )
    baseline_rows = (torch.tensor(0.25), torch.tensor(0.5001), torch.tensor(1.0))
    candidate_rows = (torch.tensor(0.2501), torch.tensor(0.50), torch.tensor(0.0))
    for dataset_index, raw, baseline_iou, candidate_iou in zip(
            (8, 3, 5), raw_rows, baseline_rows, candidate_rows):
        for digest, value in (
                (raw_digest, raw),
                (baseline_digest, baseline_iou),
                (candidate_digest, candidate_iou)):
            shape, payload = rec_source_gate._calibration_digest_record(value)
            rec_source_gate._append_calibration_digest_record(
                digest, dataset_index, shape, payload
            )

    assert report["baseline"]["hits025"] == 2
    assert report["baseline"]["hits050"] == 2
    assert report["candidate"]["hits025"] == 2
    assert report["candidate"]["hits050"] == 0
    assert report["baseline"]["oracle_hits025"] == 3
    assert report["baseline"]["oracle_hits050"] == 2
    assert report["baseline"]["raw_query_hits025"] == 3
    assert report["baseline"]["raw_query_hits050"] == 2
    assert report["baseline"]["raw_query_iou_sha256"] == \
        raw_digest.hexdigest()
    assert report["baseline"]["selected_iou_sha256"] == \
        baseline_digest.hexdigest()
    assert report["candidate"]["selected_iou_sha256"] == \
        candidate_digest.hexdigest()
    for field in online.ONLINE_INVARIANT_FIELDS:
        assert report["baseline"][field] == report["candidate"][field]
    assert report["baseline"]["selected_iou_sha256"] != report[
        "candidate"
    ]["selected_iou_sha256"]

    reordered = online.OnlineCalibrationAccumulator((8, 3, 5))
    with pytest.raises(ValueError, match="order"):
        reordered.update(torch.tensor([3]), {
            "baseline_selected_ious": torch.tensor([1.0]),
            "candidate_selected_ious": torch.tensor([1.0]),
            "candidate_ious": torch.tensor([[1.0]]),
            "candidate_valid": torch.tensor([[True]]),
            "raw_query_ious": torch.tensor([[1.0]]),
            "row_materialization": torch.tensor([[1.0]]),
        })


def test_online_record_binds_exact_staged_sha_and_policy(tmp_path):
    staged = tmp_path / "selected_hierarchical.pth"
    staged.write_bytes(b"staged")
    record = _record(staged, _sha256(staged))

    validated = online.validate_online_calibration_record(
        record, expected_staged_sha256=_sha256(staged)
    )

    assert validated["gate"]["passed"] is True
    for mutation in (
            "sample_count", "staged_sha", "validation", "ground_truth",
            "gate"):
        changed = copy.deepcopy(record)
        if mutation == "sample_count":
            changed["sample_count"] = 3624
        elif mutation == "staged_sha":
            changed["staged_artifact"]["sha256"] = "f" * 64
        elif mutation == "validation":
            changed["validation_data_accessed"] = True
        elif mutation == "ground_truth":
            changed["inference_uses_ground_truth"] = True
        else:
            changed["gate"]["passed"] = False
        with pytest.raises(ValueError):
            online.validate_online_calibration_record(
                changed, expected_staged_sha256=_sha256(staged)
            )


def test_online_record_rejects_provenance_or_protected_input_drift(tmp_path):
    staged = tmp_path / "selected_hierarchical.pth"
    staged.write_bytes(b"staged")
    record = _record(staged, _sha256(staged))

    assert online.validate_online_calibration_record(
        record, expected_staged_sha256=_sha256(staged)
    )["provenance"]["protected_before"] == record[
        "provenance"
    ]["protected_after"]

    mutations = (
        ("command", lambda value: value["command"].append("--official")),
        ("environment", lambda value: value["environment"].update(
            CUDA_VISIBLE_DEVICES="1"
        )),
        ("code", lambda value: value["code"].update(sha256="0" * 64)),
        ("protected", lambda value: value["protected_after"]["geometry"].update(
            sha256="0" * 64
        )),
        ("source", lambda value: value[
            "source_gate_baseline_receipt"
        ].update(sha256="0" * 64)),
        ("staged_receipt", lambda value: value[
            "staged_result_receipt"
        ].update(sha256="0" * 64)),
    )
    for _label, mutate in mutations:
        changed = copy.deepcopy(record)
        mutate(changed["provenance"])
        with pytest.raises(ValueError):
            online.validate_online_calibration_record(
                changed, expected_staged_sha256=_sha256(staged)
            )


def test_live_calibration_builds_both_gt_free_scores_before_attaching_targets():
    events = []
    batch = {
        "dataset_index": torch.tensor([11, 4]),
        "point_clouds": torch.ones(2, 1),
        "center_label": object(),
        "size_gts": object(),
    }
    initialized = {
        "device": "cpu",
        "initial_state": {
            "mcln": object(),
            "parent": object(),
            "parent_artifact": {"name": "parent"},
            "geometry": object(),
            "geometry_artifact": {"name": "geometry"},
        },
        "data": {
            "calibration_view": SimpleNamespace(indices=(11, 4)),
            "calibration_loader": [batch],
        },
        "train_data_contract": {
            "calibration_sample_count": 2,
            "validation_data_accessed": False,
        },
    }

    def move_batch(value, device):
        assert device == torch.device("cpu")
        return value

    def input_builder(value):
        events.append("inputs")
        assert value["center_label"] is batch["center_label"]
        return {"point_clouds": value["point_clouds"]}

    def mcln_forward(inputs):
        events.append("mcln")
        assert "center_label" not in inputs
        return {"last_center": torch.zeros(2, 1, 3)}

    def full_state_builder(end_points, inputs):
        events.append("raw-query-state")
        assert "center_label" not in end_points
        assert "center_label" not in inputs
        return {"deployable": torch.ones(2, 1)}

    def parent_builder(end_points, inputs, parent, artifact):
        events.append("parent")
        assert "center_label" not in end_points
        assert "center_label" not in inputs
        return {"parent": torch.ones(2, 1)}

    boxes = torch.arange(2 * 112 * 6, dtype=torch.float32).reshape(
        2, 112, 6
    )
    valid = torch.ones(2, 112, dtype=torch.bool)
    parent_scores = torch.zeros(2, 256)
    baseline_scores = torch.zeros(2, 112)
    baseline_scores[:, 0] = 1.0
    candidate_scores = baseline_scores.clone()
    candidate_scores[:, 0] = 0.0
    candidate_scores[:, 1] = 2.0

    def geometry_builder(end_points, inputs, parent_outputs, geometry,
                         artifact, hierarchical_model=None,
                         hierarchical_artifact=None):
        assert "center_label" not in end_points
        assert "center_label" not in inputs
        if hierarchical_model is None:
            events.append("baseline-score")
            scores = baseline_scores
        else:
            events.append("candidate-score")
            scores = candidate_scores
        return {
            "rec_reranker_scores": parent_scores,
            "rec_geometry_runtime_mode": "flat_geometry_axis",
            "rec_geometry_boxes": boxes,
            "rec_geometry_scores": scores,
            "rec_geometry_valid_mask": valid,
            "rec_geometry_fallback_index": torch.zeros(
                2, dtype=torch.long
            ),
        }

    def full_target_attacher(full_state, moved, root_only):
        events.append("raw-query-targets")
        assert events.index("raw-query-targets") > events.index(
            "candidate-score"
        )
        assert moved["center_label"] is batch["center_label"]
        assert root_only is True
        return torch.tensor([
            [0.1, 0.6, 0.0],
            [0.1, 0.2, 0.3],
        ])

    def geometry_target_attacher(geometry_batch, moved, root_only):
        events.append("geometry-targets")
        assert events.index("geometry-targets") > events.index(
            "candidate-score"
        )
        assert tuple(geometry_batch["boxes"].shape) == (2, 16, 7, 6)
        candidate_ious = torch.zeros(2, 16, 7)
        candidate_ious[0, 0, 0] = 0.6
        candidate_ious[0, 0, 1] = 0.7
        candidate_ious[1, 0, 0] = 0.4
        candidate_ious[1, 0, 1] = 0.2
        return {"geometry_ious": candidate_ious}

    report = online.evaluate_live_online_calibration(
        initialized,
        hierarchical_model=object(),
        hierarchical_artifact={"name": "hierarchy"},
        move_batch=move_batch,
        input_builder=input_builder,
        mcln_forward=mcln_forward,
        full_state_builder=full_state_builder,
        parent_output_builder=parent_builder,
        geometry_output_builder=geometry_builder,
        full_target_attacher=full_target_attacher,
        geometry_target_attacher=geometry_target_attacher,
    )

    assert events == [
        "inputs", "mcln", "raw-query-state", "parent",
        "baseline-score", "candidate-score", "raw-query-targets",
        "geometry-targets",
    ]
    assert report["baseline"]["sample_count"] == 2
    assert report["baseline"]["hits025"] == 2
    assert report["baseline"]["hits050"] == 1
    assert report["candidate"]["hits025"] == 1
    assert report["candidate"]["hits050"] == 1


def _source_gate_receipt():
    return {
        "calibration": {
            "reproduced": {
                "sample_count": 3625,
                "metrics": {
                    "top1": {
                        "geometry": {"hits025": 3461, "hits050": 3316},
                    },
                    "candidate_oracle": {
                        "geometry_candidate": {
                            "hits025": 3606, "hits050": 3588,
                        },
                        "raw_query": {
                            "hits025": 3615, "hits050": 3559,
                        },
                    },
                },
                "digests": {
                    "raw_query_ious_sha256": online.ONLINE_RAW_QUERY_IOU_SHA256,
                    "geometry_selected_ious_sha256": (
                        online.ONLINE_BASELINE_SELECTED_IOU_SHA256
                    ),
                },
            },
        },
        "validation_data_accessed": False,
        "checkpoint_written": False,
        "deployable": False,
    }


def test_source_gate_baseline_receipt_extracts_only_authoritative_reproduction():
    receipt = _source_gate_receipt()

    evidence = online.validate_source_gate_baseline_evidence(receipt)

    assert evidence == {
        "sample_count": 3625,
        "hits025": 3461,
        "hits050": 3316,
        "oracle_hits025": 3606,
        "oracle_hits050": 3588,
        "raw_query_hits025": 3615,
        "raw_query_hits050": 3559,
        "raw_query_iou_sha256": online.ONLINE_RAW_QUERY_IOU_SHA256,
        "selected_iou_sha256": online.ONLINE_BASELINE_SELECTED_IOU_SHA256,
    }
    for path, value in (
            (("calibration", "reproduced", "metrics", "top1", "geometry",
              "hits050"), 3315),
            (("calibration", "reproduced", "metrics", "candidate_oracle",
              "geometry_candidate", "hits025"), 3605),
            (("calibration", "reproduced", "digests",
              "raw_query_ious_sha256"), "0" * 64),
            (("validation_data_accessed",), True)):
        changed = copy.deepcopy(receipt)
        target = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValueError):
            online.validate_source_gate_baseline_evidence(changed)


def test_initializer_preflights_six_frozen_inputs_and_builds_train_only_data(
        monkeypatch, tmp_path):
    paths = {}
    for name in (
            "backbone", "parent", "geometry", "staged",
            "staged_receipt", "source_receipt"):
        path = tmp_path / (name + (".json" if "receipt" in name else ".pth"))
        path.write_bytes(name.encode("ascii"))
        path.chmod(0o444)
        paths[name] = path
    monkeypatch.setattr(
        online, "AUTHORITATIVE_BACKBONE_SHA256", _sha256(paths["backbone"])
    )
    monkeypatch.setattr(
        online, "AUTHORITATIVE_PARENT_ARTIFACT_SHA256", _sha256(paths["parent"])
    )
    monkeypatch.setattr(
        online, "AUTHORITATIVE_GEOMETRY_ARTIFACT_SHA256", _sha256(paths["geometry"])
    )
    monkeypatch.setattr(
        online, "AUTHORITATIVE_SOURCE_GATE_RECEIPT_SHA256",
        _sha256(paths["source_receipt"]),
    )
    (tmp_path / "train-root").mkdir()
    args = SimpleNamespace(
        data_root=str(tmp_path / "train-root"),
        backbone_checkpoint=str(paths["backbone"]),
        parent_artifact=str(paths["parent"]),
        geometry_artifact=str(paths["geometry"]),
        staged_hierarchical_artifact=str(paths["staged"]),
        staged_result_receipt=str(paths["staged_receipt"]),
        source_gate_baseline_receipt=str(paths["source_receipt"]),
        output_dir=str(tmp_path / "online-output"),
        device="cuda:0",
    )
    staged_receipt = {
        "selected": "staged_hierarchical",
        "deployable": False,
        "validation_data_accessed": False,
        "artifact": {
            "name": paths["staged"].name,
            "sha256": _sha256(paths["staged"]),
        },
        "calibration": {"gate": {"passed": True}},
    }
    source_receipt = _source_gate_receipt()
    config = SimpleNamespace()
    mcln = torch.nn.Linear(1, 1)
    parent = torch.nn.Linear(1, 1)
    geometry = torch.nn.Linear(1, 1)
    hierarchy = torch.nn.Linear(1, 1)
    parent._artifact_sha256 = _sha256(paths["parent"])
    geometry._artifact_sha256 = _sha256(paths["geometry"])
    hierarchy._artifact_sha256 = _sha256(paths["staged"])
    calls = []

    def initial_state_loader(backbone, parent_path, geometry_path, data_root,
                             device):
        calls.append("models")
        assert tuple(map(Path, (backbone, parent_path, geometry_path))) == (
            paths["backbone"].absolute(),
            paths["parent"].absolute(),
            paths["geometry"].absolute(),
        )
        assert Path(data_root) == Path(args.data_root).absolute()
        assert device == "cuda:0"
        return {
            "config": config,
            "mcln": mcln,
            "parent": parent,
            "parent_artifact": {"feature_names": ["parent"]},
            "geometry": geometry,
            "geometry_artifact": {"feature_names": ["geometry"]},
            "groups": object(),
            "optimizer": object(),
        }

    data = {
        "dataset": object(),
        "split": object(),
        "fit_view": object(),
        "calibration_view": SimpleNamespace(indices=tuple(range(3625))),
        "fit_loader": object(),
        "calibration_loader": [],
    }

    def data_builder(received_config, device):
        calls.append("train-data")
        assert received_config is config
        assert device == "cuda:0"
        return data

    def contract_builder(initialized):
        assert initialized["data"] is data
        return {
            "calibration_sample_count": 3625,
            "validation_data_accessed": False,
        }

    def hierarchy_loader(path, **kwargs):
        calls.append("hierarchy")
        assert Path(path) == paths["staged"].absolute()
        assert kwargs["expected_artifact_sha256"] == _sha256(paths["staged"])
        assert kwargs["expected_deployable"] is False
        assert kwargs["parent_sha256"] == _sha256(paths["parent"])
        assert kwargs["geometry_sha256"] == _sha256(paths["geometry"])
        return hierarchy, {"deployable": False}

    code = _provenance(paths["staged"], _sha256(paths["staged"]))["code"]
    environment = _provenance(
        paths["staged"], _sha256(paths["staged"])
    )["environment"]
    before = {name: path.read_bytes() for name, path in paths.items()}

    initialized = online.initialize_live_online_calibration(
        args,
        initial_state_loader=initial_state_loader,
        data_builder=data_builder,
        data_contract_builder=contract_builder,
        hierarchy_loader=hierarchy_loader,
        staged_receipt_loader=lambda _path: staged_receipt,
        staged_receipt_validator=lambda value: value,
        source_receipt_loader=lambda _path: source_receipt,
        code_manifest_builder=lambda: code,
        command_builder=lambda _args: ["python", str(Path(online.__file__).absolute())],
        environment_builder=lambda _args: environment,
    )

    assert calls == ["models", "hierarchy", "train-data"]
    assert initialized["data"] is data
    assert initialized["train_data_contract"]["calibration_sample_count"] == 3625
    assert initialized["source_gate_baseline"]["hits050"] == 3316
    assert set(initialized["provenance"]["protected_before"]) == {
        "backbone", "parent", "geometry", "staged_hierarchical",
        "staged_result_receipt", "source_gate_baseline",
    }
    assert initialized["provenance"]["protected_before"] == \
        initialized["provenance"]["protected_after"]
    assert not Path(args.output_dir).exists()
    assert {name: path.read_bytes() for name, path in paths.items()} == before
    for model in (mcln, parent, geometry, hierarchy):
        assert model.training is False
        assert all(not parameter.requires_grad for parameter in model.parameters())


def test_finalizer_rechecks_all_protected_inputs_after_live_evaluation(tmp_path):
    protected_paths = {}
    for name in (
            "backbone", "parent", "geometry", "staged_hierarchical",
            "staged_result_receipt", "source_gate_baseline"):
        path = tmp_path / (name + ".bin")
        path.write_bytes(name.encode("ascii"))
        path.chmod(0o444)
        protected_paths[name] = path
    before = online._capture_online_protected_inputs(protected_paths)
    staged = protected_paths["staged_hierarchical"]
    provenance = _provenance(staged, _sha256(staged))
    provenance["protected_before"] = before
    provenance["protected_after"] = copy.deepcopy(before)
    provenance["source_gate_baseline_receipt"] = {
        "path": before["source_gate_baseline"]["path"],
        "sha256": before["source_gate_baseline"]["sha256"],
    }
    provenance["staged_result_receipt"] = {
        "path": before["staged_result_receipt"]["path"],
        "sha256": before["staged_result_receipt"]["sha256"],
    }
    initialized = {
        "protected_paths": protected_paths,
        "provenance": provenance,
        "staged_artifact_path": staged.absolute(),
        "staged_artifact_sha256": _sha256(staged),
    }

    finalized = online.finalize_live_online_calibration(initialized)

    assert finalized is not initialized
    assert finalized["provenance"]["protected_before"] == \
        finalized["provenance"]["protected_after"]
    assert initialized["provenance"]["protected_after"] == before

    changed = copy.copy(initialized)
    changed["provenance"] = copy.deepcopy(initialized["provenance"])
    geometry = protected_paths["geometry"]
    geometry.chmod(0o644)
    geometry.write_bytes(b"changed")
    geometry.chmod(0o444)
    with pytest.raises(RuntimeError, match="during live evaluation"):
        online.finalize_live_online_calibration(changed)


def test_deployed_copy_flips_only_policy_and_never_mutates_staged(monkeypatch):
    staged = {
        "schema": "rec-hierarchical-query-variant-v1",
        "version": 1,
        "deployable": False,
        "validation_data_accessed": False,
        "payload": {"value": torch.tensor([1.0])},
    }
    original = copy.deepcopy(staged)
    calls = []

    def validate(artifact, **kwargs):
        calls.append((artifact["deployable"], kwargs["expected_deployable"]))
        assert artifact["deployable"] is kwargs["expected_deployable"]

    monkeypatch.setattr(online, "validate_hierarchical_artifact", validate)

    deployed = online.build_deployed_hierarchical_copy(staged)

    assert calls == [(False, False), (True, True)]
    assert staged["deployable"] is False
    assert staged["payload"]["value"].equal(original["payload"]["value"])
    expected = copy.deepcopy(staged)
    expected["deployable"] = True
    assert deployed.keys() == expected.keys()
    assert deployed["deployable"] is True
    assert deployed["payload"]["value"].equal(expected["payload"]["value"])


def test_publication_is_fresh_read_only_and_preserves_staged(
        monkeypatch, tmp_path):
    staged = tmp_path / "selected_hierarchical.pth"
    staged.write_bytes(b"immutable-staged")
    staged.chmod(0o444)
    staged_before = staged.read_bytes()
    staged_sha = _sha256(staged)
    record = _record(staged, staged_sha)
    output = tmp_path / "online"
    deployed_payload = b"deployable-copy"

    published = online.publish_online_calibration(
        output,
        record,
        deployed_payload,
        expected_staged_path=staged,
        expected_staged_sha256=staged_sha,
    )

    deployed = output / online.DEPLOYED_ARTIFACT_NAME
    receipt = output / online.ONLINE_CALIBRATION_NAME
    assert deployed.read_bytes() == deployed_payload
    assert deployed.stat().st_mode & 0o777 == 0o444
    assert receipt.stat().st_mode & 0o777 == 0o444
    assert staged.read_bytes() == staged_before
    assert staged.stat().st_mode & 0o777 == 0o444
    assert published["deployed_artifact"] == {
        "path": str(deployed.absolute()),
        "sha256": _sha256(deployed),
        "deployable": True,
    }
    assert json.loads(receipt.read_text(encoding="utf-8")) == published

    with pytest.raises(FileExistsError):
        online.publish_online_calibration(
            output,
            record,
            deployed_payload,
            expected_staged_path=staged,
            expected_staged_sha256=staged_sha,
        )


def test_failed_gate_seals_diagnostics_but_never_deploys(tmp_path):
    staged = tmp_path / "selected_hierarchical.pth"
    staged.write_bytes(b"staged")
    staged.chmod(0o444)
    record = _record(staged, _sha256(staged), candidate_hits025=3523)
    record["gate"] = {
        "passed": False,
        "failures": ["hits025"],
        "required_hits025": 3524,
        "required_hits050": 3316,
    }
    output = tmp_path / "online-failed"

    published = online.publish_online_calibration(
        output,
        record,
        None,
        expected_staged_path=staged,
        expected_staged_sha256=_sha256(staged),
    )

    assert published["gate"]["passed"] is False
    assert published["deployed_artifact"] is None
    assert not (output / online.DEPLOYED_ARTIFACT_NAME).exists()
    receipt = output / online.ONLINE_CALIBRATION_NAME
    assert receipt.stat().st_mode & 0o777 == 0o444
    assert json.loads(receipt.read_text(encoding="utf-8")) == published


@pytest.mark.parametrize("candidate_hits025,expect_payload", [
    (3524, True),
    (3523, False),
])
def test_one_shot_runner_serializes_only_after_online_gate(
        tmp_path, candidate_hits025, expect_payload):
    staged = tmp_path / "selected_hierarchical.pth"
    staged.write_bytes(b"staged")
    staged.chmod(0o444)
    staged_sha = _sha256(staged)
    args = SimpleNamespace(
        output_dir=str(tmp_path / "online"),
        staged_hierarchical_artifact=str(staged),
    )
    initialized = {
        "hierarchical_model": object(),
        "hierarchical_artifact": {"deployable": False},
        "staged_artifact_path": staged.absolute(),
        "staged_artifact_sha256": staged_sha,
        "provenance": _provenance(staged, staged_sha),
    }
    events = []

    def initialize(received):
        events.append("initialize")
        assert received is args
        return initialized

    def evaluate(received, hierarchical_model, hierarchical_artifact):
        events.append("evaluate")
        assert received is initialized
        assert hierarchical_model is initialized["hierarchical_model"]
        assert hierarchical_artifact is initialized["hierarchical_artifact"]
        baseline = _evidence(3461, 3316)
        candidate = _evidence(candidate_hits025, 3316)
        candidate["selected_iou_sha256"] = "4" * 64
        return {"baseline": baseline, "candidate": candidate}

    def finalize(received):
        events.append("finalize")
        assert received is initialized
        return received

    def serialize(artifact):
        events.append("serialize")
        assert artifact is initialized["hierarchical_artifact"]
        return b"deployed"

    def publish(output_dir, record, payload, **kwargs):
        events.append("publish")
        assert output_dir == args.output_dir
        assert record["gate"]["passed"] is expect_payload
        assert (payload == b"deployed") is expect_payload
        assert kwargs == {
            "expected_staged_path": staged.absolute(),
            "expected_staged_sha256": staged_sha,
        }
        return record

    result = online.run_online_calibration(
        args,
        initializer=initialize,
        evaluator=evaluate,
        finalizer=finalize,
        deployed_serializer=serialize,
        publisher=publish,
    )

    expected_events = ["initialize", "evaluate", "finalize"]
    if expect_payload:
        expected_events.append("serialize")
    expected_events.append("publish")
    assert events == expected_events
    assert result["gate"]["passed"] is expect_payload
    assert result["validation_data_accessed"] is False
    assert result["inference_uses_ground_truth"] is False


def test_main_runs_live_orchestration_and_prints_canonical_result(capsys):
    argv = [
        "--data-root", "/a/train-root",
        "--backbone-checkpoint", "/a/backbone.pth",
        "--parent-artifact", "/a/parent.pth",
        "--geometry-artifact", "/a/geometry.pth",
        "--staged-hierarchical-artifact", "/a/staged.pth",
        "--staged-result-receipt", "/a/result-receipt.json",
        "--source-gate-baseline-receipt", "/a/source-gate.json",
        "--output-dir", "/a/output",
        "--device", "cuda:0",
    ]
    result = {"gate": {"passed": True}, "sample_count": 3625}
    calls = []

    def runner(args):
        calls.append(args)
        assert args.staged_hierarchical_artifact == "/a/staged.pth"
        return result

    returned = online.main(argv, runner=runner)

    assert returned == result
    assert len(calls) == 1
    assert capsys.readouterr().out == json.dumps(
        result, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ) + "\n"


def test_cli_is_exact_train_only_and_has_no_validation_argument():
    parsed = online.parse_args([
        "--data-root", "/a/train-root",
        "--backbone-checkpoint", "/a/backbone.pth",
        "--parent-artifact", "/a/parent.pth",
        "--geometry-artifact", "/a/geometry.pth",
        "--staged-hierarchical-artifact", "/a/staged.pth",
        "--staged-result-receipt", "/a/result-receipt.json",
        "--source-gate-baseline-receipt", "/a/source-gate.json",
        "--output-dir", "/a/output",
        "--device", "cuda:0",
    ])

    assert parsed.device == "cuda:0"
    assert not any(
        "val" in name or "official" in name
        for name in vars(parsed)
    )
    with pytest.raises(SystemExit):
        online.parse_args([
            "--data-root", "/a/train-root",
            "--backbone-checkpoint", "/a/backbone.pth",
            "--parent-artifact", "/a/parent.pth",
            "--geometry-artifact", "/a/geometry.pth",
            "--staged-hierarchical-artifact", "/a/staged.pth",
            "--staged-result-receipt", "/a/result-receipt.json",
            "--source-gate-baseline-receipt", "/a/source-gate.json",
            "--output-dir", "/a/output",
            "--validation-cache", "/forbidden",
        ])
