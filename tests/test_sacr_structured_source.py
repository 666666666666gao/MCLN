import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from main_utils import (
    _requires_joint_det_structured_collate,
    joint_det_structured_collate,
)
from models.joint_query_quality import (
    JointQueryQualityReranker,
    compute_joint_query_quality_loss,
)
from models.sacr_head import SACRHead
from models.source_choice_adapter import build_mcln_source_choice_batch
from models.structured_slots import StructuredSlotBuilder
from models.structured_source import (
    apply_authoritative_coverage,
    build_decomposition_masks,
    build_token_span_tensors,
)
from scripts.audit_sacr_structured_data import EXPECTED_SPLIT_COUNTS
from src.joint_det_dataset import Joint3DDataset
from src.structured_annotations import (
    build_structured_annotation,
    realign_structured_annotation,
)


class _FakeTokenized(dict):
    def __init__(self, texts):
        max_length = max(len(text) for text in texts) + 2
        super().__init__({
            "input_ids": torch.ones(len(texts), max_length, dtype=torch.long),
            "attention_mask": torch.ones(
                len(texts), max_length, dtype=torch.long
            ),
        })
        self.texts = texts

    def char_to_token(self, batch_index, char_index):
        if 0 <= char_index < len(self.texts[batch_index]):
            return char_index + 1
        return None


def _structured_record():
    return {
        "target_slot": {"text": "red chair", "start": 4, "end": 13},
        "entities": [
            {"text": "red chair", "start": 4, "end": 13},
            {"text": "table", "start": 23, "end": 28},
        ],
        "attr_slot": {
            "items": [{"text": "red", "start": 4, "end": 7}]
        },
        "rel_slots": [
            {"text": "left of", "start": 14, "end": 21}
        ],
        "anchor_slots": [
            {"text": "table", "start": 23, "end": 28}
        ],
        "parse_confidence": 0.9,
        "coverage_stats": {
            "has_target": 1,
            "valid_tuple_count": 1,
            "num_attrs": 1,
        },
    }


def test_sacr_audit_pins_authoritative_train_and_validation_counts():
    assert EXPECTED_SPLIT_COUNTS == {
        "train": {
            "scanrefer": 36665,
            "nr3d": 32919,
            "sr3d": 65846,
        },
        "val": {
            "scanrefer": 9508,
            "nr3d": 7899,
            "sr3d": 17726,
        },
    }


def test_structured_annotation_builds_explicit_target_and_anchor_contract():
    annotation = build_structured_annotation(
        _structured_record(), "the red chair left of a table"
    )
    assert annotation["structured_annotation_available"] is True
    assert annotation["target_spans"][0]["text"] == "red chair"
    assert annotation["structured_anchor_ids"] == [1]
    assert annotation["parse_confidence"] == pytest.approx(0.9)


def test_structured_annotation_fails_closed_without_target():
    record = _structured_record()
    record["target_slot"] = {}
    annotation = build_structured_annotation(
        record, "the red chair left of a table"
    )
    assert annotation["structured_annotation_available"] is False
    assert annotation["parse_confidence"] == 0.0
    assert annotation["decomp_global_only_mask"] is True


def test_token_alignment_repairs_unique_normalized_offset():
    text = "the red chair left of a table"
    tokenized = _FakeTokenized([text])
    spans = [[{"text": "red chair", "start": 100, "end": 109}]]
    tensor = build_token_span_tensors(
        tokenized, spans, [text], torch.device("cpu")
    )
    assert tensor.tolist() == [[[5, 14]]]


def test_structured_spans_realign_after_legacy_text_normalization():
    source = "the red 2-tiered chair, left of a table"
    target = "the red 2 - tiered chair , left of a table"
    annotation = build_structured_annotation(
        {
            "target_slot": {"text": "red 2-tiered chair", "start": 4,
                            "end": 22},
            "entities": [
                {"text": "red 2-tiered chair", "start": 4, "end": 22},
                {"text": "table", "start": 34, "end": 39},
            ],
            "rel_slots": [
                {"text": "left of", "start": 24, "end": 31}
            ],
            "anchor_slots": [
                {"text": "table", "start": 34, "end": 39}
            ],
            "coverage_stats": {"has_target": 1, "parse_confidence": 0.9},
        },
        source,
    )
    realign_structured_annotation(annotation, source, target)
    assert annotation["target_spans"][0]["text"] == "red 2 - tiered chair"
    assert annotation["rel_spans"][0]["text"] == "left of"
    anchor_id = annotation["structured_anchor_ids"][0]
    assert annotation["entity_spans"][anchor_id]["text"] == "table"
    for key in ("target_spans", "entity_spans", "rel_spans"):
        for span in annotation[key]:
            assert target[span["start"]:span["end"]] == span["text"]


def test_structured_realign_fails_closed_when_target_is_deleted():
    annotation = build_structured_annotation(
        {
            "target_slot": {"text": "chair", "start": 4, "end": 9},
            "entities": [{"text": "chair", "start": 4, "end": 9}],
            "coverage_stats": {"has_target": 1, "parse_confidence": 0.9},
        },
        "the chair",
    )
    realign_structured_annotation(annotation, "the chair", "the")
    assert annotation["target_spans"] == []
    assert annotation["structured_annotation_available"] is False
    assert annotation["parse_confidence"] == 0.0
    assert annotation["decomp_global_only_mask"] is True


def test_referit3d_sidecar_key_preserves_duplicate_sr3d_rows():
    row = {
        "stimulus_id": "scene0000_00-chair-2-1-2",
        "scan_id": "scene0000_00",
        "target_id": "1",
        "utterance": "the chair beside the table",
    }
    duplicate = dict(row)
    structured = dict(row, target_slot='{"text":"chair"}')
    key = Joint3DDataset._referit3d_structured_key
    lookup = {key(structured, "sr3d"): structured}
    matched = [lookup.get(key(item, "sr3d")) for item in (row, duplicate)]
    assert matched == [structured, structured]


def test_nr3d_sidecar_key_uses_unique_assignment_id():
    first = {
        "assignmentid": "first",
        "stimulus_id": "same",
        "scan_id": "scene0000_00",
        "target_id": "1",
        "utterance": "same utterance",
    }
    second = dict(first, assignmentid="second")
    key = Joint3DDataset._referit3d_structured_key
    assert key(first, "nr3d") != key(second, "nr3d")


def test_slots_and_sacr_gradient_with_valid_relation():
    torch.manual_seed(4)
    batch_size, token_count, feature_dim = 2, 32, 16
    token_feats = torch.randn(
        batch_size, token_count, feature_dim, requires_grad=True
    )
    target = torch.tensor([[[5, 14]], [[-1, -1]]])
    entities = torch.tensor([
        [[5, 14], [24, 29]],
        [[-1, -1], [-1, -1]],
    ])
    attrs = torch.tensor([[[5, 8]], [[-1, -1]]])
    relations = torch.tensor([[[15, 22]], [[-1, -1]]])
    anchor_ids = torch.tensor([[1, -1, -1], [-1, -1, -1]])
    builder = StructuredSlotBuilder(
        d_model=feature_dim, max_pairs=2
    )
    slots = builder(
        token_feats,
        torch.ones(batch_size, token_count, dtype=torch.long),
        target_spans=target,
        entity_spans=entities,
        attr_spans=attrs,
        rel_spans=relations,
        anchor_ids=anchor_ids,
    )
    inputs = {
        "coverage_stats": [
            {"has_target": 1, "parse_confidence": 0.9},
            {"has_target": 0, "parse_confidence": 0.0},
        ],
        "structured_annotation_available": torch.tensor([True, False]),
    }
    slots = apply_authoritative_coverage(inputs, slots)
    global_only, weak = build_decomposition_masks(inputs, slots)
    head = SACRHead(
        d_model=feature_dim,
        hidden_dim=24,
        top_m_targets=4,
        top_k_anchors=3,
        geo_dim=8,
    )
    query = torch.randn(batch_size, 8, feature_dim, requires_grad=True)
    boxes = torch.rand(batch_size, 8, 6)
    boxes[..., 3:] += 0.2
    output = head(
        query,
        boxes,
        torch.randn(batch_size, 8),
        slots,
        global_only_mask=global_only,
        weak_generic_target_mask=weak,
    )
    assert output["structured_valid_mask"].tolist() == [True, False]
    assert torch.equal(
        output["structured_scores"][1], torch.zeros(8)
    )
    output["structured_scores"][0].sum().backward()
    assert query.grad is not None and query.grad.abs().sum() > 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in head.parameters()
    )


def test_joint_source_alignment_reaches_sacr_head_at_step_zero():
    torch.manual_seed(5)
    query_count, token_count, feature_dim = 8, 32, 16
    token_feats = torch.randn(1, token_count, feature_dim)
    builder = StructuredSlotBuilder(d_model=feature_dim, max_pairs=2)
    slots = builder(
        token_feats,
        torch.ones(1, token_count, dtype=torch.long),
        target_spans=torch.tensor([[[5, 14]]]),
        entity_spans=torch.tensor([[[5, 14], [24, 29]]]),
        attr_spans=torch.tensor([[[5, 8]]]),
        rel_spans=torch.tensor([[[15, 22]]]),
        anchor_ids=torch.tensor([[1, -1]]),
    )
    head = SACRHead(
        d_model=feature_dim, hidden_dim=24,
        top_m_targets=4, top_k_anchors=3, geo_dim=8,
    )
    query = torch.randn(1, query_count, feature_dim)
    boxes = torch.rand(1, query_count, 6)
    boxes[..., 3:] += 0.2
    baseline = torch.randn(1, query_count)
    structured = head(
        query, boxes, baseline, slots,
        global_only_mask=torch.tensor([False]),
    )["structured_scores"]
    residual_scale = torch.nn.Parameter(torch.tensor([0.1]))
    sacr_scores = (
        baseline.detach()
        + residual_scale.tanh() * 0.9 * structured
    )
    source_scores = torch.stack((
        baseline.detach(),
        torch.flip(baseline.detach(), dims=(1,)),
        torch.roll(baseline.detach(), shifts=1, dims=1),
        sacr_scores,
    ), dim=-1)
    reranker = JointQueryQualityReranker(
        8, hidden_dim=16, num_heads=4, dropout=0.0,
        detach_inputs=False,
        use_adaptive_source_mixing=True,
        source_count=4, shared_source_index=0,
    )
    valid = torch.ones(1, query_count, dtype=torch.bool)
    outputs = reranker(
        torch.randn(1, query_count, 8), baseline, valid,
        source_score_stack=source_scores,
        source_validity=torch.ones_like(source_scores, dtype=torch.bool),
    )
    box_ious = torch.tensor([[0.9, 0.1, 0.7, 0.2, 0.6, 0.3, 0.8, 0.4]])
    mask_ious = torch.tensor([[0.8, 0.2, 0.5, 0.1, 0.7, 0.4, 0.6, 0.3]])
    loss = compute_joint_query_quality_loss(
        outputs, box_ious, mask_ious,
        source_mix_loss_weight=0.25,
        source_mix_alignment_temperature=0.25,
        source_mix_query_focus_weight=0.75,
    )["loss"]

    assert torch.equal(outputs["scores"], baseline)
    loss.backward()
    assert residual_scale.grad is not None
    assert torch.isfinite(residual_scale.grad).all()
    assert residual_scale.grad.abs().sum().item() > 0.0
    for module in (head, builder):
        dead_parameters = [
            name for name, parameter in module.named_parameters()
            if parameter.grad is None
            or not bool(torch.isfinite(parameter.grad).all().item())
            or int(torch.count_nonzero(parameter.grad).item()) == 0
        ]
        assert dead_parameters == []


def test_source_adapter_broadcasts_sacr_row_validity():
    batch_size, query_count, token_count = 2, 4, 6
    end_points = {
        "last_center": torch.zeros(batch_size, query_count, 3),
        "last_pred_size": torch.ones(batch_size, query_count, 3),
        "source_choice_candidate_feats": torch.randn(
            batch_size, query_count, 8
        ),
        "last_sem_cls_scores": torch.randn(
            batch_size, query_count, token_count
        ),
        "sacr_structured_residual": torch.randn(batch_size, query_count),
        "sacr_structured_valid_mask": torch.tensor([True, False]),
    }
    inputs = {
        "positive_map": torch.ones(batch_size, 1, token_count),
    }
    batch = build_mcln_source_choice_batch(
        end_points,
        inputs,
        source_names=("default", "sacr_structured"),
    )
    assert batch["source_validity"][:, :, 0].all()
    assert batch["source_validity"][0, :, 1].all()
    assert not batch["source_validity"][1, :, 1].any()


def test_structured_collate_preserves_variable_span_lists():
    first = {
        "point_clouds": torch.zeros(2, 3),
        "target_spans": [{"start": 0, "end": 1, "text": "a"}],
        "entity_spans": [],
        "attr_spans": [],
        "rel_spans": [],
        "coverage_stats": {"has_target": 1},
        "decomposition_status": "ok",
    }
    second = dict(first)
    second["target_spans"] = []
    second["entity_spans"] = [
        {"start": 0, "end": 1, "text": "b"},
        {"start": 2, "end": 3, "text": "c"},
    ]
    collated = joint_det_structured_collate([first, second])
    assert collated["point_clouds"].shape == (2, 2, 3)
    assert [len(row) for row in collated["entity_spans"]] == [0, 2]


@pytest.mark.parametrize(
    "enabled_flag",
    [
        "use_sacr_source",
        "use_sacr_score_refiner",
        "use_parent_relative_text_verifier",
    ],
)
def test_structured_consumers_select_variable_length_collate(enabled_flag):
    args = SimpleNamespace(
        use_sacr_source=False,
        use_sacr_score_refiner=False,
        use_parent_relative_text_verifier=False,
    )
    setattr(args, enabled_flag, True)
    assert _requires_joint_det_structured_collate(args) is True


def test_unstructured_runs_keep_default_collate():
    args = SimpleNamespace(
        use_sacr_source=False,
        use_sacr_score_refiner=False,
        use_parent_relative_text_verifier=False,
    )
    assert _requires_joint_det_structured_collate(args) is False


@pytest.mark.parametrize(
    "asset",
    [
        "/root/autodl-tmp/DATA_ROOT/refer_it_3d/nr3d_spacy.csv",
        "/root/autodl-tmp/DATA_ROOT/refer_it_3d/sr3d_spacy.csv",
    ],
)
def test_referit3d_structured_assets_satisfy_contract(asset):
    path = Path(asset)
    if not path.is_file():
        pytest.skip("structured data asset is unavailable")
    with path.open() as handle:
        row = next(csv.DictReader(handle))
    annotation = build_structured_annotation(row, row["utterance"])
    assert annotation["target_spans"]
    assert annotation["structured_annotation_available"] is True
    assert 0.0 < annotation["parse_confidence"] <= 1.0


def test_scanrefer_structured_asset_satisfies_contract():
    path = Path(
        "/root/autodl-tmp/DATA_ROOT/scanrefer/"
        "ScanRefer_filtered_val_spacy.json"
    )
    if not path.is_file():
        pytest.skip("structured data asset is unavailable")
    with path.open() as handle:
        record = json.load(handle)[0]
    utterance = record.get("description", " ".join(record.get("tokens", [])))
    annotation = build_structured_annotation(record, utterance)
    assert annotation["target_spans"]
    assert annotation["structured_annotation_available"] is True
    assert 0.0 < annotation["parse_confidence"] <= 1.0
