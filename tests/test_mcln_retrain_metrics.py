import json
import math
from types import SimpleNamespace

import pytest

import train_dist_mod
from main_utils import (
    build_source_moe_gate_decision_diagnostics,
    save_eval_metrics_receipt,
    save_source_choice_diagnostics_receipt,
)
from src.grounding_evaluator import GroundingEvaluator
from train_dist_mod import TrainTester


SOURCE_KEYS = (
    ("source_choice", "fixed_default", 0.25, 1),
    ("source_choice", "fixed_default", 0.5, 1),
    ("source_choice", "learned_selector", 0.25, 1),
    ("source_choice", "learned_selector", 0.5, 1),
)

EXPECTED_RECEIPT = {
    "schema": "mcln-retrain-metrics-v1",
    "sample_count": 4,
    "position": {
        "fixed_default": {"hits025": 2, "hits050": 1},
        "learned_selector": {"hits025": 3, "hits050": 2},
    },
    "position_subgroups": {
        "unique": {
            "sample_count": 1,
            "hits025": 1,
            "hits050": 1,
            "acc025": 1.0,
            "acc050": 1.0,
        },
        "multiple": {
            "sample_count": 3,
            "hits025": 2,
            "hits050": 1,
            "acc025": 2.0 / 3.0,
            "acc050": 1.0 / 3.0,
        },
    },
    "mask": {
        "hits025": 3,
        "hits050": 2,
        "iou_sum": 2.25,
        "miou": 0.5625,
        "position_subgroups": {
            "unique": {
                "sample_count": 1,
                "hits025": 1,
                "hits050": 1,
                "acc025": 1.0,
                "acc050": 1.0,
            },
            "multiple": {
                "sample_count": 3,
                "hits025": 2,
                "hits050": 1,
                "acc025": 2.0 / 3.0,
                "acc050": 1.0 / 3.0,
            },
        },
    },
}


def test_gate_decision_diagnostics_use_global_counts():
    diagnostics = build_source_moe_gate_decision_diagnostics({
        "source_moe_gate_supervised_sample_count": 10,
        "source_moe_gate_oracle_switch_count": 4,
        "source_moe_gate_predicted_switch_count": 3,
        "source_moe_gate_beneficial_switch_count": 2,
        "source_moe_gate_harmful_switch_count": 1,
        "source_moe_gate_oracle_query_match_count": 3,
    })

    assert diagnostics == {
        "sample_count": 10,
        "oracle_switch_count": 4,
        "predicted_switch_count": 3,
        "beneficial_switch_count": 2,
        "harmful_switch_count": 1,
        "oracle_query_match_count": 3,
        "oracle_switch_rate": 0.4,
        "predicted_switch_rate": 0.3,
        "oracle_switch_recall": 0.5,
        "predicted_switch_precision": pytest.approx(2.0 / 3.0),
        "false_switch_rate": pytest.approx(1.0 / 3.0),
        "oracle_query_match_rate": 0.75,
    }


def test_gate_decision_diagnostics_include_optional_pairwise_target_count():
    diagnostics = build_source_moe_gate_decision_diagnostics({
        "source_moe_gate_supervised_sample_count": 10,
        "source_moe_gate_oracle_switch_count": 4,
        "source_moe_gate_row_target_switch_count": 3,
        "source_moe_gate_predicted_switch_count": 2,
        "source_moe_gate_beneficial_switch_count": 1,
        "source_moe_gate_harmful_switch_count": 1,
        "source_moe_gate_oracle_query_match_count": 2,
    })

    assert diagnostics["row_target_switch_count"] == 3
    assert diagnostics["row_target_switch_rate"] == pytest.approx(0.3)


def test_gate_decision_diagnostics_reject_inconsistent_partition():
    with pytest.raises(ValueError, match="partition predictions"):
        build_source_moe_gate_decision_diagnostics({
            "source_moe_gate_supervised_sample_count": 10,
            "source_moe_gate_oracle_switch_count": 4,
            "source_moe_gate_predicted_switch_count": 3,
            "source_moe_gate_beneficial_switch_count": 2,
            "source_moe_gate_harmful_switch_count": 2,
            "source_moe_gate_oracle_query_match_count": 3,
        })


def test_eval_metrics_receipt_is_atomic_and_exact(tmp_path):
    path = save_eval_metrics_receipt(tmp_path, 7, EXPECTED_RECEIPT)

    assert path == str(tmp_path / "eval_metrics_epoch_7.json")
    with open(path, "r", encoding="utf-8") as handle:
        assert json.load(handle) == EXPECTED_RECEIPT
    assert not list(tmp_path.glob("*.tmp"))


def test_source_choice_diagnostics_receipt_is_separate_and_atomic(tmp_path):
    diagnostics = {
        "schema": "mcln-source-choice-diagnostics-v1",
        "sample_count": 4,
        "gate_candidate_oracle": {
            "hits025": 3,
            "hits050": 2,
            "iou_sum": 2.5,
            "miou": 0.625,
        },
        "gate_oracle_headroom": {
            "hits025": 1,
            "hits050": 1,
            "rate025": 0.25,
            "rate050": 0.25,
        },
    }

    path = save_source_choice_diagnostics_receipt(
        tmp_path, 7, diagnostics
    )

    assert path == str(tmp_path / "source_choice_diagnostics_epoch_7.json")
    with open(path, "r", encoding="utf-8") as handle:
        assert json.load(handle) == diagnostics
    assert not list(tmp_path.glob("*.tmp"))


def _receipt_evaluator():
    evaluator = GroundingEvaluator(
        only_root=True,
        thresholds=[0.25, 0.5],
        topks=[1],
        prefixes=["last_"],
    )
    evaluator.dets.update({
        SOURCE_KEYS[0]: 2,
        SOURCE_KEYS[1]: 1,
        SOURCE_KEYS[2]: 3,
        SOURCE_KEYS[3]: 2,
        "overall_mask": 3,
        "overall50_mask": 2,
        "mask_sem": 2.25,
    })
    evaluator.gts.update({key: 4 for key in SOURCE_KEYS})
    # Real mask accumulation retains reset()'s tiny division sentinel.
    evaluator.gts["mask_sem"] = 4 + 1e-14
    evaluator.gts.update({"unique_num": 1, "multi_num": 3})
    evaluator.dets.update({
        "unique_mask": 1,
        "unique50_mask": 1,
        "multi_mask": 2,
        "multi50_mask": 1,
    })
    for threshold, unique_hits, multiple_hits in (
            (0.25, 1, 2), (0.5, 1, 1)):
        unique_key = ("position_subgroup", "unique", threshold)
        multiple_key = ("position_subgroup", "multiple", threshold)
        evaluator.dets[unique_key] = unique_hits
        evaluator.gts[unique_key] = 1
        evaluator.dets[multiple_key] = multiple_hits
        evaluator.gts[multiple_key] = 3
    return evaluator


def test_export_retrain_metrics_returns_exact_five_metric_receipt():
    evaluator = _receipt_evaluator()

    assert evaluator.export_retrain_metrics(
        expected_sample_count=4
    ) == EXPECTED_RECEIPT


@pytest.mark.parametrize("mapping_name", ["dets", "gts"])
def test_export_retrain_metrics_rejects_missing_source_counter(mapping_name):
    evaluator = _receipt_evaluator()
    getattr(evaluator, mapping_name).pop(SOURCE_KEYS[0])

    with pytest.raises(ValueError):
        evaluator.export_retrain_metrics()


@pytest.mark.parametrize(
    "counter_name",
    [SOURCE_KEYS[0], SOURCE_KEYS[3], "mask_sem"],
)
def test_export_retrain_metrics_rejects_different_denominators(counter_name):
    evaluator = _receipt_evaluator()
    evaluator.gts[counter_name] = 5

    with pytest.raises(ValueError):
        evaluator.export_retrain_metrics()


@pytest.mark.parametrize(
    "hits025_key,hits050_key",
    [
        (SOURCE_KEYS[0], SOURCE_KEYS[1]),
        (SOURCE_KEYS[2], SOURCE_KEYS[3]),
        ("overall_mask", "overall50_mask"),
    ],
)
def test_export_retrain_metrics_rejects_hits050_above_hits025(
        hits025_key, hits050_key):
    evaluator = _receipt_evaluator()
    evaluator.dets[hits025_key] = 1
    evaluator.dets[hits050_key] = 2

    with pytest.raises(ValueError):
        evaluator.export_retrain_metrics()


@pytest.mark.parametrize("iou_sum", [math.nan, math.inf, -math.inf])
def test_export_retrain_metrics_rejects_nonfinite_mask_iou_sum(iou_sum):
    evaluator = _receipt_evaluator()
    evaluator.dets["mask_sem"] = iou_sum

    with pytest.raises(ValueError):
        evaluator.export_retrain_metrics()


def test_export_retrain_metrics_does_not_count_mask_sentinel_as_a_sample():
    evaluator = _receipt_evaluator()
    for key in SOURCE_KEYS:
        evaluator.dets[key] = 0
        evaluator.gts[key] = 1
    evaluator.dets["overall_mask"] = 0
    evaluator.dets["overall50_mask"] = 0
    evaluator.dets["mask_sem"] = 0.0
    evaluator.gts["mask_sem"] = 1e-14

    with pytest.raises(ValueError):
        evaluator.export_retrain_metrics()


def test_export_retrain_metrics_rejects_expected_sample_count_mismatch():
    evaluator = _receipt_evaluator()

    with pytest.raises(ValueError):
        evaluator.export_retrain_metrics(expected_sample_count=5)


@pytest.mark.parametrize(
    "mapping_name,counter_name,value",
    [
        ("gts", SOURCE_KEYS[0], 4.000000002),
        ("gts", "mask_sem", 4.000000002),
        ("dets", SOURCE_KEYS[0], 2.000000002),
        ("dets", "overall_mask", 3.000000002),
    ],
)
def test_export_retrain_metrics_rejects_noninteger_samples_and_hits(
        mapping_name, counter_name, value):
    evaluator = _receipt_evaluator()
    getattr(evaluator, mapping_name)[counter_name] = value

    with pytest.raises(ValueError):
        evaluator.export_retrain_metrics()


def test_export_retrain_metrics_accepts_counters_within_integer_tolerance():
    evaluator = _receipt_evaluator()
    evaluator.dets[SOURCE_KEYS[0]] = 2.0000000005
    for key in SOURCE_KEYS:
        evaluator.gts[key] = 4.0000000005
    evaluator.gts["mask_sem"] = 4.0000000005

    receipt = evaluator.export_retrain_metrics(expected_sample_count=4)

    assert receipt["sample_count"] == 4
    assert receipt["position"]["fixed_default"]["hits025"] == 2


class _EvalModel:
    def __init__(self):
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1


class _EvalTensorboard:
    def __init__(self, events):
        self.events = events
        self.item = {
            "val_score": {},
            "val_loss": {},
        }

    def dump_tensorboard(self, name, epoch):
        self.events.append(("dump", name, epoch))


class _EvalReceiptEvaluator:
    def __init__(self, events, receipt):
        self.events = events
        self.receipt = receipt
        self.dets = {
            ("last_", 0.25, 1, "bbs"): 2,
            ("last_", 0.5, 1, "bbs"): 1,
            ("last_", 0.25, 1, "bbf"): 3,
            ("last_", 0.5, 1, "bbf"): 2,
        }
        self.gts = {key: 4 for key in self.dets}

    def synchronize_between_processes(self):
        self.events.append("synchronize")

    def print_stats(self):
        self.events.append("print_stats")

    def export_retrain_metrics(self, expected_sample_count=None):
        self.events.append(("export", expected_sample_count))
        return self.receipt


def _recording_real_evaluator(events):
    evaluator = _receipt_evaluator()
    evaluator.dets.update({
        ("last_", 0.25, 1, "bbs"): 2,
        ("last_", 0.5, 1, "bbs"): 1,
        ("last_", 0.25, 1, "bbf"): 3,
        ("last_", 0.5, 1, "bbf"): 2,
    })
    evaluator.gts.update({
        ("last_", 0.25, 1, "bbs"): 4,
        ("last_", 0.5, 1, "bbs"): 4,
        ("last_", 0.25, 1, "bbf"): 4,
        ("last_", 0.5, 1, "bbf"): 4,
    })
    evaluator.synchronize_between_processes = lambda: events.append(
        "synchronize"
    )
    evaluator.print_stats = lambda: events.append("print_stats")
    export = evaluator.export_retrain_metrics

    def recording_export(expected_sample_count=None):
        events.append(("export", expected_sample_count))
        return export(expected_sample_count=expected_sample_count)

    evaluator.export_retrain_metrics = recording_export
    return evaluator


def _minimal_eval_tester(monkeypatch, rank, expected_sample_count=4,
                         use_source_choice_selector=True,
                         use_source_moe=False,
                         use_real_exporter=False):
    events = []
    if use_real_exporter:
        receipt = EXPECTED_RECEIPT
        evaluator = _recording_real_evaluator(events)
    else:
        receipt = {"schema": "mcln-retrain-metrics-v1"}
        evaluator = _EvalReceiptEvaluator(events, receipt)
    tester = TrainTester.__new__(TrainTester)
    tester.tensorboard = _EvalTensorboard(events)
    tester._log_source_moe_diagnostics = lambda _stats, _denom: None
    tester._build_grounding_evaluator = lambda _args, _prefixes: evaluator
    monkeypatch.setattr(train_dist_mod, "tqdm", lambda values, ascii: values)
    monkeypatch.setattr(train_dist_mod.dist, "get_rank", lambda: rank)
    args = SimpleNamespace(
        test_dataset="scanrefer",
        num_decoder_layers=1,
        expected_eval_sample_count=expected_sample_count,
        use_source_choice_selector=use_source_choice_selector,
        use_source_moe=use_source_moe,
    )
    return tester, evaluator, events, receipt, args


def test_evaluate_one_epoch_returns_receipt_on_rank_zero_after_existing_logs(
        monkeypatch):
    tester, _evaluator, events, receipt, args = _minimal_eval_tester(
        monkeypatch, rank=0, use_real_exporter=True
    )
    model = _EvalModel()

    result = tester.evaluate_one_epoch(
        epoch=7,
        test_loader=[],
        model=model,
        criterion=None,
        set_criterion=None,
        args=args,
    )

    assert result == receipt
    assert model.eval_calls == 1
    assert tester.tensorboard.item["val_score"] == {
        "soft_token_0.25": 0.5,
        "soft_token_0.5": 0.25,
        "contrastive_0.25": 0.75,
        "contrastive_0.5": 0.5,
    }
    assert events == [
        "synchronize",
        ("dump", "val_score", 7),
        ("dump", "val_loss", 7),
        "print_stats",
        ("export", 4),
    ]


def test_evaluate_one_epoch_returns_none_without_exporting_on_non_root(
        monkeypatch):
    tester, _evaluator, events, _receipt, args = _minimal_eval_tester(
        monkeypatch, rank=1
    )

    result = tester.evaluate_one_epoch(
        epoch=7,
        test_loader=[],
        model=_EvalModel(),
        criterion=None,
        set_criterion=None,
        args=args,
    )

    assert result is None
    assert events == ["synchronize"]


def test_evaluate_one_epoch_keeps_legacy_selector_disabled_root_compatible(
        monkeypatch):
    tester, _evaluator, events, _receipt, args = _minimal_eval_tester(
        monkeypatch,
        rank=0,
        use_source_choice_selector=False,
    )

    result = tester.evaluate_one_epoch(
        epoch=7,
        test_loader=[],
        model=_EvalModel(),
        criterion=None,
        set_criterion=None,
        args=args,
    )

    assert result is None
    assert events == [
        "synchronize",
        ("dump", "val_score", 7),
        ("dump", "val_loss", 7),
        "print_stats",
    ]


def test_evaluate_one_epoch_exports_same_receipt_for_source_moe(monkeypatch):
    tester, _evaluator, events, receipt, args = _minimal_eval_tester(
        monkeypatch,
        rank=0,
        use_source_choice_selector=False,
        use_source_moe=True,
    )

    result = tester.evaluate_one_epoch(
        epoch=7,
        test_loader=[],
        model=_EvalModel(),
        criterion=None,
        set_criterion=None,
        args=args,
    )

    assert result == receipt
    assert events[-1] == ("export", 4)


def test_evaluate_one_epoch_persists_gate_diagnostics_receipt(
        monkeypatch, tmp_path):
    tester, evaluator, events, receipt, args = _minimal_eval_tester(
        monkeypatch,
        rank=0,
        use_source_choice_selector=False,
        use_source_moe=True,
    )
    diagnostics = {
        "schema": "mcln-source-choice-diagnostics-v1",
        "sample_count": 4,
        "gate_candidate_oracle": {
            "hits025": 3,
            "hits050": 2,
            "iou_sum": 2.5,
            "miou": 0.625,
        },
        "gate_oracle_headroom": {
            "hits025": 1,
            "hits050": 1,
            "rate025": 0.25,
            "rate050": 0.25,
        },
    }

    def export_diagnostics(expected_sample_count=None):
        events.append(("export_diagnostics", expected_sample_count))
        return diagnostics

    evaluator.export_source_choice_diagnostics = export_diagnostics
    args.log_dir = str(tmp_path)

    result = tester.evaluate_one_epoch(
        epoch=7,
        test_loader=[],
        model=_EvalModel(),
        criterion=None,
        set_criterion=None,
        args=args,
    )

    assert result == receipt
    output = tmp_path / "source_choice_diagnostics_epoch_7.json"
    with output.open("r", encoding="utf-8") as handle:
        assert json.load(handle) == diagnostics
    assert ("export_diagnostics", 4) in events
