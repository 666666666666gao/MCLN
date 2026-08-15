from pathlib import Path

import pytest

from scripts.run_frozen_v99_pareto_contextual_official import (
    ARTIFACT_PATH,
    EXPERIMENT,
    MASTER_PORT,
    build_authoritative_command,
    parse_metrics,
    validate_authoritative_command,
)


def test_authoritative_command_adds_only_v99_hierarchy(tmp_path):
    command = build_authoritative_command(tmp_path / "official")
    validate_authoritative_command(command, tmp_path / "official")
    assert command[command.index("--master_port") + 1] == str(MASTER_PORT)
    assert command[command.index("--exp") + 1] == EXPERIMENT
    assert command[
        command.index("--rec_hierarchical_reranker_checkpoint") + 1
    ] == str(ARTIFACT_PATH)
    assert "--eval_use_rec_hierarchical_reranker_scores" in command
    assert "--eval_use_rec_selective_residual_scores" not in command


def test_authoritative_command_rejects_ground_truth_flag(tmp_path):
    command = build_authoritative_command(tmp_path / "official") + ["--butd_gt"]
    with pytest.raises(ValueError, match="changed"):
        validate_authoritative_command(command, tmp_path / "official")


def test_parse_metrics_recovers_exact_rec_and_mask_counts():
    text = """
length of testing dataset: 9508
last_ position alignment Acc0.25: Top-1: 0.59003, Top-5: 0.70000, Top-10: 0.80000
last_ position alignment Acc0.50: Top-1: 0.49001, Top-5: 0.60000, Top-10: 0.70000
mask_sem 0.4186131
overall25 0.598233066891038
overall50 0.4913756836348338
"""
    metrics = parse_metrics(text)
    assert metrics["rec_hits025"] == 5610
    assert metrics["rec_hits050"] == 4659
    assert metrics["mask_hits025"] == 5688
    assert metrics["mask_hits050"] == 4672
    assert metrics["mask_miou"] == 0.4186131


def test_parse_metrics_rejects_duplicate_mask_metric():
    text = """
last_ position alignment Acc0.25: Top-1: 0.59003,
last_ position alignment Acc0.50: Top-1: 0.49001,
mask_sem 0.4186131
overall25 0.598233066891038
overall25 0.598233066891038
overall50 0.4913756836348338
"""
    with pytest.raises(ValueError, match="duplicate mask_acc025"):
        parse_metrics(text)


def test_mask_baseline_and_v19_reference_constants_are_distinct():
    import scripts.run_frozen_v99_pareto_contextual_official as official
    assert official.MASK_GEOMETRY_BASELINE_HITS025 == 5676
    assert official.MASK_GEOMETRY_BASELINE_HITS050 == 4662
    assert official.MASK_V19_BEST_HITS025 == 5688
    assert official.MASK_V19_BEST_HITS050 == 4672
    assert official.MASK_V19_BEST_MIOU > official.MASK_GEOMETRY_BASELINE_MIOU
