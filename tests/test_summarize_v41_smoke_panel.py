import json

from scripts.summarize_v41_smoke_panel import (
    SCHEMAS,
    atomic_write_json,
    build_summary,
    summarize_record,
)


def test_v43_selector_profile_has_a_distinct_receipt_schema():
    assert SCHEMAS["v43_selector"] == (
        "mcln-v43-selector-smoke-panel-v1"
    )


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _record(tmp_path, query_std=0.02, profile="v41",
            calibration_diagnostic=0.02, source_evidence_diagnostic=0.03,
            gate_evidence_diagnostic=0.04,
            candidate_mask_diagnostic=0.1, candidate_lovasz_loss=0.2,
            spatial_mask_diagnostic=0.05,
            spatial_superpoint_diagnostic=None,
            spatial_query_diagnostic=None,
            source_mix_residual_diagnostic=0.05,
            source_mix_router_diagnostic=0.03,
            source_mix_weight_std_diagnostic=0.04,
            source_mix_effective_count_diagnostic=1.8,
            source_mix_alignment_loss=0.7,
            source_mix_alignment_top1_acc=0.6,
            source_mix_alignment_target_effective_count=1.5,
            sacr_scale=0.1, sacr_valid_ratio=0.98,
            sacr_relation_ratio=0.55):
    if spatial_superpoint_diagnostic is None:
        spatial_superpoint_diagnostic = spatial_mask_diagnostic
    if spatial_query_diagnostic is None:
        spatial_query_diagnostic = spatial_mask_diagnostic
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_json(run_dir / "eval_metrics_epoch_3.json", {
        "schema": "mcln-retrain-metrics-v1",
        "sample_count": 128,
        "mask": {"hits025": 70, "hits050": 50, "miou": 0.4},
        "position": {"learned_selector": {"hits025": 72, "hits050": 52}},
    })
    _write_json(run_dir / "audit_{}.json".format(profile), {
        "schema": "mcln-source-moe-checkpoint-audit-v1",
        "profile": profile,
        "pass": True,
    })
    log_path = tmp_path / "launcher.log"
    log_path.write_text(
        "joint_query_quality_residual_abs_mean 0.0300\n"
        "joint_query_quality_residual_abs_max 0.1000\n"
        "joint_query_quality_residual_query_std {:.4f}\n"
        "joint_query_quality_switch_ratio 0.0500\n"
        "joint_query_quality_mask_alpha_residual_abs_mean {:.4f}\n"
        "joint_query_quality_mask_logit_bias_abs_mean {:.4f}\n"
        "joint_query_quality_mask_logit_bias_abs_max {:.4f}\n"
        "joint_query_quality_mask_weight_std_mean {:.4f}\n"
        "joint_query_quality_source_mask_evidence_query_std {:.4f}\n"
        "joint_query_quality_source_mask_disagreement_mean {:.4f}\n"
        "joint_query_quality_gate_evidence_query_std {:.4f}\n"
        "joint_query_quality_gate_candidate_ratio {:.4f}\n"
        "joint_query_quality_mask_spatial_residual_abs_mean {:.4f}\n"
        "joint_query_quality_mask_spatial_residual_abs_max {:.4f}\n"
        "joint_query_quality_mask_spatial_superpoint_std_mean {:.4f}\n"
        "joint_query_quality_mask_spatial_query_std_mean {:.4f}\n"
        "joint_query_quality_candidate_mask_query_ratio {:.4f}\n"
        "joint_query_quality_candidate_lovasz_loss {:.4f}\n"
        "joint_query_quality_source_mix_alignment_loss {:.4f}\n"
        "joint_query_quality_source_mix_alignment_target_top1_acc {:.4f}\n"
        "joint_query_quality_source_mix_alignment_target_effective_count_mean {:.4f}\n"
        "joint_query_quality_source_mix_residual_abs_mean {:.4f}\n"
        "joint_query_quality_source_mix_router_residual_abs_mean {:.4f}\n"
        "joint_query_quality_source_mix_weight_query_std_mean {:.4f}\n"
        "joint_query_quality_source_mix_effective_count_mean {:.4f}\n"
        "sacr_residual_scale_value {:.4f}\n"
        "sacr_valid_ratio {:.4f}\n"
        "sacr_relation_active_ratio {:.4f}\n".format(
            query_std,
            calibration_diagnostic,
            calibration_diagnostic,
            calibration_diagnostic,
            calibration_diagnostic,
            source_evidence_diagnostic,
            source_evidence_diagnostic,
            gate_evidence_diagnostic,
            gate_evidence_diagnostic,
            spatial_mask_diagnostic,
            spatial_mask_diagnostic,
            spatial_superpoint_diagnostic,
            spatial_query_diagnostic,
            candidate_mask_diagnostic,
            candidate_lovasz_loss,
            source_mix_alignment_loss,
            source_mix_alignment_top1_acc,
            source_mix_alignment_target_effective_count,
            source_mix_residual_diagnostic,
            source_mix_router_diagnostic,
            source_mix_weight_std_diagnostic,
            source_mix_effective_count_diagnostic,
            sacr_scale,
            sacr_valid_ratio,
            sacr_relation_ratio,
        ),
        encoding="utf-8",
    )
    return run_dir, log_path


def test_smoke_record_passes_complete_nonzero_contract(tmp_path):
    run_dir, log_path = _record(tmp_path)
    record = summarize_record(
        "base", str(run_dir), str(log_path), 3, 128
    )
    assert record["pass"] is True
    summary = build_summary([record], 3, 128)
    assert summary["pass"] is True


def test_smoke_record_rejects_zero_query_variation(tmp_path):
    run_dir, log_path = _record(tmp_path, query_std=0.0)
    record = summarize_record(
        "collapsed", str(run_dir), str(log_path), 3, 128
    )
    assert record["pass"] is False
    assert build_summary([record], 3, 128)["pass"] is False


def test_v42_smoke_requires_nonzero_mask_calibration_diagnostics(tmp_path):
    run_dir, log_path = _record(tmp_path, profile="v42")
    record = summarize_record(
        "v42", str(run_dir), str(log_path), 3, 128, profile="v42"
    )
    assert record["pass"] is True
    summary = build_summary([record], 3, 128, profile="v42")
    assert summary["schema"] == "mcln-v42-smoke-panel-v1"


def test_v42_smoke_rejects_collapsed_mask_calibration(tmp_path):
    run_dir, log_path = _record(
        tmp_path, profile="v42", calibration_diagnostic=0.0
    )
    record = summarize_record(
        "v42-collapsed", str(run_dir), str(log_path),
        3, 128, profile="v42",
    )
    assert record["pass"] is False


def test_v43_smoke_requires_the_same_noncollapsed_calibration_gate(tmp_path):
    run_dir, log_path = _record(tmp_path, profile="v43")
    record = summarize_record(
        "v43", str(run_dir), str(log_path), 3, 128, profile="v43"
    )
    summary = build_summary([record], 3, 128, profile="v43")

    assert record["pass"] is True
    assert summary["schema"] == "mcln-v43-smoke-panel-v1"
    assert summary["profile"] == "v43"


def test_v43_smoke_rejects_collapsed_source_mask_evidence(tmp_path):
    run_dir, log_path = _record(
        tmp_path, profile="v43", source_evidence_diagnostic=0.0
    )
    record = summarize_record(
        "v43-collapsed-evidence", str(run_dir), str(log_path),
        3, 128, profile="v43",
    )

    assert record["pass"] is False


def test_v46_smoke_requires_noncollapsed_gate_evidence(tmp_path):
    run_dir, log_path = _record(tmp_path, profile="v46")
    passing = summarize_record(
        "v46", str(run_dir), str(log_path), 3, 128, profile="v46"
    )
    assert passing["pass"] is True
    assert build_summary(
        [passing], 3, 128, profile="v46"
    )["schema"] == "mcln-v46-smoke-panel-v1"


def test_v46_smoke_rejects_collapsed_gate_evidence(tmp_path):
    run_dir, log_path = _record(
        tmp_path, profile="v46", gate_evidence_diagnostic=0.0
    )
    record = summarize_record(
        "v46-collapsed-gate", str(run_dir), str(log_path),
        3, 128, profile="v46",
    )
    assert record["pass"] is False


def test_v48_smoke_requires_noncollapsed_spatial_mask_residual(tmp_path):
    run_dir, log_path = _record(tmp_path, profile="v48")
    passing = summarize_record(
        "v48", str(run_dir), str(log_path), 3, 128, profile="v48"
    )
    assert passing["pass"] is True
    assert build_summary(
        [passing], 3, 128, profile="v48"
    )["schema"] == "mcln-v48-smoke-panel-v1"

    collapsed_dir = tmp_path / "collapsed"
    collapsed_dir.mkdir()
    collapsed_run, collapsed_log = _record(
        collapsed_dir, profile="v48", spatial_mask_diagnostic=0.0
    )
    collapsed = summarize_record(
        "v48-collapsed", str(collapsed_run), str(collapsed_log),
        3, 128, profile="v48",
    )
    assert collapsed["pass"] is False


def test_v48_smoke_rejects_nonzero_but_spatially_constant_residual(tmp_path):
    run_dir, log_path = _record(
        tmp_path,
        profile="v48",
        spatial_mask_diagnostic=0.05,
        spatial_superpoint_diagnostic=0.0,
        spatial_query_diagnostic=0.04,
    )
    record = summarize_record(
        "v48-constant", str(run_dir), str(log_path),
        3, 128, profile="v48",
    )

    assert record["pass"] is False


def test_v49_smoke_requires_adaptive_source_mix_diagnostics(tmp_path):
    run_dir, log_path = _record(tmp_path, profile="v49")
    passing = summarize_record(
        "v49", str(run_dir), str(log_path), 3, 128, profile="v49"
    )
    assert passing["pass"] is True
    assert build_summary(
        [passing], 3, 128, profile="v49"
    )["schema"] == "mcln-v49-smoke-panel-v1"

    collapsed_dir = tmp_path / "collapsed-v49"
    collapsed_dir.mkdir()
    collapsed_run, collapsed_log = _record(
        collapsed_dir,
        profile="v49",
        source_mix_router_diagnostic=0.0,
    )
    collapsed = summarize_record(
        "v49-collapsed", str(collapsed_run), str(collapsed_log),
        3, 128, profile="v49",
    )
    assert collapsed["pass"] is False


def test_v49_smoke_rejects_missing_source_mix_alignment_signal(tmp_path):
    run_dir, log_path = _record(
        tmp_path,
        profile="v49",
        source_mix_alignment_loss=0.0,
    )
    record = summarize_record(
        "v49-no-alignment", str(run_dir), str(log_path),
        3, 128, profile="v49",
    )

    assert record["pass"] is False


def test_v50_sacr_smoke_requires_active_structured_source(tmp_path):
    run_dir, log_path = _record(tmp_path, profile="v50_sacr")
    passing = summarize_record(
        "v50", str(run_dir), str(log_path), 3, 128,
        profile="v50_sacr",
    )
    assert passing["pass"] is True
    assert build_summary(
        [passing], 3, 128, profile="v50_sacr"
    )["schema"] == "mcln-v50-sacr-smoke-panel-v1"

    failed_root = tmp_path / "inactive-sacr"
    failed_root.mkdir()
    failed_run, failed_log = _record(
        failed_root, profile="v50_sacr", sacr_valid_ratio=0.0,
    )
    failed = summarize_record(
        "v50-inactive", str(failed_run), str(failed_log), 3, 128,
        profile="v50_sacr",
    )
    assert failed["pass"] is False


def test_v44_gate_requires_nonzero_candidate_mask_coverage(tmp_path):
    run_dir, log_path = _record(
        tmp_path, profile="v43", candidate_mask_diagnostic=0.0
    )
    record = summarize_record(
        "v44-collapsed-candidates", str(run_dir), str(log_path),
        3, 128, profile="v43", require_candidate_mask=True,
    )

    assert record["candidate_mask_required"] is True
    assert record["pass"] is False


def test_v45_gate_requires_positive_lovasz_loss_only_when_enabled(tmp_path):
    run_dir, log_path = _record(
        tmp_path, profile="v43", candidate_lovasz_loss=0.0
    )
    control = summarize_record(
        "v45-control", str(run_dir), str(log_path),
        3, 128, profile="v43", require_candidate_mask=True,
    )
    enabled = summarize_record(
        "v45-enabled", str(run_dir), str(log_path),
        3, 128, profile="v43", require_candidate_mask=True,
        require_candidate_lovasz=True,
    )

    assert control["pass"] is True
    assert control["candidate_lovasz_required"] is False
    assert enabled["pass"] is False
    assert enabled["candidate_lovasz_required"] is True


def test_atomic_summary_replaces_existing_file(tmp_path):
    output = tmp_path / "summary.json"
    output.write_text("stale", encoding="utf-8")
    atomic_write_json(str(output), {"schema": "test", "pass": True})
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "schema": "test",
        "pass": True,
    }
