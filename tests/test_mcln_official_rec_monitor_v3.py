import importlib.util
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mcln_official_rec_monitor_v3.py"


def _load_monitor():
    spec = importlib.util.spec_from_file_location(
        "mcln_official_rec_monitor_v3", str(SCRIPT)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MONITOR = _load_monitor()


def _write_receipt(
        root,
        relative_dir,
        epoch,
        sample_count,
        hits025,
        hits050,
        subgroup_sample_count=None,
        acc025=None,
        acc050=None,
        mtime_ns=None):
    directory = root / relative_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "eval_metrics_epoch_{}.json".format(epoch)
    subgroup_count = (
        sample_count
        if subgroup_sample_count is None
        else subgroup_sample_count
    )
    payload = {
        "schema": MONITOR.RECEIPT_SCHEMA,
        "sample_count": sample_count,
        "position_subgroups": {
            "unique": {
                "sample_count": 0,
                "hits025": 0,
                "hits050": 0,
                "acc025": 0.0,
                "acc050": 0.0,
            },
            "multiple": {
                "sample_count": subgroup_count,
                "hits025": hits025,
                "hits050": hits050,
                "acc025": (
                    hits025 / float(subgroup_count)
                    if acc025 is None
                    else acc025
                ),
                "acc050": (
                    hits050 / float(subgroup_count)
                    if acc050 is None
                    else acc050
                ),
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    if mtime_ns is not None:
        os.utime(str(path), ns=(mtime_ns, mtime_ns))
    return path


def test_load_rows_accepts_only_exact_nr3d_formal_receipts(tmp_path):
    formal = _write_receipt(
        tmp_path, "formal", 57, 7899, 4475, 3759
    )
    _write_receipt(
        tmp_path, "scene_audit", 58, 6205, 5975, 5069
    )
    _write_receipt(
        tmp_path,
        "subgroup_mismatch",
        59,
        7899,
        4400,
        3700,
        subgroup_sample_count=6205,
    )
    _write_receipt(
        tmp_path,
        "invalid_accuracy",
        60,
        7899,
        4400,
        3700,
        acc025=0.99,
    )

    rows, ignored, invalid = MONITOR.load_rows(str(tmp_path), 7899)

    assert len(rows) == 1
    assert rows[0]["receipt"] == str(formal)
    assert rows[0]["hits025"] == 4475
    assert ignored == 2
    assert invalid == 1


def test_impossible_hits_and_wrong_schema_are_invalid(tmp_path):
    impossible = _write_receipt(
        tmp_path, "impossible", 57, 7899, 4000, 4100
    )
    wrong_schema = _write_receipt(
        tmp_path, "wrong_schema", 58, 7899, 4400, 3700
    )
    payload = json.loads(wrong_schema.read_text(encoding="utf-8"))
    payload["schema"] = "unreviewed-metrics-v9"
    wrong_schema.write_text(json.dumps(payload), encoding="utf-8")

    rows, ignored, invalid = MONITOR.load_rows(str(tmp_path), 7899)

    assert rows == []
    assert ignored == 0
    assert invalid == 2
    assert impossible.exists()


def test_update_replaces_contaminated_audit_state_with_formal_rows(tmp_path):
    best = _write_receipt(
        tmp_path, "formal_best", 57, 7899, 4475, 3759,
        mtime_ns=1_000_000_000
    )
    latest = _write_receipt(
        tmp_path, "formal_latest", 59, 7899, 4400, 3700,
        mtime_ns=2_000_000_000
    )
    audit = _write_receipt(
        tmp_path, "audit", 58, 6205, 5975, 5000,
        mtime_ns=3_000_000_000
    )
    control = tmp_path / "control"
    control.mkdir()
    state = {
        "dataset": "nr3d",
        "target_rec025": 0.600,
        "preserved_best_rec025": 4475 / 7899.0,
        "preserved_best_epoch": 57,
        "preserved_checkpoint": str(control / "protected_e57.pth"),
        "last_receipt": str(audit),
        "latest_epoch": 58,
        "latest_rec025": 5975 / 6205.0,
        "metric_best_receipt_epoch": 58,
        "metric_best_receipt_rec025": 5975 / 6205.0,
        "metric_best_receipt": str(audit),
    }

    updated, changed = MONITOR._update_state_once(
        state, str(tmp_path), str(control), 7899
    )

    assert changed["receipt"] == str(latest)
    assert updated["last_receipt"] == str(latest)
    assert updated["latest_epoch"] == 59
    assert updated["latest_rec025"] == pytest.approx(4400 / 7899.0)
    assert updated["metric_best_receipt_epoch"] == 57
    assert updated["metric_best_receipt_rec025"] == pytest.approx(
        4475 / 7899.0
    )
    assert updated["metric_best_receipt"] == str(best)
    assert updated["formal_receipt_count"] == 2
    assert updated["ignored_nonformal_receipt_count"] == 1
    assert updated["invalid_receipt_count"] == 0
    assert updated["target_reached"] is False


def test_no_formal_rows_clear_stale_formal_receipt_fields(tmp_path):
    audit = _write_receipt(
        tmp_path, "audit", 58, 6205, 5975, 5000
    )
    control = tmp_path / "control"
    control.mkdir()
    state = {
        "dataset": "nr3d",
        "target_rec025": 0.600,
        "preserved_best_rec025": 4475 / 7899.0,
        "last_receipt": str(audit),
        "latest_epoch": 58,
        "latest_rec025": 5975 / 6205.0,
        "latest_rec050": 5000 / 6205.0,
        "metric_best_receipt_epoch": 58,
        "metric_best_receipt_rec025": 5975 / 6205.0,
        "metric_best_receipt": str(audit),
    }

    updated, changed = MONITOR._update_state_once(
        state, str(tmp_path), str(control), 7899
    )

    assert changed is None
    assert updated["formal_receipt_count"] == 0
    assert updated["ignored_nonformal_receipt_count"] == 1
    assert updated["last_receipt"] is None
    assert updated["target_reached"] is False
    for key in (
            "latest_epoch",
            "latest_rec025",
            "latest_rec050",
            "metric_best_receipt_epoch",
            "metric_best_receipt_rec025",
            "metric_best_receipt"):
        assert key not in updated


def test_target_requires_strictly_exceeding_reviewed_threshold(tmp_path):
    control = tmp_path / "control"
    control.mkdir()
    state = {
        "dataset": "nr3d",
        "target_rec025": 0.600,
        "preserved_best_rec025": 0.600,
        "preserved_best_epoch": 60,
        "preserved_checkpoint": str(control / "protected_e60.pth"),
    }

    updated, changed = MONITOR._update_state_once(
        state, str(tmp_path), str(control), 7899
    )

    assert changed is None
    assert updated["target_reached"] is False


def test_v2_preserved_fields_cannot_override_reviewed_initial_state(tmp_path):
    formal = _write_receipt(
        tmp_path, "formal", 57, 7899, 4475, 3759
    )
    rows, ignored, invalid = MONITOR.load_rows(str(tmp_path), 7899)
    assert len(rows) == 1
    assert ignored == 0
    assert invalid == 0
    initial_checkpoint = str(tmp_path / "official_e57.pth")
    previous_v2 = {
        "schema": MONITOR.STATE_SCHEMA,
        "dataset": "nr3d",
        "preserved_best_rec025": 0.9629331184528606,
        "preserved_best_epoch": 58,
        "preserved_checkpoint": str(tmp_path / "audit_e58.pth"),
    }

    preserved = MONITOR._validated_preserved_state(
        previous_v2,
        rows,
        str(tmp_path),
        7899,
        4475 / 7899.0,
        57,
        initial_checkpoint,
        "a" * 64,
    )

    assert preserved == {
        "preserved_best_rec025": 4475 / 7899.0,
        "preserved_best_epoch": 57,
        "preserved_checkpoint": initial_checkpoint,
        "preserved_checkpoint_sha256": "a" * 64,
    }
    assert formal.exists()


def test_valid_v3_formal_best_survives_monitor_restart(tmp_path):
    _write_receipt(tmp_path, "formal", 61, 7899, 4800, 3900)
    rows, _, _ = MONITOR.load_rows(str(tmp_path), 7899)
    best = 4800 / 7899.0
    checkpoint = tmp_path / MONITOR._preserved_checkpoint_name(61, best)
    MONITOR.torch.save({"epoch": 61, "model": {}}, str(checkpoint))
    checkpoint_sha256 = MONITOR._validated_checkpoint_identity(
        str(checkpoint), 61
    )
    previous_v3 = {
        "schema": MONITOR.STATE_SCHEMA,
        "dataset": "nr3d",
        "receipt_scope": MONITOR.RECEIPT_SCOPE,
        "expected_sample_count": 7899,
        "preserved_best_rec025": best,
        "preserved_best_epoch": 61,
        "preserved_checkpoint": str(checkpoint),
        "preserved_checkpoint_sha256": checkpoint_sha256,
    }

    preserved = MONITOR._validated_preserved_state(
        previous_v3,
        rows,
        str(tmp_path),
        7899,
        4475 / 7899.0,
        57,
        str(tmp_path / "initial_e57.pth"),
        "b" * 64,
    )

    assert preserved == {
        "preserved_best_rec025": best,
        "preserved_best_epoch": 61,
        "preserved_checkpoint": str(checkpoint),
        "preserved_checkpoint_sha256": checkpoint_sha256,
    }


@pytest.mark.parametrize("checkpoint_kind", ["corrupt", "wrong_epoch"])
def test_invalid_v3_checkpoint_falls_back_to_reviewed_initial(
        tmp_path, checkpoint_kind):
    _write_receipt(tmp_path, "formal", 61, 7899, 4800, 3900)
    rows, _, _ = MONITOR.load_rows(str(tmp_path), 7899)
    best = 4800 / 7899.0
    checkpoint = tmp_path / MONITOR._preserved_checkpoint_name(61, best)
    if checkpoint_kind == "corrupt":
        checkpoint.write_bytes(b"not-a-checkpoint")
    else:
        MONITOR.torch.save({"epoch": 60, "model": {}}, str(checkpoint))
    with checkpoint.open("rb") as handle:
        checkpoint_sha256 = MONITOR._hash_open_file(handle)
    previous_v3 = {
        "schema": MONITOR.STATE_SCHEMA,
        "dataset": "nr3d",
        "receipt_scope": MONITOR.RECEIPT_SCOPE,
        "expected_sample_count": 7899,
        "preserved_best_rec025": best,
        "preserved_best_epoch": 61,
        "preserved_checkpoint": str(checkpoint),
        "preserved_checkpoint_sha256": checkpoint_sha256,
    }
    initial_checkpoint = str(tmp_path / "initial_e57.pth")

    preserved = MONITOR._validated_preserved_state(
        previous_v3,
        rows,
        str(tmp_path),
        7899,
        4475 / 7899.0,
        57,
        initial_checkpoint,
        "b" * 64,
    )

    assert preserved == {
        "preserved_best_rec025": 4475 / 7899.0,
        "preserved_best_epoch": 57,
        "preserved_checkpoint": initial_checkpoint,
        "preserved_checkpoint_sha256": "b" * 64,
    }


def test_checkpoint_publish_validates_copied_inode(tmp_path):
    receipt_dir = tmp_path / "run"
    receipt_dir.mkdir()
    receipt = receipt_dir / "eval_metrics_epoch_61.json"
    receipt.write_text("{}", encoding="utf-8")
    source = receipt_dir / "ckpt_epoch_61.pth"
    MONITOR.torch.save({"epoch": 61, "model": {"value": 3}}, str(source))
    control = tmp_path / "control"
    control.mkdir()
    row = {
        "receipt": str(receipt),
        "epoch": 61,
        "rec025": 0.61,
    }

    destination, destination_sha256 = MONITOR._copy_valid_checkpoint(
        row, str(control)
    )

    assert destination == str(
        control / "official_best_rec025_epoch_61_0p61000000.pth"
    )
    assert os.stat(str(source)).st_ino != os.stat(destination).st_ino
    assert os.stat(destination).st_mode & 0o777 == 0o444
    loaded = MONITOR.torch.load(destination, map_location="cpu")
    assert loaded["epoch"] == 61
    assert loaded["model"]["value"] == 3
    assert destination_sha256 == MONITOR._validated_checkpoint_identity(
        destination, 61
    )


def test_single_instance_lock_rejects_second_monitor(tmp_path):
    first = MONITOR._acquire_single_instance_lock(str(tmp_path))
    try:
        with pytest.raises(RuntimeError, match="holds the lock"):
            MONITOR._acquire_single_instance_lock(str(tmp_path))
    finally:
        first.close()


@pytest.mark.parametrize(
    "dataset, expected",
    [("nr3d", 7899), ("sr3d", 17726)],
)
def test_dataset_sample_count_contract_is_fixed(dataset, expected):
    assert MONITOR.DATASET_FORMAL_SAMPLE_COUNTS[dataset] == expected


def test_main_rejects_scene_audit_count_for_nr3d(monkeypatch):
    monkeypatch.setattr(
        MONITOR.sys,
        "argv",
        [
            str(SCRIPT),
            "nr3d",
            "/unused/root",
            "/unused/control",
            "0.600",
            str(4475 / 7899.0),
            "57",
            "/unused/e57.pth",
            "6205",
        ],
    )

    with pytest.raises(SystemExit, match="must be 7899"):
        MONITOR.main()
