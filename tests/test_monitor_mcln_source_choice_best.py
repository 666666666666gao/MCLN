from pathlib import Path

from scripts.monitor_mcln_source_choice_best import preserve_best


def write_log(path: Path):
    path.write_text(
        "\n".join(
            [
                "[00:00] root INFO: epoch 71, total time 1.00",
                (
                    "[00:01] root INFO: learned_selector Acc0.25 Top-1: "
                    "0.57730, Acc0.50 Top-1: 0.46161"
                ),
                (
                    "[00:01] root INFO: oracle Acc0.25 Top-1: "
                    "0.58162, Acc0.50 Top-1: 0.46824"
                ),
            ]
        )
    )


def write_multi_epoch_log(path: Path):
    path.write_text(
        "\n".join(
            [
                "[00:00] root INFO: epoch 71, total time 1.00",
                (
                    "[00:01] root INFO: learned_selector Acc0.25 Top-1: "
                    "0.57730, Acc0.50 Top-1: 0.46161"
                ),
                (
                    "[00:01] root INFO: oracle Acc0.25 Top-1: "
                    "0.58162, Acc0.50 Top-1: 0.46824"
                ),
                "[00:02] root INFO: epoch 75, total time 1.00",
                (
                    "[00:03] root INFO: learned_selector Acc0.25 Top-1: "
                    "0.57699, Acc0.50 Top-1: 0.46592"
                ),
                (
                    "[00:03] root INFO: oracle Acc0.25 Top-1: "
                    "0.58183, Acc0.50 Top-1: 0.47160"
                ),
            ]
        )
    )


def write_log_with_non_best_latest(path: Path):
    path.write_text(
        "\n".join(
            [
                "[00:00] root INFO: epoch 71, total time 1.00",
                (
                    "[00:01] root INFO: learned_selector Acc0.25 Top-1: "
                    "0.57730, Acc0.50 Top-1: 0.46161"
                ),
                (
                    "[00:01] root INFO: oracle Acc0.25 Top-1: "
                    "0.58162, Acc0.50 Top-1: 0.46824"
                ),
                "[00:02] root INFO: epoch 77, total time 1.00",
                (
                    "[00:03] root INFO: learned_selector Acc0.25 Top-1: "
                    "0.57152, Acc0.50 Top-1: 0.46109"
                ),
                (
                    "[00:03] root INFO: oracle Acc0.25 Top-1: "
                    "0.57636, Acc0.50 Top-1: 0.46634"
                ),
            ]
        )
    )


def test_preserve_best_does_not_prune_unvalidated_future_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    preserve_dir = tmp_path / "preserved"
    run_dir.mkdir()
    write_log(run_dir / "log.txt")

    for epoch in (71, 72):
        (run_dir / f"ckpt_epoch_{epoch}.pth").write_bytes(
            f"epoch {epoch}".encode("ascii")
        )

    preserve_best(run_dir, preserve_dir, keep_latest=1)

    assert (run_dir / "ckpt_epoch_71.pth").exists()
    assert (run_dir / "ckpt_epoch_72.pth").exists()
    assert list(preserve_dir.glob("*acc050_epoch71_0.46161.pth"))


def test_preserve_best_keeps_best_available_when_log_best_checkpoint_missing(tmp_path):
    run_dir = tmp_path / "run"
    preserve_dir = tmp_path / "preserved"
    run_dir.mkdir()
    write_multi_epoch_log(run_dir / "log.txt")
    (run_dir / "ckpt_epoch_71.pth").write_bytes(b"epoch 71")
    preserve_dir.mkdir()
    stale = preserve_dir / "mcln_contrastive_text_best_acc050_epoch71_0.46161.pth"
    stale.write_bytes(b"stale")

    preserve_best(run_dir, preserve_dir, keep_latest=1)

    assert list(
        preserve_dir.glob("*best_available_acc050_epoch71_0.46161.pth")
    )
    assert not stale.exists()


def test_preserve_best_prunes_validated_checkpoint_without_metric_refresh(tmp_path):
    run_dir = tmp_path / "run"
    preserve_dir = tmp_path / "preserved"
    run_dir.mkdir()
    write_log_with_non_best_latest(run_dir / "log.txt")

    for epoch in (71, 77):
        (run_dir / f"ckpt_epoch_{epoch}.pth").write_bytes(
            f"epoch {epoch}".encode("ascii")
        )

    preserve_best(run_dir, preserve_dir, keep_latest=1)

    assert (run_dir / "ckpt_epoch_71.pth").exists()
    assert not (run_dir / "ckpt_epoch_77.pth").exists()


def test_preserve_best_keeps_checkpoint_that_refreshes_either_metric(tmp_path):
    run_dir = tmp_path / "run"
    preserve_dir = tmp_path / "preserved"
    run_dir.mkdir()
    write_multi_epoch_log(run_dir / "log.txt")

    for epoch in (71, 75):
        (run_dir / f"ckpt_epoch_{epoch}.pth").write_bytes(
            f"epoch {epoch}".encode("ascii")
        )

    preserve_best(run_dir, preserve_dir, keep_latest=1)

    assert (run_dir / "ckpt_epoch_71.pth").exists()
    assert (run_dir / "ckpt_epoch_75.pth").exists()
    assert list(preserve_dir.glob("*best_acc050_epoch75_0.46592.pth"))


def test_preserve_best_prunes_checkpoint_below_baseline_metrics(tmp_path):
    run_dir = tmp_path / "run"
    preserve_dir = tmp_path / "preserved"
    run_dir.mkdir()
    write_multi_epoch_log(run_dir / "log.txt")

    for epoch in (71, 75):
        (run_dir / f"ckpt_epoch_{epoch}.pth").write_bytes(
            f"epoch {epoch}".encode("ascii")
        )

    preserve_best(
        run_dir,
        preserve_dir,
        keep_latest=1,
        baseline_acc025=0.58,
        baseline_acc050=0.47,
    )

    assert not (run_dir / "ckpt_epoch_71.pth").exists()
    assert not (run_dir / "ckpt_epoch_75.pth").exists()
    assert not list(preserve_dir.glob("*.pth"))
