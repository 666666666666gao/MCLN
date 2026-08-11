from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/tuning/run_optuna_mcln_source_choice_continue20.sh"


def test_optuna_continue_uses_protected_acc025_successor():
    text = SCRIPT.read_text()
    assert (
        "ACC25_CKPT=${ACC25_CKPT:-${DATA_ROOT%/}/protected_mcln_artifacts/"
        "scanrefer_best_backbone_acc025_0.582878_component.pth}"
    ) in text
    assert "ACC25_CKPT=${ACC25_CKPT:-${SOURCE_RUN_DIR}/" not in text
    assert (
        "ACC50_CKPT=${ACC50_CKPT:-${SOURCE_RUN_DIR}/"
        "best_available_rec_acc050_epoch68.pth}"
    ) in text
