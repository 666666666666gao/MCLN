from pathlib import Path

import pytest

from src.joint_det_dataset import resolve_referit3d_csv


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "train_scanrefer_joint_query_quality.sh"


def test_referit3d_csv_resolver_accepts_public_lowercase_layout(tmp_path):
    csv_path = tmp_path / "refer_it_3d" / "nr3d.csv"
    csv_path.parent.mkdir()
    csv_path.write_text("scan_id,target_id\n", encoding="ascii")

    assert resolve_referit3d_csv(str(tmp_path), "nr3d.csv") == str(csv_path)


def test_referit3d_csv_resolver_fails_with_checked_paths(tmp_path):
    with pytest.raises(FileNotFoundError, match="ReferIt3D CSV is missing"):
        resolve_referit3d_csv(str(tmp_path), "sr3d.csv")


def test_joint_query_launcher_has_fail_closed_dataset_contract():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'LANGUAGE_DATASET="${LANGUAGE_DATASET:-scanrefer}"' in text
    assert '--dataset "${LANGUAGE_DATASET}" --test_dataset "${TEST_DATASET}"' in text
    assert 'CHECKPOINT_PATH=<${LANGUAGE_DATASET} checkpoint> is required' in text
    assert 'EXPECTED_EVAL_SAMPLE_COUNT is required for ${TEST_DATASET}' in text
    assert '--dataset scanrefer --test_dataset scanrefer' not in text


def test_joint_query_launcher_keeps_parent_and_joint_source_pools_separate():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'JOINT_QUERY_QUALITY_SOURCE_NAMES="${JOINT_QUERY_QUALITY_SOURCE_NAMES:-}"' in text
    assert '--joint_query_quality_source_names "${JOINT_QUERY_QUALITY_SOURCE_NAMES}"' in text
    assert 'USE_SACR_SOURCE="${USE_SACR_SOURCE:-0}"' in text
    assert '--use_sacr_source' in text
    assert 'SACR requires sacr_structured in JOINT_QUERY_QUALITY_SOURCE_NAMES' in text
