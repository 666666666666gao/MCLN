"""Synthetic file fixtures test the launch boundary, never benchmark results."""

import json

import pytest

from scripts import evaluate_scanrefer_range_official as evaluation
from scripts.queue_native_candidate_range_preflight import sha, validate_formal


def write(path, value):
    path.write_text(json.dumps(value))


def formal_fixture(root, candidate_hits=(5572, 4797), wrong_point_identity=False):
    (root / 'result').mkdir()
    (root / 'controller.exit').write_text('0\n')
    write(root / 'input_manifest.json', {'candidate_predeclared': 'local_v99'})
    rows = {}
    for arm, counts in [('protected_v99', (5572, 4797)), ('center_v99', (5700, 4900)),
                        ('local_v99', candidate_hits)]:
        rows[arm] = [{'row_id': i, 'scan_id': 'synthetic_fixture', 'physical_space': 'synthetic_fixture',
            'point_sha256': '0' * 64,
            'rec_iou': .75 if i < counts[1] else .3 if i < counts[0] else 0.,
            'mask_iou': .9 if i < 4975 else .3 if i < 5689 else 0.} for i in range(9508)]
    if wrong_point_identity:
        rows['local_v99'][-1]['point_sha256'] = '1' * 64
    write(root / 'result/rows.json', rows)
    metrics = {arm: evaluation.row_metrics(value) for arm, value in rows.items()}
    promotion = evaluation.promotion_check(metrics['protected_v99'], metrics['local_v99'])
    receipt = {'schema': 'mcln-scanrefer-range-official-v1', 'status': 'complete', 'formal_rows': 9508,
        'optimizer_steps': 0, 'checkpoint_writes': 0, 'all_model_states_unchanged': True,
        'native_evaluators_match_row_metrics': True, 'manifest_sha256': sha(root / 'input_manifest.json'),
        'rows_sha256': sha(root / 'result/rows.json'), 'metrics': metrics, 'promotion': promotion,
        'data_root': '/synthetic-fixture/'}
    write(root / 'result/receipt.json', receipt)
    audit = {'schema': 'mcln-scanrefer-range-official-audit-v1', 'formal_rows': 9508, 'integrity_pass': True,
        'receipt_sha256': sha(root / 'result/receipt.json'), 'metrics': metrics, 'promotion': promotion}
    write(root / 'result/independent_audit.json', audit)
    return {'formal_receipt_sha256': sha(root / 'result/receipt.json'),
        'formal_audit_sha256': sha(root / 'result/independent_audit.json'), 'metrics': metrics,
        'promotion': promotion, 'native_range_preflight_required': promotion['advance_to_nr3d_sr3d_rec']}


def test_center_cannot_replace_the_predeclared_extent_candidate(tmp_path):
    decision = formal_fixture(tmp_path, candidate_hits=(5572, 4796))
    promotion, metrics, _ = validate_formal(tmp_path, decision, evaluation)
    assert metrics['center_v99']['rec_hits050'] > metrics['protected_v99']['rec_hits050']
    assert not promotion['advance_to_nr3d_sr3d_rec']
    assert not promotion['checks']['rec050_historical_v99']


def test_exact_existing_floors_pass_without_requiring_stretch_goals(tmp_path):
    decision = formal_fixture(tmp_path)
    promotion, _, _ = validate_formal(tmp_path, decision, evaluation)
    assert promotion['advance_to_nr3d_sr3d_rec']
    assert promotion['nr3d_sr3d_mask_not_a_promotion_gate']


def test_matching_counts_cannot_hide_different_point_inputs(tmp_path):
    decision = formal_fixture(tmp_path, wrong_point_identity=True)
    with pytest.raises(AssertionError):
        validate_formal(tmp_path, decision, evaluation)
