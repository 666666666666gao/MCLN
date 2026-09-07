import pytest

from scripts.queue_scanrefer_native_box_transfer_posttraining import validated_module_pass


def completed(baseline, control, candidate, expected):
    def metrics(pair):
        return dict(zip(['rec_hits025', 'rec_hits050'], pair))
    receipt = {'schema': 'mcln-scanrefer-native-box-transfer-pair-v1', 'status': 'complete',
        'formal_rows': 0, 'steps_per_arm': 2482, 'holdout_rows': 6887,
        'baseline_metrics': {'gt_teacher_box': metrics(baseline)},
        'terminal_metrics': {'gt_only': metrics(control), 'gt_teacher_box': metrics(candidate)},
        'eligible_for_fixed_terminal_formal_evaluation': expected}
    audit = {'status': 'pass', 'formal_rows': 0, 'eligible_for_fixed_terminal_formal_evaluation': expected}
    return receipt, audit


def test_no_formal_when_candidate_loses_to_gt_only_despite_improving_start():
    receipt, audit = completed((6684, 6426), (6684, 6449), (6684, 6442), False)
    assert not validated_module_pass(receipt, audit)


def test_fixed_candidate_can_continue_on_both_nonregression_checks():
    receipt, audit = completed((6684, 6426), (6682, 6440), (6684, 6440), True)
    assert validated_module_pass(receipt, audit)


def test_incomplete_training_cannot_start_formal():
    receipt, audit = completed((6684, 6426), (6682, 6440), (6684, 6440), True)
    receipt['steps_per_arm'] = 64
    with pytest.raises(AssertionError):
        validated_module_pass(receipt, audit)


def test_disagreeing_integrity_receipt_cannot_authorize_formal():
    receipt, audit = completed((6684, 6426), (6682, 6440), (6684, 6440), True)
    audit['eligible_for_fixed_terminal_formal_evaluation'] = False
    with pytest.raises(AssertionError):
        validated_module_pass(receipt, audit)
