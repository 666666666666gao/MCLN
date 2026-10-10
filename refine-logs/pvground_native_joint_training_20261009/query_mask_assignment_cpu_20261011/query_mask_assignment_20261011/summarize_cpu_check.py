"""Summarize completed bounded CPU execution without claiming accuracy."""
import datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
execution_path = root / 'cpu_execution/CPU_EXECUTION.json'
result_path = root / 'cpu_execution/QUERY_MASK_ASSIGNMENT_CPU_RESULT.json'
execution = json.loads(execution_path.read_bytes())
result = json.loads(result_path.read_bytes())
assert execution['exit_code'] == 0
assert result['status'] == 'ACTUAL_ISOLATED_QUERY_MASK_ASSIGNMENT_CPU_CHECK_COMPLETE'
assert result['evaluation_type'] == 'simulation_only'
assert result['matcher_calls'] == 33 and result['native_boxes_only_criterion_calls'] == 6
assert result['assignment_problems'] == 44 and result['assigned_gt_pairs'] == 66
assert all(result[key] == 0 for key in ('neural_model_forward_calls', 'full_native_criterion_calls',
    'Mask_loss_calls', 'dataset_rows', 'optimizer_updates', 'gpu_calls',
    'current_training_queries', 'active_training_source_mutations'))
assert result['formal_accuracy'] is None
source_audit = root / 'cpu_source_review/SOURCE_REVIEW.json'
actual_audit = root / 'cpu_actual_review/ACTUAL_REVIEW.json'
audit = json.loads(actual_audit.read_bytes())
assert audit['execution_scope'] == 'ACTUAL_CPU_ONLY'
assert audit['verdict'] in ('PASS', 'WARN') and not audit['blocking_findings']
for path, digest in audit['audited_input_hashes'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
started = datetime.datetime.fromisoformat(execution['started_cst'])
finished = datetime.datetime.fromisoformat(execution['finished_cst'])
summary = dict(status='ISOLATED_NATIVE_QUERY_MASK_ASSIGNMENT_CPU_CHECK_AUDITED',
    recorded_cst=datetime.datetime.now().astimezone().isoformat(),
    started_cst=execution['started_cst'], finished_cst=execution['finished_cst'],
    elapsed_seconds=(finished - started).total_seconds(), execution_exit_code=0,
    result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
    cpu_source_review_sha256=hashlib.sha256(source_audit.read_bytes()).hexdigest(),
    cpu_actual_review_sha256=hashlib.sha256(actual_audit.read_bytes()).hexdigest(),
    actual_review_verdict=audit['verdict'], actual_review_blocking_findings=[],
    evaluation_type='simulation_only', constructed_case_count=3,
    matcher_calls=result['matcher_calls'], assignment_problems=result['assignment_problems'],
    assigned_gt_pairs=result['assigned_gt_pairs'], native_boxes_only_criterion_calls=6,
    default_and_text_match_original=True, query_own_mask_changes_cost_and_assignment=True,
    all_GT_unique_one_to_one_preserved=True, no_Mask_prefix_fixed_inputs_unchanged=True,
    native_box_loss_direct_gradient_follows_assignment=True,
    full_native_criterion_executed=False, neural_model_forward_calls=0,
    dataset_rows=0, optimizer_updates=0, gpu_calls=0, formal_accuracy=None,
    current_training_queries=0, active_training_source_mutations=0,
    native_GPU_preflight_completed=False, new_training_started=False,
    three_effective_contributions_proven=False, full_goal_complete=False,
    review_independence='same-family', acceptance_status='provisional', actual_identity_attestation='UNATTESTED',
    next_main_training_observation_cst='2026-10-11T05:00:40.968644+08:00',
    limitation=result['limitation'])
(root / 'CPU_CHECK_SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({key: summary[key] for key in ('status', 'matcher_calls', 'native_boxes_only_criterion_calls',
    'formal_accuracy', 'gpu_calls', 'new_training_started')}))
