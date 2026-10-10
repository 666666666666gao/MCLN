"""Record exact controlled-check evidence only after the fresh actual audit."""
import datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
audit_path = root / 'actual_review/ACTUAL_REVIEW.json'
audit = json.loads(audit_path.read_bytes())
assert audit['execution_scope'] == 'ACTUAL_NATIVE_MATCHER_CONTROLLED_CPU_ONLY'
assert audit['verdict'] in ('PASS', 'WARN') and not audit['blocking_findings']
for path, digest in audit['audited_input_hashes'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
result_path = root / 'cpu_execution/MATCHER_CPU_RESULT.json'
result = json.loads(result_path.read_bytes())
receipt = json.loads((root / 'cpu_execution/CPU_EXECUTION.json').read_bytes())
assert receipt['exit_code'] == 0
assert result['status'] == 'ACTUAL_NATIVE_MATCHER_CONTROLLED_CPU_CHECK_COMPLETE'
assert result['fixture_count'] == result['matcher_calls'] == len(result['records']) == 9
assert result['formal_accuracy'] is None and result['evaluation_type'] == 'simulation_only'
for key in ('neural_model_forward_calls', 'criterion_calls', 'optimizer_updates', 'dataset_rows', 'gpu_calls',
            'active_training_source_mutations', 'current_training_queries'):
    assert result[key] == 0
batch_problems = sum(len(record['assignments']) for record in result['records'])
gt_pairs = sum(len(row['gt']) for record in result['records'] for row in record['assignments'])
assert (batch_problems, gt_pairs) == (11, 16)
own_checks = [row for row in result['checks'] if 'maximum_native_cost_difference' in row]
column_checks = [row for row in result['checks'] if 'maximum_column_shift_spread' in row]
assert len(own_checks) == len(column_checks) == 3
assert all(value == 0 for row in own_checks for value in row['maximum_native_cost_difference'])
maximum_spread = max(row['maximum_column_shift_spread'] for row in column_checks)
assert maximum_spread < 1e-6
summary = dict(status='ACTUAL_NATIVE_MATCHER_CPU_CHECK_AUDITED',
    recorded_cst=datetime.datetime.now().astimezone().isoformat(),
    actual_started_cst=receipt['started_cst'], actual_finished_cst=receipt['finished_cst'],
    original_helper_functions_cell=706, original_helper_exit_code=0, original_handle_consumed=True,
    fixture_count=9, matcher_calls=9, batch_assignment_problems=batch_problems, gt_assignments=gt_pairs,
    evaluation_type='simulation_only', source_review_verdict=json.loads((root/'source_review/SOURCE_REVIEW.json').read_bytes())['verdict'],
    actual_review_verdict=audit['verdict'], actual_review_blocking_findings=[],
    review_independence='same-family', acceptance_status='provisional', actual_identity_attestation='UNATTESTED',
    actual_review_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
    actual_result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
    own_mask_change_native_cost_max_difference=0,
    shared_text_change_max_column_shift_spread=maximum_spread,
    captured_cost_matrices_persisted=False, cost_difference_summaries_and_assignments_persisted=True,
    query_mask_indirect_geometry_path_preserved=True, training_degradation_cause_proven=False,
    formal_accuracy=None, gpu_calls=0, neural_model_forward_calls=0, criterion_calls=0,
    optimizer_updates=0, dataset_rows=0, current_training_queries=0,
    active_training_source_mutations=0, new_matcher_strategy_implemented=False,
    full_goal_complete=False, next_main_observation_cst='2026-10-11T05:00:40.968644+08:00')
(root / 'ACTUAL_CHECK_SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({key: summary[key] for key in ('status','fixture_count','matcher_calls',
    'batch_assignment_problems','gt_assignments','actual_review_verdict','formal_accuracy','next_main_observation_cst')}))
