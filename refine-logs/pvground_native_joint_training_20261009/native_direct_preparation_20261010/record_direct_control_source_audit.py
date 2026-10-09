import datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
controls = root / 'native_direct_controls_20261010'
audit_path = controls / 'actual_source_review/EXPERIMENT_AUDIT.json'
audit = json.loads(audit_path.read_bytes())
assert audit['verdict'] == 'WARN' and audit['blocking_issue_count'] == 0
assert audit['execution_scope'] == 'ISOLATED_NATIVE_CONTROL_SOURCE_NOT_LAUNCHED'
for name, digest in audit['audited_input_hashes'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest.removeprefix('sha256:')
transport_exit = json.loads((controls / 'CPU_TRANSPORT_EXIT.json').read_bytes())
assert transport_exit['exit_code'] == 255
assert (controls / 'CPU_REMOTE_RAW_STDOUT.json').stat().st_size == 0
receipt = dict(status='SOURCE_REVIEW_COMPLETE_CPU_TRANSPORT_NO_RECEIPT',
    time_cst=datetime.datetime.now().astimezone().isoformat(), source_audit=str(audit_path),
    source_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
    source_verdict=audit['verdict'], source_blocking_issues=0,
    reviewed_input_count=len(audit['audited_input_hashes']), all_reviewed_hashes_reverified=True,
    CPU_transport_exit_code=255, CPU_remote_execution_observed=False,
    CPU_remote_directory_state='UNKNOWN_UNTIL_EXACT_READ_ONLY_WITNESS',
    normal_training_restart_calls=0, normal_training_status_reads=0,
    full_PV_constructor_and_GPU_control_validation_pending=True,
    acceptance_status='provisional', review_independence='same-family',
    actual_reviewer_model='UNATTESTED', full_goal_complete=False)
(controls / 'SOURCE_AUDIT_AND_CPU_TRANSPORT_STATE.json').write_text(
    json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
for path in (root.parent.parent / 'pvground_referit_mask_reference_20261006/current_research_goals.json',
             root.parent.parent / 'pvground_query_supported_geometry_20261005/active_continuation_state.json'):
    value = json.loads(path.read_bytes())
    value['normal_training_requirement']['native_direct_control_source_review'] = receipt
    value.update(current_turn_classification='ACTUAL_PROGRESS_SOURCE_CONTROL_AUDIT_WARN_ZERO_BLOCKS',
                 full_goal_complete=False)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
note = ('\nPV-Ground ' + receipt['time_cst'] + ': fresh source audit WARN / 0 CPU-source blocks; '
    '49 inputs sealed/reverified, only3 sources changed in isolated controls. Actual reviewerUNATTESTED '
    'same-family/provisional. CPU transportattempt exited255 withConnectionClosed andemptystdout; '
    'no CPU executionreceipt, actualremoteCPUrootunknown pending exactreadonlywitness. '
    'No normalNNstatusquery/restart/GPUcall; do not infer trainingterminal fromSSHfailure. '
    'FullPV/GPU/criterion/optimizer/coldrestore remainpending. FullgoalACTIVE_UNMET.\n')
for path in (Path('C:/Users/gb/memory/2026-10-10.md'), Path('C:/Users/gb/MEMORY.md')):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(note)
print(json.dumps(dict(status=receipt['status'], audited_inputs=receipt['reviewed_input_count'],
    current_normal_training_queried=False, source_mutations=0)))
