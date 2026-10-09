import datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
controls = root / 'native_direct_controls_20261010'
cpu = controls / 'cpu_transport_attempt2'
execution = json.loads((cpu / 'CPU_MODULE_EXECUTION.json').read_bytes())
witness = json.loads((cpu / 'CPU_MODULE_WITNESS.json').read_bytes())
remote = json.loads((root / 'NATIVE_DIRECT_CONTROL_REMOTE_SYNC_ATTEMPT2_RECEIPT.json').read_bytes())
publication_path = root / 'native_direct_controls_publication.json'
publication = json.loads(publication_path.read_bytes())
assert execution['exit_code'] == 0 and execution['GPU_calls'] == 0
assert witness['formal_accuracy'] is None and witness['full_PV_constructor_or_1295_state_checked'] is False
assert remote['remote_sync_complete'] is True and remote['doc_sha256'] == publication['doc_sha256']
assert publication['remote_handoff_sync_complete'] is True and publication['github_main_verified'] is True
source_review_path = controls / 'actual_source_review/EXPERIMENT_AUDIT.json'
assert execution['source_audit_sha256'] == hashlib.sha256(source_review_path.read_bytes()).hexdigest()
source_review = json.loads(source_review_path.read_bytes())
for path, digest in source_review['audited_input_hashes'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest.removeprefix('sha256:')
report = dict(status='STATIC_HANDOFF_SYNC_AND_ACTUAL_CPU_MODULE_CHECKS_COMPLETE_OUTCOME_REVIEW_PENDING',
    time_cst=datetime.datetime.now().astimezone().isoformat(),
    section=publication['section'], remote_doc_sha256=remote['doc_sha256'], remote_evidence_files=remote['files'],
    remote_sync_time_cst=remote['time_cst'], original_failure_receipts_preserved=True,
    CPU_execution=str(cpu / 'CPU_MODULE_EXECUTION.json'), CPU_witness=str(cpu / 'CPU_MODULE_WITNESS.json'),
    CPU_execution_sha256=hashlib.sha256((cpu / 'CPU_MODULE_EXECUTION.json').read_bytes()).hexdigest(),
    CPU_witness_sha256=hashlib.sha256((cpu / 'CPU_MODULE_WITNESS.json').read_bytes()).hexdigest(),
    CPU_execution_finished_cst=execution['finished_cst'],
    CPU_fixture_type=witness['evaluation_type'], current_normal_training_queries=0,
    control_GPU_or_full_model_admission=False, actual_CPU_outcome_review_pending=True,
    new_normal_accuracy=None, full_goal_complete=False)
(root / 'STATIC_COPY_AND_CPU_RECOVERY.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
for path in (root.parent.parent / 'pvground_referit_mask_reference_20261006/current_research_goals.json',
             root.parent.parent / 'pvground_query_supported_geometry_20261005/active_continuation_state.json'):
    value = json.loads(path.read_bytes())
    assert value['latest_handoff_section'] == '20.376.130'
    value.update(remote_handoff_sync_pending=False, remote_handoff_last_confirmed_section='20.376.130',
        remote_handoff_last_confirmed_sha256=remote['doc_sha256'],
        current_turn_classification='ACTUAL_PROGRESS_SSH_STATIC_SYNC_RECOVERED_CPU_MODULE_CHECKS_EXECUTED',
        full_goal_complete=False)
    value['normal_training_requirement'].update(current_native_direct_control_publication=publication,
        current_native_control_CPU_checks=report,
        control_CPU_transport_status='ACTUAL_SECOND_ATTEMPT_COMPLETE_CPU_ENGINEERING_ONLY',
        pending_static_remote_sync_script=None)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
note = ('\nPV-Ground ' + report['time_cst'] + ': SSHreadonly05:49 verifiedremote129/exactoldSHA, '
    'no current67evidencefiles/CPUrootabsent. Explicitseparateattempt2 CPU completedexit0 '
    '05:50:36.915 CPU Torch1.10.2+cu111/CUDA_VISIBLE_DEVICES empty/synthetic1x256/8SP/50000points; '
    'actuald06/f989 states loaded, commonA/B oldmodule outputsequal, bypassA/query-maskpreserved, '
    'content6inputszero, fixedBhalf/frozen/limitedindependent-referencegradienthalf. '
    'Not fullPV/1295/nativecriterion/optimizer/coldrecovery/accuracy; actualCPUfreshreviewpending. '
    'Staticsection130 67files/docSHA' + remote['doc_sha256'] + ' remotecompleted05:50:43.994; '
    'allpriorfailedreceiptskept, oldsyncguardadvancedonlyafteractualremotecopy. '
    'No normalNNquery/GPU/restart, original08:19readerunchanged. FullgoalACTIVE_UNMET.\n')
for path in (Path('C:/Users/gb/memory/2026-10-10.md'), Path('C:/Users/gb/MEMORY.md')):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(note)
print(json.dumps(dict(status=report['status'], remote_sync_complete=True,
    actual_CPU_checks=True, new_formal_accuracy=None, normal_training_queried=False)))
