"""Prepare section140 from existing actual receipts; no training query."""
import datetime
import json
from pathlib import Path

root = Path(__file__).resolve().parent
control = root / 'normal_controls_20261010'
intake = json.loads((control/'ENGINEERING_INTAKE.json').read_bytes())
receipt = json.loads((control/'actual_engineering/witness/NATIVE_M0_RECEIPT.json').read_bytes())
audit = json.loads((control/'actual_review/EXPERIMENT_AUDIT.json').read_bytes())
launch = json.loads((control/'NORMAL_LAUNCH.json').read_bytes())
start = json.loads((control/'NORMAL_START_WITNESS.json').read_bytes())
observer = json.loads((control/'NORMAL_OBSERVER_OWNER.json').read_bytes())
dedup = json.loads((root/'normal_degradation_assessment_20261010/OLD_INITIAL_ARRAY_DEDUPLICATION.json').read_bytes())
assert intake['remote_status']['status']=='complete' and receipt['actual_updates']==2
assert receipt['initial_model_state_exactly_matches_normal_E0'] and receipt['full_recovery_exact']
assert audit['execution_scope']=='ACTUAL_NATIVE_C_OFF_PREFLIGHT' and not audit['blocking_findings']
assert start['controller_pid']==launch['controller_pid']==observer['remote_controller_pid']
assert launch['admission']['selected_mask_supervision_disabled']
assert dedup['all_five_paths_and_bytes_preserved'] and dedup['duplicate_file_bytes']==169255424
summary = json.loads((root/'NORMAL_TERMINAL_SUMMARY_20261010.json').read_bytes())
summary.update(status='C_OFF_ACTUAL_PREFLIGHT_COMPLETE_AND_NORMAL_NATIVE_ENTRY_LAUNCHED',
    prepared_cst=datetime.datetime.now().astimezone().isoformat(),
    c_off_actual_M0=intake['remote_status'], c_off_actual_M0_receipt=receipt,
    c_off_actual_audit=dict(verdict=audit['verdict'],execution_scope=audit['execution_scope'],
        blocking_findings=[],actual_runtime_model='UNATTESTED',independence='same-family',acceptance='provisional'),
    duplicate_old_initial_array_deduplication=dedup,
    c_off_normal_launch=launch,c_off_start_witness=start,c_off_observer_owner=observer,
    c_off_normal_training_controller_started=True,c_off_optimizer_updates_observed=False,
    c_off_new_formal_accuracy=None,c_off_engineering_checkpoint_retired=launch['retirement'],
    historical_pretrained_parents_already_C_adapted=True,
    no_history_never_C_claim=True,active_training_source_modified=False,
    publication_neural_calls=0,publication_training_status_queries=0,
    full_goal_complete=False)
(control/'C_OFF_NORMAL_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
names = ['normal_controls_20261010/'+name for name in (
    'HANDOFF_C_OFF_NORMAL_20261010.md','C_OFF_NORMAL_SUMMARY.json','ENGINEERING_INTAKE.json',
    'actual_engineering/engineering_status.json','actual_engineering/preflight.exit',
    'actual_engineering/preflight.log','actual_engineering/controller.log',
    'actual_engineering/witness/NATIVE_M0_RECEIPT.json',
    'actual_engineering/witness/NATIVE_M0_TRAINING_DIAGNOSTIC.json',
    'actual_review/EXPERIMENT_AUDIT.md','actual_review/EXPERIMENT_AUDIT.json',
    'NORMAL_LAUNCH.json','NORMAL_LAUNCH_STDOUT.json','NORMAL_LAUNCH_EXIT.json',
    'NORMAL_START_WITNESS.json','NORMAL_START_WITNESS_EXIT.json',
    'NORMAL_START_LOG_HEAD.txt','NORMAL_START_LOG_TAIL.txt','NORMAL_OBSERVER_OWNER.json',
    'prepare_normal_observers.py','witness_normal_start_authorized.py',
    'observe_normal_planned_authorized.py','record_normal_launch.py')]
names += ['normal_degradation_assessment_20261010/'+name for name in (
    'OLD_INITIAL_ARRAY_DUPLICATES.json','OLD_INITIAL_ARRAY_DEDUPLICATION.json',
    'INITIAL_DEDUPLICATION_STDOUT.json','INITIAL_DEDUPLICATION_EXIT.json',
    'deduplicate_exact_old_initial_arrays_authorized.py')]
names += ['prepare_c_off_normal_publication.py']
assert all((root/name).is_file() for name in names)
assert not any('STDERR' in name or 'private/' in name or 'RAW_' in name for name in names)
(root/'C_OFF_NORMAL_PUBLIC_FILE_LIST.json').write_text(json.dumps(sorted(set(names)),indent=2)+'\n')
content=(root/'publish_c_off_runtime_authorized.py').read_text(encoding='utf-8')
replacements = {
    "'normal_terminal_publication.json'":"'c_off_runtime_publication.json'",
    "prior['section'] == '20.376.138'":"prior['section'] == '20.376.139'",
    "b'20.376.139' not in old":"b'20.376.140' not in old",
    "section='20.376.139'":"section='20.376.140'",
    'c_off_runtime_publication.json\').exists()':'c_off_normal_publication.json\').exists()',
    'C_OFF_RUNTIME_PUBLIC':'C_OFF_NORMAL_PUBLIC',
    'normal_controls_20261010/C_OFF_RUNTIME_SUMMARY.json':'normal_controls_20261010/C_OFF_NORMAL_SUMMARY.json',
    'normal_controls_20261010/HANDOFF_C_OFF_RUNTIME_20261010.md':'normal_controls_20261010/HANDOFF_C_OFF_NORMAL_20261010.md',
    'c_off_runtime_20261010/':'c_off_normal_20261010/',
    '/c_off_runtime_20261010\'':'/c_off_normal_20261010\'',
    'publish_c_off_runtime_authorized.py':'publish_c_off_normal_authorized.py',
    'c_off_runtime_local_commit.json':'c_off_normal_local_commit.json',
    'C_OFF_RUNTIME_ALL_':'C_OFF_NORMAL_ALL_',
    'Record bounded normal degradation diagnosis and C-off engineering launch':'Record completed C-off preflight and ordinary joint training launch',
    'control_training_launched=False':'control_training_launched=True',
    "(root / 'c_off_runtime_publication.json').write_text":"(root / 'c_off_normal_publication.json').write_text",
}
for old,new in replacements.items():
    assert old in content,old
    content=content.replace(old,new)
compile(content,'publish_c_off_normal_authorized.py','exec')
(root/'publish_c_off_normal_authorized.py').write_text(content,encoding='utf-8')
print(json.dumps(dict(status=summary['status'],public_files=len(names),first_due_cst=observer['first_due_cst'])))
