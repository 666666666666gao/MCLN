import ast
import datetime
import json
from pathlib import Path

root = Path(__file__).resolve().parent
owner = json.loads((root / 'NORMAL_EPOCH3_BOUNDARY_OBSERVER_OWNER.json').read_bytes())
witness = json.loads((root / 'NORMAL_EPOCH3_LOCAL_OWNER_VERIFIED_20261010.json').read_bytes())
plan = json.loads((root / 'NORMAL_EPOCH3_BOUNDARY_PLAN.json').read_bytes())
assert owner['local_pid'] == witness['local_pid'] == 45492 and witness['native_session'] == 73816
assert owner['first_due_cst'] == plan['due_cst'] == witness['due_cst']
summary_path = root / 'NORMAL_EPOCH2_FULL_OBSERVATION_SUMMARY.json'
summary = json.loads(summary_path.read_bytes())
summary['formal_results_unaudited'] = True
summary_path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
for path in [root.parent.parent / 'pvground_referit_mask_reference_20261006/current_research_goals.json',
             root.parent.parent / 'pvground_query_supported_geometry_20261005/active_continuation_state.json',
             root / 'NORMAL_CONTINUATION_STATE.json']:
    value = json.loads(path.read_bytes())
    value.update(current_normal_observer=dict(owner, native_session=73816),
                 latest_verified_wait=witness, next_observation_cst=plan['due_cst'],
                 current_turn_classification='ACTUAL_PROGRESS_NORMAL_E2_FULL9508_AND_E3_SINGLE_OBSERVER_ADMITTED',
                 full_goal_complete=False)
    if 'normal_training_requirement' in value:
        value['normal_training_requirement'].update(
            current_normal_epoch2_formal_result=summary,
            current_normal_observer_session_id=73816,
            current_normal_observer_live_verified_PID=45492,
            current_normal_observer_live_verified_cst=witness['checked_cst'],
            current_normal_observer_owner=owner,
            current_normal_observer_session_closed=False,
            current_normal_next_observation_plan=plan,
            next_normal_training_observation_not_before=plan['due_cst'])
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

source = (root / 'publish_normal_epoch1_authorized.py').read_text(encoding='utf-8')
source = source.replace('normal_epoch1', 'normal_epoch2').replace('NORMAL_EPOCH1', 'NORMAL_EPOCH2')
source = source.replace('20.376.135', '20.376.137').replace('20.376.134', '20.376.136')
source = source.replace('face_residual_cpu_publication.json', 'referit_metadata_publication.json')
source = source.replace('NORMAL_FIRST_OBSERVATION_SUMMARY.json', 'NORMAL_EPOCH2_FULL_OBSERVATION_SUMMARY.json')
source = source.replace('NORMAL_FIRST_OBSERVATION', 'NORMAL_EPOCH2_FINAL_OBSERVATION')
source = source.replace('NORMAL_NEXT_OBSERVATION_PLAN.json', 'NORMAL_EPOCH3_BOUNDARY_PLAN.json')
source = source.replace('NORMAL_EPOCH2_OBSERVER_OWNER.json', 'NORMAL_EPOCH3_BOUNDARY_OBSERVER_OWNER.json')
source = source.replace("owner['local_pid']==51724", "owner['local_pid']==45492")
source = source.replace("summary['epoch1_hits_delta']==[-25,-324]", "summary['epoch2_delta_from_initial']==[-125,-368]")
source = source.replace('HANDOFF_NORMAL_EPOCH1', 'HANDOFF_NORMAL_EPOCH2')
source = source.replace('normal_first_observation', 'normal_epoch2_final_observation')
source = source.replace('Record first normal joint epoch and measured next observation time',
                        'Record second normal joint epoch and final scheduled observation')
start = source.index("names=['HANDOFF_NORMAL_EPOCH2")
end = source.index('files={prefix+name:', start)
names = ['HANDOFF_NORMAL_EPOCH2_20261010.md', 'NORMAL_EPOCH2_FULL_OBSERVATION_SUMMARY.json',
    'NORMAL_EPOCH2_FINAL_OBSERVATION.json', 'NORMAL_EPOCH2_FINAL_OBSERVATION_STDOUT.json',
    'NORMAL_EPOCH2_FINAL_OBSERVATION_EXIT.json', 'NORMAL_EPOCH2_OBSERVATION_SUMMARY.json',
    'NORMAL_EPOCH2_BOUNDARY_TRANSPORT_FAILURE.json', 'NORMAL_EPOCH2_RECOVERY_SUMMARY.json',
    'NORMAL_EPOCH3_BOUNDARY_PLAN.json', 'NORMAL_EPOCH3_BOUNDARY_OBSERVER_OWNER.json',
    'NORMAL_EPOCH3_LOCAL_OWNER_VERIFIED_20261010.json', 'NORMAL_EPOCH3_MAINTENANCE_PUBLIC_SUMMARY.json',
    'NORMAL_VALIDATION_POINT_INPUT_SOURCE_CHECK_20261010.md',
    'NORMAL_VALIDATION_POINT_INPUT_SOURCE_CHECK_20261010.json',
    'observe_normal_epoch2_final_authorized.py', 'observe_normal_epoch3_boundary_authorized.py',
    'prepare_normal_epoch3_boundary.py', 'prepare_normal_epoch2_publication.py',
    'publish_normal_epoch2_authorized.py']
source = source[:start] + 'names=' + repr(names) + "\nnames.extend(path.relative_to(root).as_posix() for path in (root/'normal_epoch2_final_observation').rglob('*') if path.is_file())\n" + source[end:]
ast.parse(source)
assert '20.376.136' in source and '20.376.137' in source
assert 'RAW_STDERR' not in source
path = root / 'publish_normal_epoch2_authorized.py'
assert not path.exists()
path.write_text(source, encoding='utf-8')
print(json.dumps(dict(status='NORMAL_E2_STATIC_PUBLICATION_PREPARED', section='20.376.137',
    prior_section='20.376.136', observer_pid=45492, native_session=73816,
    current_training_queries=0, active_source_changed=False)))
