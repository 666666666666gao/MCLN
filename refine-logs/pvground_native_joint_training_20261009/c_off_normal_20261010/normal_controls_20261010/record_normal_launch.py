"""Record the admitted C-off normal run and its original observer ownership."""
import datetime
import json
from pathlib import Path

root = Path(__file__).resolve().parent
previous = root.parent
launch = json.loads((root / 'NORMAL_LAUNCH.json').read_bytes())
start = json.loads((root / 'NORMAL_START_WITNESS.json').read_bytes())
observer = json.loads((root / 'NORMAL_OBSERVER_OWNER.json').read_bytes())
assert start['status'] == 'ORDINARY_NATIVE_ENTRY_ALIVE'
assert start['controller_pid'] == launch['controller_pid'] == observer['remote_controller_pid']
assert start['child_pid'] == observer['remote_child_pid']
now = datetime.datetime.now().astimezone().isoformat()
record = dict(status='C_OFF_ORDINARY_NATIVE_ENTRY_LIVE_LOADING_DATA', time_cst=now,
    launch=launch, start_witness=start, observer=observer, observer_native_session=21631,
    engineering_completed=True, actual_preflight_review_verdict='WARN',
    actual_preflight_blockers=0, normal_training_started=True, optimizer_steps_not_yet_observed=True,
    first_observation_cst=observer['first_due_cst'], later_poll_seconds=240,
    new_formal_accuracy=None, best_retained_counts=[5677,4920],
    no_multiseed=True, three_effective_contributions_proven=False, full_goal_complete=False,
    publication_section140_pending=True)
(root / 'C_OFF_CURRENT_STATUS.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
state_path = previous / 'NORMAL_CONTINUATION_STATE.json'
state = json.loads(state_path.read_bytes())
state.update(status=record['status'], time_cst=now, c_off_normal=record,
    c_off_normal_training_live=True, normal_training_live=False,
    c_off_engineering_complete=True, c_off_engineering_actual_audit_complete=True,
    current_turn_classification='ACTUAL_PROGRESS_C_OFF_NORMAL_RUN_LAUNCHED',
    next_action='Publish section140 actual M0/dedup/main launch; consume original single observer21631 at01:03; no earlier accuracy polling.')
state_path.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
for path in [Path('C:/Users/gb/.codex/tmp/pvground_referit_mask_reference_20261006/current_research_goals.json'),
             Path('C:/Users/gb/.codex/tmp/pvground_query_supported_geometry_20261005/active_continuation_state.json')]:
    value = json.loads(path.read_bytes())
    value.update(updated_cst=now,pvground_c_off_current_status=record,
        current_turn_classification='ACTUAL_PROGRESS_C_OFF_NORMAL_RUN_LAUNCHED',full_goal_complete=False)
    value['normal_training_requirement'].update(
        current_normal_c_off_training_live=True,current_normal_control_actual_admission_complete=True,
        current_normal_c_off_controller_pid=start['controller_pid'],
        current_normal_c_off_child_pid=start['child_pid'],
        current_normal_c_off_observer_native_session=21631,
        current_normal_c_off_observer_pid=observer['local_pid'],
        current_normal_c_off_first_due_cst=observer['first_due_cst'],
        current_normal_c_off_new_formal_accuracy=None,
        current_normal_c_off_engineering_checkpoint_retired=True,
        next_action=state['next_action'])
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
note = '\nPV-Ground '+now+': actual C-off M0 completed19:37:14, exact1295E0state/two native updates/fullrecover; actualreview WARN0block19:55, engineeringonly/noaccuracy. Four exact duplicate old boxes cache copies hardlinked19:46:29, allfivepaths/SHA retained,169255424Bphysical freed. Main C-off normal admitted20:03:10 target, sole controller185395/nativechild185396; same3epochs13749updates/seed2027/B8/freshAdam/sameparents, only selected-mask CFalse. Original unscored841675936B M0weight retired insideadmission afteractualaudit; free2633846784B exceeds2605151860 reserve. One startup witness20:04:26 childalive, data textdecoupling/loading, no GPU process yet, no optimizersteps/new precision claimed. Original planned observer21631/worker48772 created20:04:32.647034, first01:03:05.277556 Oct11, onequerythenreestimate/later240s. DoNOTlaunchduplicateobserver/training orpollbeforetime. Retainedbest5677/4920=59.7076/51.7459; normal priorE3=5576/4488, localfullstatearchived/remoteinferiorretired. Section139 stilllatestpublic; section140 actualM0/dedup/mainlaunch pendingpublication. EGcachearchive51919/47024 stillowned/unclosed, doNOTretryobsolete2targetcleanup. GoalACTIVE_UNMET: >59.5/>51 plus3effective mechanisms then sameNr/Sr; no multiseed.\n'
for path in [Path('C:/Users/gb/memory/2026-10-10.md'),Path('C:/Users/gb/MEMORY.md')]:
    with path.open('a',encoding='utf-8') as stream:
        stream.write(note)
print(json.dumps(dict(status=record['status'],controller_pid=start['controller_pid'],
    child_pid=start['child_pid'],next_observation=observer['first_due_cst'],full_goal_complete=False)))
