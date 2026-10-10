"""Record the observed C-off E1 and the authorized sole E2 observer."""
import datetime
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
data=root/'normal_controls_20261010'
publication=json.loads((root/'c_off_epoch1_publication.json').read_bytes())
assert publication['status']=='C_OFF_EPOCH1_ALL_COPIES_AND_GITHUB_SYNCHRONIZED'
assert publication['section']=='20.376.145' and publication['github_main_verified']
summary=json.loads((data/'NORMAL_FIRST_OBSERVATION_SUMMARY.json').read_bytes())
plan=json.loads((data/'NORMAL_NEXT_OBSERVATION_PLAN.json').read_bytes())
owner=json.loads((data/'NORMAL_EPOCH2_AUTHORIZED_OBSERVER_OWNER.json').read_bytes())
assert owner['local_pid']==51540 and owner['first_due_cst']==plan['due_cst']
doc='docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
for name in ['.codex_mcln_g0_20260905','.codex_pvground_cs_20261002','.codex_mcln_v99_internal_20260928']:
    assert hashlib.sha256((Path('C:/Users/gb')/name/doc).read_bytes()).hexdigest()==publication['doc_sha256']
assert hashlib.sha256((Path('C:/Users/gb/Desktop/document')/Path(doc).name).read_bytes()).hexdigest()==publication['doc_sha256']
now=datetime.datetime.now().astimezone().isoformat()
current=dict(status=summary['status'],snapshot_cst=summary['observed_cst'],remote_root='/root/autodl-tmp/pvground_native_joint_controls_20261010/selected_mask_off',
    controller_pid=185395,child_pid=185396,observed_metrics=summary['metrics'],latest_completed_epoch=1,
    current_epoch_at_snapshot=2,current_update_at_snapshot=269,terminal_result=None,formal_results_unaudited=True,
    best_retained_counts=[5677,4920],three_effective_contributions_proven=False,source_changed=False,
    original_first_observer_native_session=21631,original_first_observer_consumed_exit_code=0,
    next_observer_pid=51540,next_observer_native_session=52851,next_observer_owner=owner,
    next_observation_cst=plan['due_cst'],next_observation_plan=plan,
    canceled_future_observer_native_session=48982,canceled_future_observer_pid=38028,
    cancellation_reason='Missing authorized environment on direct route; cancel future timer only and use original authorized route',
    training_restart=False,full_goal_complete=False)
classification='ACTUAL_PROGRESS_C_OFF_NORMAL_E1_OBSERVED_SECTION145_SYNCED_SINGLE_E2_OBSERVER_ACTIVE'
for path in [Path('C:/Users/gb/.codex/tmp/pvground_referit_mask_reference_20261006/current_research_goals.json'),Path('C:/Users/gb/.codex/tmp/pvground_query_supported_geometry_20261005/active_continuation_state.json')]:
    value=json.loads(path.read_bytes())
    assert value['latest_handoff_section']=='20.376.144'
    value.update(latest_handoff_section=publication['section'],handoff_sha256=publication['doc_sha256'],published_heads=publication['heads'],
        remote_handoff_sync_pending=False,remote_handoff_last_confirmed_section=publication['section'],remote_handoff_last_confirmed_sha256=publication['doc_sha256'],updated_cst=now,
        current_turn_classification=classification,current_normal_job=current,current_c_off_epoch1_publication=publication,full_goal_complete=False)
    value['normal_training_requirement'].update(current_normal_job=current,current_normal_interim_observation=summary,
        current_normal_precision_result=summary['metrics'][1],normal_formal_results_unaudited=True,
        next_normal_training_observation_not_before=plan['due_cst'],current_normal_observer_native_session=52851,current_normal_observer_pid=51540,
        current_normal_next_observation_plan=plan,current_normal_terminal_result=None,three_effective_contributions_confirmed=False)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
path=root/'NORMAL_CONTINUATION_STATE.json'
state=json.loads(path.read_bytes())
state.update(current_normal_job=current,current_c_off_epoch1_observation=summary,current_c_off_epoch1_publication=publication,
    current_normal_observer=owner,current_normal_observer_native_session=52851,current_normal_observer_pid=51540,
    current_normal_next_observation_plan=plan,current_turn_classification=classification,
    next_action='Consume sole authorized observer52851/PID51540 atOct11 05:00:40; do not query earlier or modify current training.',full_goal_complete=False)
path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
path=data/'C_OFF_CURRENT_STATUS.json'
old=json.loads(path.read_bytes())
old.update(status=summary['status'],time_cst=now,current_job=current,original_first_observer_closed=True,
    current_observer_native_session=52851,current_observer=owner,next_observation_cst=plan['due_cst'],
    new_formal_accuracy=summary['metrics'][1],formal_results_unaudited=True,optimizer_steps_not_yet_observed=False,
    publication_section=publication['section'],publication_doc_sha256=publication['doc_sha256'],full_goal_complete=False)
path.write_text(json.dumps(old,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
note='\nPV-Ground '+now+': section20.376.145 synchronized MAIN '+publication['heads'][0]+', docSHA '+publication['doc_sha256']+'. Original21631 consumed0: due01:03:12 snapshot E0 5677/4920, C-off E1 5644/4604=59.3605/48.4224, ownE0delta-33/-316, historicalC-onE1delta-8/+8; interim unaudited/notterminal/notC never-trained parent. E2 269/4583 in817s; E1val1446s; nextdue05:00:40.968644, estimatedvalend05:05:40.968644. Resource directauth failed255 preserved; authorizedattempt2 successful01:09:27: original185395/185396 alive, A10028655/40960MiB100%, systemfree312578048B/datafree1458102272B, best615023752/latest841676832 kept;0NN/metricreads/deletions. Future unauth localtimer38028/native48982 canceled/consumedexit1 ONLY; training unchanged. Sole authorizednewtimer52851/PID51540 started01:09:53, keeporiginal/donotrecreate orpollbeforetime. Same3epoch13749updates/seed2027/B8/nativecore-G-A-B/Coff unchanged. Best5677/4920,3effective andsamefullNrSr unproved; goalACTIVE_UNMET.\n'
for path in [Path('C:/Users/gb/memory/2026-10-11.md'),Path('C:/Users/gb/MEMORY.md')]:
    with path.open('a',encoding='utf-8') as stream:stream.write(note)
print(json.dumps(dict(status='C_OFF_EPOCH1_SECTION145_CANONICAL_AND_MEMORY_RECORDED',section=publication['section'],next_due_cst=plan['due_cst'],observer_native_session=52851,full_goal_complete=False)))
