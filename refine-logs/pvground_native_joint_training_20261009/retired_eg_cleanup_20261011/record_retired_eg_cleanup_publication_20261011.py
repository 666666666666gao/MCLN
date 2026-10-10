"""Close the archived-cache task and preserve the one live training observer."""
import datetime
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
data=root/'normal_controls_20261010'
publication=json.loads((root/'retired_eg_cleanup_publication.json').read_bytes())
assert publication['status']=='RETIRED_EG_CLEANUP_ALL_COPIES_AND_GITHUB_SYNCHRONIZED'
assert publication['section']=='20.376.146' and publication['github_main_verified']
retire=json.loads((data/'RETIRED_EG_ARRAY_RETIREMENT.json').read_bytes())
assert retire['logical_bytes_removed']==411318656 and retire['model_weights_deleted']==0
assert retire['original_archive_native_session']==84511 and retire['original_archive_native_exit_code']==0
owner=json.loads((data/'NORMAL_EPOCH2_AUTHORIZED_OBSERVER_OWNER.json').read_bytes())
assert owner['local_pid']==51540 and owner['first_due_cst']=='2026-10-11T05:00:40.968644+08:00'
doc='docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
for name in ['.codex_mcln_g0_20260905','.codex_pvground_cs_20261002','.codex_mcln_v99_internal_20260928']:
 assert hashlib.sha256((Path('C:/Users/gb')/name/doc).read_bytes()).hexdigest()==publication['doc_sha256']
assert hashlib.sha256((Path('C:/Users/gb/Desktop/document')/Path(doc).name).read_bytes()).hexdigest()==publication['doc_sha256']
now=datetime.datetime.now().astimezone().isoformat()
completion=dict(status=retire['status'],archive_complete=True,remote_deletion_completed=True,
 original_archive_native_session=84511,original_archive_native_exit_code=0,
 logical_bytes_removed=411318656,actual_data_free_increase_bytes=retire['data_free_increase_bytes'],
 data_free_after_bytes=retire['storage_after']['/root/autodl-tmp']['free'],
 system_free_after_bytes=retire['storage_after']['/']['free'],model_weights_deleted=0,
 local_archive='C:/Users/gb/.codex/archives/pvg_retired_candidate_arrays_20261011',receipt=str(data/'RETIRED_EG_ARRAY_RETIREMENT.json'),
 obsolete_archive_check_cst='2026-10-11T04:08:46.7119381+08:00',obsolete_archive_check_cancelled=True,
 neural_calls=0,training_status_reads=0,training_restart=False)
classification='ACTUAL_PROGRESS_THREE_OLD_EG_ARRAYS_FULLY_ARCHIVED_EXACTLY_RETIRED_SECTION146_SYNCED'
for path in [Path('C:/Users/gb/.codex/tmp/pvground_referit_mask_reference_20261006/current_research_goals.json'),Path('C:/Users/gb/.codex/tmp/pvground_query_supported_geometry_20261005/active_continuation_state.json')]:
 value=json.loads(path.read_bytes());assert value['latest_handoff_section']=='20.376.145'
 value.pop('retired_EG_arrays_archive_pending',None)
 value.update(retired_EG_arrays_archival=completion,latest_retired_EG_cleanup_publication=publication,
  latest_handoff_section=publication['section'],handoff_sha256=publication['doc_sha256'],published_heads=publication['heads'],
  remote_handoff_sync_pending=False,remote_handoff_last_confirmed_section=publication['section'],remote_handoff_last_confirmed_sha256=publication['doc_sha256'],
  updated_cst=now,current_turn_classification=classification,full_goal_complete=False)
 value['normal_training_requirement'].update(next_normal_training_observation_not_before=owner['first_due_cst'],current_normal_observer_native_session=52851,current_normal_observer_pid=51540)
 path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
path=root/'NORMAL_CONTINUATION_STATE.json';value=json.loads(path.read_bytes())
value.pop('retired_EG_arrays_archive_pending',None)
value.update(retired_EG_arrays_archival=completion,retired_EG_cleanup_publication=publication,current_turn_classification=classification,
 next_action='Sole authorized normal observer52851/PID51540 dueOct11 05:00:40; archive84511 completed0 and exact retirement closed, do not repeat old archive check.',full_goal_complete=False)
path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
note='\nPV-Ground '+now+': section20.376.146 synchronized MAIN '+publication['heads'][0]+', docSHA '+publication['doc_sha256']+'. Original gzip archiver84511 consumedexit0, completed02:11:33 in2801.933s; threeoldEG NR-adapt/NR-transfer/Scan-acceptance arrays full411318656B saved asraw+gz322504481B in localarchive pvg_retired_candidate_arrays_20261011, no remote large temp. Full final localSHA re-read02:13:49 andallremoteSHA repeated; exactlythreeoldcandidates.npy deleted02:13:57, nlink1 each, datafree1457647616->1868984320B(+411336704 physical), systemfree320847872B unchangedbythisdeletion;0weights/source/data/runtime modifications/0NNqueries. No large matching installedwheel pipcache found, no packagecachedeletion. Obsolete04:08 archivecheck closed/donotrestart84511. Soleauthorized normalobserver52851/PID51540 remainsdueOct11 05:00:40; source/recipe/score/best5677/4920 unchanged, latestactualE1 C-off5644/4604 stillinterim. Threeeffective andcompleteNrSr unproved, goalACTIVE_UNMET.\n'
for path in [Path('C:/Users/gb/memory/2026-10-11.md'),Path('C:/Users/gb/MEMORY.md')]:
 with path.open('a',encoding='utf-8') as stream:stream.write(note)
print(json.dumps(dict(status='RETIRED_EG_CLEANUP_SECTION146_CANONICAL_AND_MEMORY_RECORDED',section=publication['section'],data_free_after=completion['data_free_after_bytes'],normal_next_due_cst=owner['first_due_cst'],full_goal_complete=False)))
