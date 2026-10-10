"""Record source preparation and one verified cache retirement; do not inspect training."""
import datetime
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
data=root/'referit_data_interface_cpu_20261010'
spec=json.loads((data/'CHECK_SPEC.json').read_bytes())
cleanup=json.loads((root/'normal_degradation_assessment_20261010/SR_ARCHIVED_ARRAY_RETIREMENT.json').read_bytes())
archive=json.loads((root/'normal_degradation_assessment_20261010/RETIRED_SR_ARRAY_ARCHIVE_COMPLETE.json').read_bytes())
assert cleanup['deletions']==1 and cleanup['weights_deleted']==0 and cleanup['local_archive_reverified']
assert archive['identity']['sha256']==cleanup['sha256'] and archive['identity']['bytes']==cleanup['bytes']
owner=json.loads((root/'normal_controls_20261010/NORMAL_OBSERVER_OWNER.json').read_bytes())
launch=json.loads((root/'normal_controls_20261010/NORMAL_LAUNCH.json').read_bytes())
assert owner['remote_controller_pid']==launch['controller_pid']==185395
assert owner['remote_child_pid']==185396
now=datetime.datetime.now().astimezone().isoformat()
prepared=dict(status='LIMITED_REAL_REFERIT_DATA_CPU_SOURCE_PREPARED_NOT_EXECUTED',
    root=str(data),source_review_report=str(data/'source_review/EXPERIMENT_CODE_REVIEW.json'),
    spec_sha256=hashlib.sha256((data/'CHECK_SPEC.json').read_bytes()).hexdigest(),
    author_flags=spec['author_flags'],annotation_cap_per_source=128,planned_distinct_annotations=6,
    expected_item_calls=10,real_data_execution_pending=True,formal_accuracy=None,
    GPU_training_admission=False,normal_training_launched=False)
job=dict(root=launch['root'],controller_pid=185395,child_pid=185396,
    source=launch['model_source'],observer_native_session=21631,observer_local_pid=48772,
    first_due_cst=owner['first_due_cst'],later_poll_seconds=240,
    last_confirmed_remote_status='ORDINARY_NATIVE_ENTRY_ALIVE_LOADING_DATA_AT20:04:26',
    new_formal_accuracy=None,current_turn_training_queries=0)
for path in [Path('C:/Users/gb/.codex/tmp/pvground_referit_mask_reference_20261006/current_research_goals.json'),
             Path('C:/Users/gb/.codex/tmp/pvground_query_supported_geometry_20261005/active_continuation_state.json'),
             root/'NORMAL_CONTINUATION_STATE.json']:
    value=json.loads(path.read_bytes())
    value.update(referit_data_interface_cpu_preparation=prepared,
        retired_sr_candidate_array_archive=archive,retired_sr_candidate_array_cleanup=cleanup,
        original_cache_native_session51919_consumed_exit0=True,current_normal_job=job,
        updated_cst=now,current_turn_classification='ACTUAL_PROGRESS_REFERIT_REAL_DATA_PREPARATION_AND_VERIFIED_CACHE_RETIREMENT',
        full_goal_complete=False)
    if path.name=='NORMAL_CONTINUATION_STATE.json':
        value['next_action']='Wait original C-off observer21631/PID48772 untilOct11 01:03:05. Original cache51919 finished21:03:05 and exact array retired21:05:28. Limited realNrSr CPU data check awaits source review; no new GPU job.'
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
note='\nPV-Ground '+now+': realNrSr data interface source prepared at referit_data_interface_cpu_20261010, source review running; not executed yet. Author butd_cls/joint_det/detect_intermediate flags explicit: GT scene proposals plus cls_results predictedclasses, not pure detector or GTclasses. Direct native dataset limited128annotations/source retains realtrain augmentation and jointscannet identity; planned6distinctitems/10calls/2collations, no PV network/criterion/GPU/accuracy; native CPUspaCy processing acknowledged, fullscenes stillloaded. Resource intake20:57 observed593935523840B RAMavailable, train/val pickles10095098474B. Original archive51919/PID47024 consumedexit0 actualcomplete21:03:05.349796,7603.887sec,308574336B/SHA8621cc084725688fecbe9ea8e2a9655728e7b14b922cb2c10774158d89d75937 localfullarchive. ExactoldEGSr candidatearray alone deleted21:05:28.034661 after local+remoteSHA recheck, datafree2326523904B;0weightsdeleted/0trainingqueries. No restart, no current training poll. Keep original C-off observer21631/PID48772 firstOct11 01:03:05, normal source/config unchanged. Publishedsection141/MAIN25d2022 remains current; cleanup/data-prep publication pending until actual evidence. Best5677/4920; three effective modules/NrSrtraining stillpending, goalACTIVE_UNMET.\n'
for path in [Path('C:/Users/gb/memory/2026-10-10.md'),Path('C:/Users/gb/MEMORY.md')]:
    with path.open('a',encoding='utf-8') as stream:stream.write(note)
print(json.dumps(dict(status='REFERIT_DATA_PREPARATION_AND_ONE_CACHE_RETIREMENT_RECORDED',
    retired_bytes=cleanup['bytes'],normal_observer_due=owner['first_due_cst'],full_goal_complete=False)))
