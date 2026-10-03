"""Save the observed launch and single observer identity, without remote polling."""
import datetime
import json
from pathlib import Path

local=Path(__file__).parent
assert not (local/'active_continuation_state.json').exists()
launch=json.loads((local/'preflight_launch.json').read_bytes())
wait=json.loads((local/'preflight_wait.json').read_bytes())
publication=json.loads((local.parent/'pvground_whole_mask_fit_20261003/terminal_publication.json').read_bytes())
state=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
    status='REAL_FROZEN_PROTOCOL_PREFLIGHT_LAUNCHED_RESULT_PENDING',
    source_review='ACTUAL_PASS_55_FILES_0_BLOCKING_SAME_FAMILY_PROVISIONAL',
    active_controller=launch['process'],preflight_launch_cst=launch['time_cst'],
    sole_local_observer_session=90776,sole_native_wait_cell='413',
    first_remote_check_cst=wait['first_check_cst'],poll_interval_seconds=240,
    estimated_preflight_finish_cst=wait['estimated_finish_cst'],
    formal_fit_launched=False,formal_accuracy_result_available=False,
    latest_actual_publication=publication['github_main'],handoff_sha256=publication['handoff_sha256'],
    retained_metric_best='original_g_5615_4495',goal_status='ACTIVE_UNMET',
    old_pair_closed_and_published=True,old_pair_local_whole_bbs=[[5603,4428],[5594,4461]],
    old_pair_owned_deleted_bytes=694233890,new_preflight_disk_weights_expected=0,
    next_steps=['consume sole observer completion','collect actual primary receipts once',
        'verify frozen two-step result','then launch prepared bounded pair if actual sanity passes'])
(local/'active_continuation_state.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
previous_path=local.parent/'pvground_whole_mask_fit_20261003/active_formal_continuation_state.json'
previous=json.loads(previous_path.read_bytes())
previous.update(goal_turn_result='PAIR_CLOSED_PUBLISHED_NEXT_STABLE_G_PROBE_LAUNCHED',
    terminal_publication=publication,old_pair_has_active_GPU_job=False,next_control_root=str(local),
    next_control_actual_preflight_controller=launch['process'],next_control_has_accuracy_result=False)
previous_path.write_text(json.dumps(previous,indent=2)+'\n',encoding='utf-8')
with Path('C:/Users/gb/memory/2026-10-04.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+state['time_cst']+' actual head-only preflight449570 started05:55:46.716447; soleobserver90776/413 first06:01:46.716447 then240s, estimate06:03:46.716447. Realresult pending/noformalfit/no newprecision/no diskweights; priorF78aac55 §39 closed/published/694233890B retired/Gprotected. Nextconsumeonce collectonce thenboundedpaironlyafterPASS. GoalACTIVE_UNMET.\n')
print(json.dumps({key:state[key] for key in ('status','active_controller','first_remote_check_cst','formal_fit_launched')}))
