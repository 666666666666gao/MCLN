"""Record the closed real probe and actual new pair launch, without extra SSH."""
import datetime
import json
from pathlib import Path

local=Path(__file__).parent
analysis=json.loads((local/'preflight_analysis.json').read_bytes())
launch=json.loads((local/'launch.json').read_bytes())
wait=json.loads((local/'wait.json').read_bytes())
path=local/'active_continuation_state.json'
state=json.loads(path.read_bytes())
assert state['status']=='REAL_FROZEN_PROTOCOL_PREFLIGHT_LAUNCHED_RESULT_PENDING'
assert analysis['engineering_status']=='PASS' and launch['head_only']
state.update(time_cst=datetime.datetime.now().astimezone().isoformat(),
    status='REAL_PREFLIGHT_CLOSED_PASS_BOUNDED_HEAD_ONLY_SOURCE_PAIR_LAUNCHED',
    preflight_observer_session_status='90776_CLOSED_EXIT0',preflight_native_wait_cells_status='413_AND_416_CLOSED',
    actual_preflight_analysis=analysis,active_controller=launch['process'],fit_launch_cst=launch['time_cst'],
    formal_fit_launched=True,formal_accuracy_result_available=False,
    sole_local_observer_session=82422,sole_native_wait_cell=None,
    first_remote_check_cst=wait['next_scheduled_cst'],minimum_poll_interval_seconds=240,
    monitoring='unchanged adaptive read-only observer; estimate from actual throughput, check near phase end',
    original_pair_observer_and_publishers_closed=True,
    next_steps=['publish actual probe and fit launch','wait until scheduled06:49:41 observation',
        'use actual phase throughput for later near-end checks','collect both actual formal endpoints',
        'retain originalG unless primary strict metric genuinely improves'])
path.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
with Path('C:/Users/gb/memory/2026-10-04.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+state['time_cst']+' actual frozen-G preflight CLOSEDPASS06:03:22/454.57s,4fullforwards/2updates/all10positiveafter2/coreunchanged/exactAdamW, serialization4815819B/allocator3866370048/reserved5540675584/0diskweights. Sole90776 and413/416 closed0; collect25actualfiles once and analyze once. Actual fullhead-only pair450028 started06:07:37.030162 B8/29778once/3723updates. Soleunchangedread-only adaptiveobserver82422 first06:49:41.508889 thennearactualestimatedphaseend/min240; do not prematurelyquery/relaunch. Currentno9508newaccuracy/G4495protected/GoalACTIVE_UNMET. Publicationpending.\n')
print(json.dumps({key:state[key] for key in ('status','active_controller','sole_local_observer_session','first_remote_check_cst','formal_accuracy_result_available')}))
