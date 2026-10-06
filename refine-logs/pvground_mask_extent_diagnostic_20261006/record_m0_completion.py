"""Record actual closed M0 evidence without rewriting historical experiments."""
import datetime
import json
from pathlib import Path

local=Path(__file__).resolve().parent
proof=json.loads((local/'preflight_wait.json').read_bytes())
cpu=json.loads((local/'local_cpu_m0/CPU_SUMMARY.json').read_bytes())
assert proof['observer_closed'] and proof['controller']['completed'] and proof['controller']['exit_code']==0
assert cpu==proof['cpu'] and cpu['preflight_actual_raw_member_rows_replayed']==8
state_path=local.parent/'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state=json.loads(state_path.read_bytes())
state.update(status='MASK_EXTENT_M0_CLOSED_FORMAL_LAUNCH_REVIEW_PENDING',
             time_cst=datetime.datetime.now().astimezone().isoformat(),owned_gpu_job_active=False,
             active_reviewer='/root/pvg_mask_extent_source_review',
             mask_extent_root=str(local),mask_extent_source_review=str(local/'SOURCE_REVIEW.json'),
             mask_extent_preflight_closed=True,mask_extent_preflight_observer_closed_session=86756,
             mask_extent_formal_launched=False,mask_extent_preflight_cpu_raw_rows=8,
             next_action='Complete formal launcher/observer SOURCE followup; run same reviewed9508 read-only diagnostic once. No new training or weights.',
             current_goal_turn_classification='PROGRESS_ACTUAL_RAW_POINT_PREFLIGHT_CLOSED')
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with Path('C:/Users/gb/memory/2026-10-06.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+state['time_cst']+': Mask extent diagnostic M0 actual8rowGPU closed exit0, controller698168/child698169; observer86756 closed0. Independent local raw50000point CPU replay matches remote, no threshold flips, model states unchanged, zero optimizers/updates/weights. Native strict4 vs exact2/q0052 on8 is implementation evidence only. Protected5616/4511 unchanged. Full9508 launcher/observer SOURCE followup pending, not launched.\n')
print(json.dumps(dict(status=state['status'],rows=8,formal_launched=False)))
