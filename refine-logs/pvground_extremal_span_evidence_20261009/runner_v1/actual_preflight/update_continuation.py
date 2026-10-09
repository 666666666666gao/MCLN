"""Record actual local receipts; no remote query or computational changes."""
import datetime
import json
from pathlib import Path

root = Path(__file__).resolve().parent
tmp = root.parents[1]
prior = tmp / 'pvground_selected_mask_training_20261009'
decision = json.loads((prior / 'postrun_results/RETAINED_PARENT_DECISION.json').read_bytes())
intake = json.loads((root / 'preflight_complete/INTAKE.json').read_bytes())
closed = json.loads((root / 'preflight_wait.json').read_bytes())
assert intake['status'] == 'CLOSED_SPAN_PHASE_ARTIFACTS_COLLECTED'
assert closed['observer_closed'] and closed['terminal']['exitcode'] == 0
now = datetime.datetime.now().astimezone().isoformat()
goal_path = tmp / 'pvground_referit_mask_reference_20261006/current_research_goals.json'
goal = json.loads(goal_path.read_bytes())
hits = decision['parent_hits']
assert hits == [5606, 4881]
scan = goal['scanrefer']
scan.update(current_best_hits025=hits[0], current_best_hits050=hits[1],
            current_best_checkpoint=decision['remote_parent_path'],
            current_best_checkpoint_sha256=decision['parent_sha256'],
            current_best_local_archive=decision['local_parent_path'],
            remaining_hits025=max(0, scan['minimum_hits025'] - hits[0]),
            remaining_hits050=max(0, scan['minimum_hits050'] - hits[1]),
            current_best_joint_target_passed=False)
goal.update(updated_cst=now, last_completed_pair_audit=str(prior / 'postrun_results/EXPERIMENT_AUDIT.json'))
active = dict(name='extremal_span_pair', status='ACTUAL_PREFLIGHT_CLOSED_COLLECTED_AUDIT_PENDING',
              launch=str(root / 'preflight_launch.json'), intake=str(root / 'preflight_complete/INTAKE.json'),
              seed=2027, no_multiseed=True, parent_hits=hits, formal_training_started=False)
if (root / 'fit_launch.json').exists():
    fit = json.loads((root / 'fit_launch.json').read_bytes())
    active.update(status='ACTUAL_FORMAL_FIT_STARTED_NOT_COMPLETED', launch=str(root / 'fit_launch.json'),
                  formal_training_started=True, first_complete_check=fit['first_observation_cst'],
                  later_poll_seconds=fit['later_poll_seconds'], controller_pid=fit['controller_pid'])
goal['active_experiment'] = active
goal_path.write_text(json.dumps(goal, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
state_path = tmp / 'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state = json.loads(state_path.read_bytes())
state['latest_verified_wait'] = dict(time_cst=now, preflight_observer_closed=True,
    preflight_wait=str(root / 'preflight_wait.json'), collector_closed_exitcode=0,
    intake=str(root / 'preflight_complete/INTAKE.json'), actual_preflight_audit_pending=True,
    formal_training_started=active['formal_training_started'], remote_queries_this_update=0)
state['extremal_span_preparation']['actual_preflight_closure'] = dict(closed, intake=str(root / 'preflight_complete/INTAKE.json'))
if active['formal_training_started']:
    state['extremal_span_preparation']['actual_fit_launch'] = fit
    state['latest_verified_wait'].update(actual_preflight_audit_pending=False,
        observer=str(root / 'fit_observer_started.json'), first_remote_observation_cst=fit['first_observation_cst'])
state.update(active_experiment=active, full_goal_status='ACTIVE_UNMET')
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status='LOCAL_CONTINUATION_UPDATED', time_cst=now, parent_hits=hits, active_experiment=active)))
