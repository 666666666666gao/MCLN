import datetime
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
observation = json.loads((root / 'NORMAL_EPOCH3_MAINTENANCE_OBSERVATION.json').read_bytes())
assert json.loads((root / 'NORMAL_EPOCH3_MAINTENANCE_OBSERVATION_EXIT.json').read_bytes())['exit_code'] == 0
assert observation['controller_alive'] and observation['child_alive']
timing = [line for line in observation['epoch_training_timing_lines'] if 'epoch 2, total time ' in line]
assert len(timing) == 1
training_seconds = float(re.search(r'epoch 2, total time ([0-9.]+)', timing[0])[1])
tail = (root / 'normal_epoch3_maintenance_observation/train_tail.txt').read_bytes()
validation = list(re.finditer(rb'1189/1189 \[([\d:]+)', tail))
validation_seconds = 0
for part in validation[-1][1].split(b':'):
    validation_seconds = validation_seconds * 60 + int(part)
assert training_seconds == 15987.77 and validation_seconds == 2278
observed = datetime.datetime.fromisoformat(observation['observed_cst'])
remaining_updates = 4583 - 77
estimate = observed + datetime.timedelta(seconds=remaining_updates * training_seconds / 4583 + validation_seconds)
due = estimate - datetime.timedelta(seconds=300)
plan = dict(status='SINGLE_NEXT_OBSERVATION_PLANNED_FROM_MEASURED_RATE',
    due_cst=due.isoformat(), estimated_epoch3_full_validation_end_cst=estimate.isoformat(),
    advance_seconds=300, later_poll_seconds=240, observations=1, neural_calls=0,
    basis='Completed E2 training15987.77s/4583 and full validation2278s; observed E3 77/4583 at13:08:35.136377. Estimate only.',
    measured_previous_epoch_training_seconds=training_seconds,
    measured_previous_epoch_validation_seconds=validation_seconds,
    current_epoch=3, current_epoch_updates=77, updates_per_epoch=4583,
    current_observed_cst=observed.isoformat(), controller_alive_at_observation=True,
    child_alive_at_observation=True, source_changed=False, job_restart=False,
    full_goal_complete=False)
path = root / 'NORMAL_EPOCH3_BOUNDARY_PLAN.json'
assert not path.exists()
path.write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
source = (root / 'observe_normal_epoch2_final_authorized.py').read_text(encoding='utf-8')
source = source.replace('NORMAL_EPOCH2_FINAL_OBSERV', 'NORMAL_EPOCH3_BOUNDARY_OBSERV')
source = source.replace('NORMAL_EPOCH2_FINAL_PLAN.json', 'NORMAL_EPOCH3_BOUNDARY_PLAN.json')
source = source.replace('normal_epoch2_final_observation', 'normal_epoch3_boundary_observation')
path = root / 'observe_normal_epoch3_boundary_authorized.py'
assert not path.exists()
compile(source, str(path), 'exec')
path.write_text(source, encoding='utf-8')
public = dict(observed_cst=observation['observed_cst'], metrics=observation['metrics'],
    epoch_training_timing_lines=observation['epoch_training_timing_lines'],
    completed_epoch2_validation_seconds=validation_seconds,
    epoch3_updates_observed=77, epoch3_elapsed_seconds_observed=374,
    gpu_csv=observation['gpu_csv'], storage=observation['storage'],
    weight_file_metadata=observation['weight_file_metadata'],
    cache_sizes_only=[dict(path=row['path'], exists=row['exists'],
        total_bytes=int(row['du_lines'][-1].split('\t')[0]) if row['exists'] else None)
        for row in observation['cache_inventory']],
    cache_deleted=False, other_tmp_directory_names_published=False,
    new_neural_calls=0, training_source_changed=False, full_goal_complete=False)
(root / 'NORMAL_EPOCH3_MAINTENANCE_PUBLIC_SUMMARY.json').write_text(json.dumps(public, indent=2) + '\n', encoding='utf-8')
for path in [root.parent.parent / 'pvground_referit_mask_reference_20261006/current_research_goals.json',
             root.parent.parent / 'pvground_query_supported_geometry_20261005/active_continuation_state.json']:
    value = json.loads(path.read_bytes())
    value.update(current_turn_classification='ACTUAL_PROGRESS_E2_FULL9508_OBSERVED_E3_TIMING_DISK_SCOPE_CHECKED',full_goal_complete=False)
    value['normal_training_requirement'].update(
        current_normal_observer_session_closed=True, current_normal_observer_session_consumed=43088,
        current_normal_epoch3_updates_observed=77, current_normal_epoch3_plan=plan,
        current_normal_next_observation_plan=plan,
        next_normal_training_observation_not_before=due.isoformat(),
        current_normal_maintenance_public_summary=public,
        last_actual_progress='E2 full9508 5552/4552, E3 entered; completed E2 training/validation durations collected, cache scope inspected without deletion, next near-end node derived')
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status='E3_BOUNDARY_NODE_PREPARED', due_cst=due.isoformat(),
                     estimated_finish_cst=estimate.isoformat(), training_changes=0)))
