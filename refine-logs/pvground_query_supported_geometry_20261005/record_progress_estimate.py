"""Calibrate the schedule from actual logged control-arm timings."""
import datetime
import json
from pathlib import Path

local = Path(__file__).resolve().parent
assert not (local / 'PROGRESS_ESTIMATE.json').exists()
record = json.loads((local / 'FIRST_PROGRESS_RESOURCE_CHECK.json').read_bytes())
assert record['status']['arm'] == 'control' and record['status']['mode'] == 'train'
assert record['training']['completed_logged_updates'] == 3723
assert record['training']['logged_fit_rows'] == 29778
initial = record['completed_stage_receipts']['initial']
progress = record['eval_progress'][0]
assert initial['rows'] == progress['total'] == 6887 and progress['stage'] == 'terminal'
observed = datetime.datetime.fromisoformat(record['observed_remote_time'])
started = datetime.datetime.fromisoformat(record['status']['started_cst'])
initial_finished = datetime.datetime.fromisoformat(initial['time_cst'])
startup = (initial_finished - started).total_seconds() - initial['elapsed_seconds']
seconds_per_eval_row = progress['seconds'] / progress['rows']
control_fit_finish = observed + datetime.timedelta(seconds=(6887-progress['rows'])*seconds_per_eval_row)
formal_seconds = startup + 9508*seconds_per_eval_row
control_formal_finish = control_fit_finish + datetime.timedelta(seconds=formal_seconds)
strategy_seconds = startup + initial['elapsed_seconds'] + record['training']['last_record']['cumulative_seconds'] + 6887*seconds_per_eval_row + formal_seconds
pair_finish = control_formal_finish + datetime.timedelta(seconds=strategy_seconds)
stamp = datetime.datetime.now().astimezone().isoformat()
estimate = dict(time_cst=stamp, actual_observation_cst=record['observed_remote_time'],
    observed_control_updates=3723, observed_control_fit_rows=29778,
    observed_terminal_eval_rows=progress['rows'], terminal_eval_total=6887,
    estimated_control_fit_finish_cst=control_fit_finish.isoformat(),
    estimated_control_formal_finish_cst=control_formal_finish.isoformat(),
    estimated_pair_finish_cst=pair_finish.isoformat(),
    seconds_per_eval_row=seconds_per_eval_row, observed_control_fit_seconds=record['training']['last_record']['cumulative_seconds'],
    estimated_startup_seconds=startup,
    basis='Actual control startup/initial6887/3723 fit updates/terminal progress; extrapolate same throughput for formal and strategy. Last logged eval snapshot may lag; strategy gradient throughput is not yet measured. Estimates are not accuracy or completion evidence.',
    new_formal_accuracy_result=False, data_free_bytes=record['data_free_bytes'],system_free_bytes=record['system_free_bytes'])
(local / 'PROGRESS_ESTIMATE.json').write_text(json.dumps(estimate, indent=2) + '\n', encoding='utf-8')
state = json.loads((local / 'active_continuation_state.json').read_bytes())
state.update(time_cst=stamp, latest_actual_progress_check=str(local / 'FIRST_PROGRESS_RESOURCE_CHECK.json'),
             current_fit_updates_unobserved=False, observed_arm='control', observed_mode='train',
             control_updates_completed=3723, control_terminal_eval_rows=progress['rows'],
             query_supported_updates_unobserved=True, formal_results_unobserved=True,
             estimated_pair_finish_cst=pair_finish.isoformat(), estimated_control_formal_finish_cst=control_formal_finish.isoformat(),
             current_goal_turn_classification='PROGRESS_ACTUAL_SCHEDULED_TRAIN_AND_RESOURCE_CHECK')
(local / 'active_continuation_state.json').write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
with Path('C:/Users/gb/memory/2026-10-05.md').open('a', encoding='utf-8') as stream:
    stream.write('\nPVGround '+stamp+': scheduled actual control progress3723/3723 updates,29778 rows, terminal6887 eval last logged5120; no formal result. GPU6981MiB, datafree2258145280B/systemfree56098816B. Calibrated pair estimate '+pair_finish.isoformat()+', same-throughput assumption; strategy throughput unmeasured. Observer11178/controller584730 alive at actual11:05 status. Previous goal turn verified wait; current actual progress.\n')
print(json.dumps(estimate), flush=True)
