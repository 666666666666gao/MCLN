"""Re-estimate from actual completed updates and evaluation receipts only."""
import datetime
import hashlib
import json
from pathlib import Path


root = Path(__file__).resolve().parent
path = root / 'SCHEDULED_PROGRESS_185943.json'
progress = json.loads(path.read_bytes())
assert progress['controller_alive'] and progress['status']['arm']=='control'
assert progress['progress']['last_logged_step']==3723
initial = progress['complete_receipts']['initial/receipt.json']
old = root.parent / 'pvground_query_supported_geometry_20261005/complete/control'
old_initial = json.loads((old / 'initial/receipt.json').read_bytes())
old_terminal = json.loads((old / 'terminal/receipt.json').read_bytes())
old_formal = json.loads((old / 'formal/receipt.json').read_bytes())
factor = initial['elapsed_seconds'] / old_initial['elapsed_seconds']
rebuild = (datetime.datetime.fromisoformat(initial['time_cst']) -
           datetime.datetime.fromisoformat(progress['status']['started_cst'])).total_seconds() - initial['elapsed_seconds']
terminal_seconds = old_terminal['elapsed_seconds'] * factor
formal_seconds = old_formal['elapsed_seconds'] * factor
remaining_terminal = terminal_seconds * (6887-progress['eval_rows_written']['terminal'])/6887
train_seconds = progress['progress']['fit_cumulative_seconds']
now = datetime.datetime.fromisoformat(progress['time_cst'])
control_train_end = now + datetime.timedelta(seconds=remaining_terminal)
control_formal_end = control_train_end + datetime.timedelta(seconds=rebuild+formal_seconds)
member_seconds = rebuild+initial['elapsed_seconds']+train_seconds+terminal_seconds+rebuild+formal_seconds
pair_end = control_formal_end + datetime.timedelta(seconds=member_seconds)
record = dict(status='ESTIMATE_FROM_ACTUAL_FIRST_CHECK_NOT_TERMINAL',
              actual_check_cst=progress['time_cst'],last_logged_control_step=3723,
              initial_holdout_complete_rows=6887,terminal_holdout_written_rows=4286,
              formal_complete=False,control_formal_end_estimate_cst=control_formal_end.isoformat(),
              pair_end_estimate_cst=pair_end.isoformat(),
              next_manual_outcome_check_cst=(control_formal_end-datetime.timedelta(minutes=2)).isoformat(),
              actual_control_train_seconds=train_seconds,observed_rebuild_and_initial_setup_seconds=rebuild,
              evaluation_cost_factor_vs_closed_predecessor=factor,
              control_terminal_eval_estimated_seconds=terminal_seconds,
              formal_eval_estimated_seconds=formal_seconds,
              basis_progress_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              caveat='Approximate stage estimate, not current server completion; assumes subsequent cost comparable to measured control and prior full9508. Existing observer alone continues240s; no new observer or GPU work.')
(root/'FIRST_CHECK_ESTIMATE.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record),flush=True)
