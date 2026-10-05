"""Use the actual whole control arm duration for the remaining member arm."""
import datetime
import hashlib
import json
from pathlib import Path


root=Path(__file__).resolve().parent
intake=json.loads((root/'CONTROL_CLOSED_INTAKE.json').read_bytes())
recount=json.loads((root/'CONTROL_CPU_RECOUNT.json').read_bytes())
assert intake['formal_rows']==9508 and recount['formal_rows']==9508
assert recount['control']['cpu_box_threshold_changes']==0
launch=json.loads((root/'fit_launch.json').read_bytes())
start=datetime.datetime.fromisoformat(launch['time_cst'])
finished=datetime.datetime.fromisoformat(intake['formal_finished_cst'])
duration=(finished-start).total_seconds()
estimate=finished+datetime.timedelta(seconds=duration)
record=dict(status='ACTUAL_CONTROL_COMPLETE_MEMBER_ESTIMATE',control_formal_finished_cst=intake['formal_finished_cst'],
    control_full_arm_elapsed_seconds=duration,member_formal_end_estimate_cst=estimate.isoformat(),
    next_manual_outcome_check_cst=(estimate-datetime.timedelta(minutes=2)).isoformat(),
    basis_intake_sha256=hashlib.sha256((root/'CONTROL_CLOSED_INTAKE.json').read_bytes()).hexdigest(),
    member_result_unobserved=True,full_pair_complete=False,
    scope='Approximate remaining-arm duration based on actual control launch through full formal completion. Native/member qualification may change throughput; stage estimate is not a result. Existing observer alone continues240s.')
assert not (root/'CONTROL_COMPLETE_ESTIMATE.json').exists()
(root/'CONTROL_COMPLETE_ESTIMATE.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record),flush=True)
