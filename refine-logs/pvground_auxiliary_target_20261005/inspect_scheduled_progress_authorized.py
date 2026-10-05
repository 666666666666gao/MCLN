"""One scheduled read-only progress/resource check; no inference or optimizer."""
import json
import os
from pathlib import Path
import shlex

import paramiko


local = Path(__file__).resolve().parent
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
python = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
probe = '''
import datetime,hashlib,json,shutil,statistics,subprocess
from pathlib import Path
root=Path('/root/autodl-tmp/pvground_auxiliary_target_20261005')
status=json.loads((root/'fit_status.json').read_bytes())
spec=json.loads((root/'control_spec.json').read_bytes())
pattern='^'+spec['runtime']+'/venv/bin/python -B -u '+str(root)+'/controller.py --phase fit$'
process=subprocess.run(['pgrep','-af',pattern],stdout=subprocess.PIPE).stdout.decode().strip()
arm=root/status['arm']
receipts={name:json.loads((arm/name).read_bytes()) for name in
    ('initial/receipt.json','terminal/receipt.json','receipt.json','formal/receipt.json') if (arm/name).exists()}
logs=arm/'train.jsonl'
progress=None
if logs.exists():
    lines=logs.read_bytes().split(b'\\n')[:-1]
    if lines:
        last=json.loads(lines[-1])
        tail=[json.loads(line) for line in lines[-512:]]
        progress=dict(last_logged_step=last['step'],total_steps=last['total_steps'],
            complete_log_records=len(lines),fit_cumulative_seconds=last['cumulative_seconds'],
            last512_mean_step_seconds=statistics.mean(x['seconds'] for x in tail),
            last_loss=last['loss'],last_gradient_norm=last['gradient_norm'],
            auxiliary_target_mode=last['auxiliary_target_mode'],extra_counts=last['extra_counts'])
eval_rows={}
for name in ('initial','terminal','formal'):
    path=arm/name/'rows.jsonl'
    if path.exists():
        eval_rows[name]=len(path.read_bytes().split(b'\\n'))-1
best=Path(spec['geometry_terminal'])
best_sha=hashlib.sha256(best.read_bytes()).hexdigest()
assert best_sha==spec['geometry_terminal_sha256']
weights={name:(arm/name).stat().st_size for name in ('latest.pth','terminal.pth') if (arm/name).exists()}
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),status=status,
    controller_alive=bool(process),process=process,progress=progress,complete_receipts=receipts,
    eval_rows_written=eval_rows,active_arm_weights=weights,
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True).strip(),
    gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],text=True).strip(),
    system_free_bytes=shutil.disk_usage('/').free,data_free_bytes=shutil.disk_usage(root).free,
    protected_best_sha256_exact=True,protected_best_bytes=best.stat().st_size,
    inference_or_optimizer_replayed=False)
print(json.dumps(record))
'''
_, stdout, stderr = client.exec_command(shlex.join([python, '-B', '-c', probe]), timeout=90)
raw = stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
record = json.loads(raw)
name = 'SCHEDULED_PROGRESS_' + record['time_cst'].split('T')[1][:8].replace(':', '') + '.json'
(local / name).write_bytes(raw)
client.close()
print(json.dumps(record), flush=True)
