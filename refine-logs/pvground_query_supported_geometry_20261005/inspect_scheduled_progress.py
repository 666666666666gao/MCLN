"""One scheduled read-only progress/resource observation; no model execution."""
import datetime
import json
import os
from pathlib import Path
import shlex
import paramiko

local = Path(__file__).resolve().parent
spec = json.loads((local / 'control_spec.json').read_bytes())
launch = json.loads((local / 'fit_launch.json').read_bytes())
assert not (local / 'FIRST_PROGRESS_RESOURCE_CHECK.json').exists()
probe = '''
import datetime,json,shutil,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);status=json.loads((root/'fit_status.json').read_bytes())
output=root/status['arm'];files=[path.name for path in output.iterdir()]
training=None
if 'train.jsonl' in files:
    records=[json.loads(line) for line in (output/'train.jsonl').read_bytes().split(b'\\n')[:-1]]
    assert [entry['step'] for entry in records]==list(range(1,len(records)+1))
    training=dict(completed_logged_updates=len(records),logged_fit_rows=sum(len(entry['rows']) for entry in records),
        last_record={key:records[-1][key] for key in ('step','total_steps','seconds','cumulative_seconds','loss','extra_geometry_loss','gradient_norm')})
log=(output/(status['mode']+'.log')).read_bytes().split(b'\\n')[:-1]
eval_progress=[json.loads(line[len(b'READBACK_EVAL_PROGRESS '):]) for line in log if line.startswith(b'READBACK_EVAL_PROGRESS ')]
receipts={}
for stage in ('initial','terminal','formal'):
    path=output/stage/'receipt.json'
    if path.is_file():
        receipts[stage]=json.loads(path.read_bytes())
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits']).decode().strip()
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits']).decode().strip()
print(json.dumps(dict(observed_remote_time=datetime.datetime.now().astimezone().isoformat(),status=status,
    training=training,eval_progress=eval_progress[-1:],completed_stage_receipts=receipts,
    weight_files={name:(output/name).stat().st_size for name in files if name.endswith('.pth')},
    data_free_bytes=shutil.disk_usage('/root/autodl-tmp').free,system_free_bytes=shutil.disk_usage('/').free,
    gpu=gpu,gpu_compute_processes=processes,inference_or_optimizer_replayed=False)))
'''
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
command = shlex.join([spec['runtime'] + '/venv/bin/python', '-B', '-c', probe, launch['root']])
_, stdout, stderr = client.exec_command(command, timeout=60)
raw = stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
record = json.loads(raw)
record['local_recorded_time_cst'] = datetime.datetime.now().astimezone().isoformat()
(local / 'FIRST_PROGRESS_RESOURCE_CHECK.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
client.close()
print(json.dumps(record), flush=True)
