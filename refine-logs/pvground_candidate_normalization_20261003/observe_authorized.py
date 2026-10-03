"""One read-only observation of the actual normalization controller and GPU."""
import datetime
import json
import os
from pathlib import Path
import shlex
import paramiko

local = Path(__file__).parent
launch = json.loads((local / 'launch.json').read_bytes())
spec = json.loads((local / 'normalized_spec.json').read_bytes())
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
python = spec['runtime'] + '/venv/bin/python'
probe = '''
import json, subprocess, sys
from pathlib import Path
root=Path(sys.argv[1]); pid=int(sys.argv[2])
status=json.loads((root/'status.json').read_bytes())
log=root/'normalized'/('train.log' if status['stage'].endswith('/train') else 'formal.log')
lines=log.read_text().splitlines()
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
receipts={}
for name in ('normalized/receipt.json','normalized/formal/receipt.json'):
    path=root/name
    if path.exists(): receipts[name]=json.loads(path.read_bytes())
print(json.dumps(dict(status=status,controller_alive=Path('/proc/%d'%pid).exists(),
    log_bytes=log.stat().st_size,log_tail=lines[-3:],gpu_compute=gpu,receipts=receipts)))
'''
pid = launch['process'].split()[0]
_, stdout, stderr = client.exec_command(shlex.join([python, '-c', probe, launch['root'], pid]), timeout=60)
raw = stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
record = json.loads(raw)
record['observed_cst'] = datetime.datetime.now().astimezone().isoformat()
path = local / ('observation_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.json')
path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
client.close()
print(json.dumps(dict(observation=str(path), status=record['status'],
    controller_alive=record['controller_alive'],gpu_compute=record['gpu_compute'],
    log_bytes=record['log_bytes'],log_tail=record['log_tail'])), flush=True)
