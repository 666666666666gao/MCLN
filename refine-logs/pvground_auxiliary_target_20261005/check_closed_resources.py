"""Check actual retained files and capacity after closed, audited cleanup."""
import datetime
import json
import os
from pathlib import Path
import shlex
import paramiko

local = Path(__file__).resolve().parent
wait = json.loads((local / 'fit_wait.json').read_bytes())
assert wait['observer_closed'] and wait['terminal']['exitcode'] == 0
retention = json.loads((local / 'weight_retention.json').read_bytes())
assert retention['status'] == 'CLOSED_NONBEST_WEIGHTS_REMOVED'
assert not (local / 'CLOSED_RESOURCES.json').exists()
spec = json.loads((local / 'control_spec.json').read_bytes())
probe = '''
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
spec=json.loads(sys.argv[1]);retention=json.loads(sys.argv[2])
official=json.loads((Path(spec['runtime'])/'env_spec.json').read_bytes())['weight_dirs']['scanrefer']
assert official['sha256']==spec['checkpoint_sha256']
parents={spec['base_terminal']:spec['base_terminal_sha256'],official['path']:official['sha256']}
best=retention['retained_best'];parents[best['path']]=best['sha256']
identities={}
for name,digest in parents.items():
    path=Path(name);actual=hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual==digest
    identities[name]=dict(bytes=path.stat().st_size,sha256=actual)
for entry in retention['deleted']:
    assert not Path(entry['path']).exists()
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits']).decode().strip()
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits']).decode().strip()
print(json.dumps(dict(protected_weights=identities,deleted_weights_absent=True,
    data_free_bytes=shutil.disk_usage('/root/autodl-tmp').free,
    system_free_bytes=shutil.disk_usage('/').free,gpu=gpu,gpu_compute_processes=processes,
    inference_or_optimizer_replayed=False)))
'''
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
command = shlex.join([spec['runtime'] + '/venv/bin/python', '-c', probe,
                      json.dumps(spec), json.dumps(retention)])
_, stdout, stderr = client.exec_command(command, timeout=120)
raw = stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
record = json.loads(raw)
record['time_cst'] = datetime.datetime.now().astimezone().isoformat()
(local / 'CLOSED_RESOURCES.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
client.close()
print(json.dumps(record), flush=True)
