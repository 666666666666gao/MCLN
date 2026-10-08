"""Deploy one read-only parent batch24 preflight using the existing runtime."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

root = Path(__file__).resolve().parent
assert not (root / 'PREFLIGHT_LAUNCH.json').exists()
spec = json.loads((root / 'spec.json').read_bytes())
remote = '/root/autodl-tmp/pvground_pretrained_protocol_20261009'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
check = """import json,shutil,subprocess
print(json.dumps(dict(gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits']).decode(), compute_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']).decode(),data_free=shutil.disk_usage('/root/autodl-tmp').free)))
"""
_, stdout, stderr = client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python', '-B', '-c', check]), timeout=30)
resources = json.loads(stdout.read())
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
assert int(resources['gpu'].strip().split(',')[1]) < 500
assert not resources['compute_processes'].strip()
assert resources['data_free'] >= 240000000
sftp = client.open_sftp()
files = []
for name in ('evaluate_parent_batches.py', 'spec.json'):
    data = (root / name).read_bytes()
    with sftp.open(remote+'/'+name, 'wx') as stream:
        stream.write(data)
    with sftp.open(remote+'/'+name, 'rb') as stream:
        actual = stream.read()
    assert actual == data
    files.append(dict(name=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
env = json.loads((root/'actual_env_spec.json').read_bytes())['env']
command = shlex.join(['env']+[key+'='+value for key,value in env.items()]+[spec['runtime']+'/venv/bin/python','-B','-u',remote+'/evaluate_parent_batches.py','--spec',remote+'/spec.json','--phase','preflight'])
body = command+' >'+shlex.quote(remote+'/preflight.log')+' 2>&1\nresult=$?\nprintf "%s\\n" "$result" >'+shlex.quote(remote+'/preflight.exit')+'\nexit "$result"\n'
with sftp.open(remote+'/run_preflight.sh','wx') as stream:
    stream.write(body.encode())
sftp.close()
screen = 'pvg_parent_batch24_preflight_20261009'
_, stdout, stderr = client.exec_command(shlex.join(['screen','-dmS',screen,'bash',remote+'/run_preflight.sh']),timeout=30)
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
_, stdout, stderr = client.exec_command('pgrep -af '+shlex.quote('[e]valuate_parent_batches.py --spec '+remote+'/spec.json --phase preflight'),timeout=30)
process = stdout.read().decode()
assert stdout.channel.recv_exit_status() == 0 and process.strip(), stderr.read().decode()
client.close()
stamp = datetime.datetime.now().astimezone()
receipt = dict(status='READ_ONLY_BATCH24_PREFLIGHT_LAUNCHED_NOT_COMPLETED',time_cst=stamp.isoformat(),remote_root=remote,screen=screen,process=process,resources=resources,files=files,expected_seconds=420,first_outcome_check_cst=(stamp+datetime.timedelta(seconds=360)).isoformat(),later_poll_seconds=240,estimate_basis='Existing official1234-state load, cached scan dataset, first24 language parses and one actual batch24; no optimizer or full validation.',new_weights=0,optimizer_updates=0,formal_accuracy_result=False,current_best_unchanged=True)
(root/'PREFLIGHT_LAUNCH.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt),flush=True)
