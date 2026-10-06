"""Deploy only the reviewed read-only M0, without installing or saving weights."""
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

local = Path(__file__).resolve().parent
assert not (local/'preflight_launch.json').exists()
review = json.loads((local/'SOURCE_REVIEW.json').read_bytes())
assert review['verdict'] in ('PASS','WARN') and not review['blocking_findings']
assert review['execution_scope'] == 'SOURCE_ONLY'
for entry in review['reviewed_files']:
    assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() == entry['sha256'], entry['path']
spec = json.loads((local/'spec.json').read_bytes())
root = spec['root']
assert root == '/root/autodl-tmp/pvground_mask_extent_diagnostic_20261006'
files = ['run_extent_diagnostic.py','extent_evidence.py','controller.py','analyze_extent.py','spec.json',
         'EXPERIMENT_PLAN.md','EXPERIMENT_TRACKER.md','research_contract.md','SOURCE_REVIEW.json','SOURCE_REVIEW.md']
for name in files:
    if name.endswith('.py'):
        ast.parse((local/name).read_text(encoding='utf-8'), feature_version=(3,7))
client = paramiko.SSHClient(); client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
probe = '''import json,shutil,subprocess
from pathlib import Path
root=Path('/root/autodl-tmp/pvground_mask_extent_diagnostic_20261006');assert not root.exists()
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
assert not gpu,gpu
print(json.dumps(dict(gpu_compute_processes=gpu,data_free_bytes=shutil.disk_usage('/root/autodl-tmp').free)))
'''
_,stdout,stderr = client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python','-B','-c',probe]),timeout=120)
resources = json.loads(stdout.read()); assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
assert resources['data_free_bytes'] > 800*1024*1024
sftp = client.open_sftp(); sftp.mkdir(root)
for name in files:
    data = (local/name).read_bytes()
    with sftp.open(root+'/'+name,'wx') as stream:
        stream.write(data)
    with sftp.open(root+'/'+name,'rb') as stream:
        assert stream.read()==data
sftp.close()
command = shlex.join(['screen','-dmS','pvg_mask_extent_m0_20261006',spec['runtime']+'/venv/bin/python',
                      '-B','-u',root+'/controller.py','--mode','preflight'])
_,stdout,stderr = client.exec_command(command,timeout=120)
stdout.read(); assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
record = dict(status='ACTUAL_READ_ONLY_PREFLIGHT_COMMAND_SUBMITTED',time_cst=datetime.datetime.now().astimezone().isoformat(),
              root=root,command=command,resources=resources,optimization_updates=0,
              files={name:hashlib.sha256((local/name).read_bytes()).hexdigest() for name in files})
(local/'preflight_launch.json').write_text(json.dumps(record,indent=2)+'\n')
client.close(); print(json.dumps(record))
