"""Deploy the reviewed bounded probe to the existing authorized SSH target."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

local = Path(__file__).resolve().parent
assert not (local/'launch.json').exists()
review = json.loads((local/'SOURCE_REVIEW.json').read_bytes())
assert review['verdict'] in ('PASS','WARN') and not review['blocking_findings']
assert review['execution_scope'] == 'SOURCE_ONLY'
for entry in review['reviewed_files']:
    assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() == entry['sha256'], entry['path']
spec = json.loads((local/'spec.json').read_bytes())
root = spec['root']
assert root == '/root/autodl-tmp/pvground_mask_geometry_responsibility_20261005'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
with sftp.open(spec['runtime']+'/env_spec.json','rb') as stream:
    env = json.loads(stream.read())
assert hashlib.sha256(json.dumps(env, sort_keys=True, separators=(',',':')).encode()).hexdigest() == spec['env_spec_sha256']
python = spec['runtime']+'/venv/bin/python'
probe = '''
import json,subprocess,shutil,sys
from pathlib import Path
root=Path(sys.argv[1]);closed=Path('/root/autodl-tmp/pvground_final_quality_20261005')
assert not root.exists()
assert json.loads((closed/'fit_status.json').read_bytes())['status']=='complete'
assert (closed/'fit_controller.exit').read_text().strip()=='0'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert shutil.disk_usage(root.parent).free>64*1024**2
print(json.dumps(dict(GPU_idle=True,closed_prior_job=True,data_free_bytes=shutil.disk_usage(root.parent).free,
    system_free_bytes=shutil.disk_usage('/').free,packages_installed=0,accuracy_result=False)))
'''
_, stdout, stderr = client.exec_command(shlex.join([python,'-c',probe,root]), timeout=60)
raw = stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
(local/'resource_check.json').write_bytes(raw)
sftp.mkdir(root)
for name in ('run_mask_geometry_probe.py','spec.json','controller.py','EXPERIMENT_PLAN.md','SOURCE_REVIEW.md','SOURCE_REVIEW.json'):
    raw = (local/name).read_bytes()
    with sftp.open(root+'/'+name,'wx') as stream:
        stream.write(raw)
    with sftp.open(root+'/'+name,'rb') as stream:
        assert stream.read() == raw
command = shlex.join(['flock','-n',env['resource_limits']['gpu_lock'],python,'-B','-u',root+'/controller.py'])
inner = command+' > '+shlex.quote(root+'/controller.log')+' 2>&1; code=$?; printf "%s\\n" "$code" > '+shlex.quote(root+'/controller.exit')+'; exit "$code"'
screen = 'pvg_mask_geometry_probe_20261005'
_, stdout, stderr = client.exec_command(shlex.join(['screen','-dmS',screen,'bash','-c',inner]), timeout=30)
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
pattern = '^'+python+' -B -u '+root+'/controller.py$'
_, stdout, stderr = client.exec_command('pgrep -af '+shlex.quote(pattern), timeout=30)
process = stdout.read().decode().strip()
assert stdout.channel.recv_exit_status() == 0 and len(process.splitlines()) == 1, stderr.read().decode()
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),root=root,process=process,screen=screen,
    status='LAUNCHED_NOT_COMPLETED',optimizer_steps=0,weight_files_created=0,accuracy_result=False,
    estimated_seconds=480,first_check_seconds=300,later_poll_seconds=240,
    estimate_basis='Warm parent reconstruction plus8 actual fit-batch forwards; no validation or optimizer.')
(local/'launch.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
with sftp.open(root+'/launch.json','wx') as stream:
    stream.write((local/'launch.json').read_bytes())
sftp.close()
client.close()
print(json.dumps(record),flush=True)
