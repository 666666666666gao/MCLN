"""Deploy the freshly reviewed fixed-budget pair, reusing actual preflight receipts."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

local = Path(__file__).parent
assert not (local/'launch.json').exists()
review = json.loads((local/'EXPERIMENT_CODE_REVIEW.json').read_bytes())
assert review['verdict'] in ('PASS', 'WARN') and not review['blocking_findings']
assert review['execution_scope'] == 'SOURCE_ONLY'
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
engineering = local.parent/'pvground_whole_mask_integration_20261003'
assert json.loads((engineering/'analysis/SUMMARY.json').read_bytes())['engineering_status'] == 'PASS'
spec = json.loads((local/'whole_range_spec.json').read_bytes())
root = '/root/autodl-tmp/pvground_whole_mask_fit_20261003'
python = spec['runtime']+'/venv/bin/python'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
with sftp.open(spec['runtime']+'/env_spec.json', 'rb') as stream:
    environment = json.loads(stream.read())
probe = '''
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]); preflight=Path('/root/autodl-tmp/pvground_whole_mask_preflight_20261003')
assert not root.exists()
assert json.loads((preflight/'status.json').read_bytes())['status']=='complete'
assert (preflight/'controller.exit').read_text().strip()=='0'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],universal_newlines=True).strip()
proofs=[json.loads((preflight/arm/'preflight.json').read_bytes()) for arm in ('local_range','whole_range')]
assert all(p['status']=='pass' and p['optimizer_steps']==2 for p in proofs)
required=2*max(p['serialization_bytes'] for p in proofs)+256*1024**2
free=shutil.disk_usage(root.parent).free
assert free>=required,(free,required)
print(json.dumps(dict(GPU_idle=True,preflight_complete=True,free_bytes=free,required_bytes=required)))
'''
_, stdout, stderr = client.exec_command(shlex.join([python, '-c', probe, root]), timeout=60)
raw = stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
resources = json.loads(raw)
sftp.mkdir(root)
for name in ('run_whole_mask_fit.py', 'controller.py', 'FORMAL_RANGE_CONTROL_PLAN.md', 'EXPERIMENT_CODE_REVIEW.json'):
    content = (local/name).read_bytes()
    with sftp.open(root+'/'+name, 'wx') as stream:
        stream.write(content)
    with sftp.open(root+'/'+name, 'rb') as stream:
        assert stream.read() == content
modules = json.loads((local/'source_preparation.json').read_bytes())['model_modules_copied_without_changes']
modules += ['initial_range_comparison.py']
for arm in ('local_range', 'whole_range'):
    directory = root+'/'+arm
    sftp.mkdir(directory)
    for name in modules:
        content = (local/name).read_bytes()
        with sftp.open(directory+'/'+name, 'wx') as stream:
            stream.write(content)
        with sftp.open(directory+'/'+name, 'rb') as stream:
            assert stream.read() == content
    content = (local/(arm+'_spec.json')).read_bytes()
    with sftp.open(directory+'/spec.json', 'wx') as stream:
        stream.write(content)
    with sftp.open(directory+'/spec.json', 'rb') as stream:
        assert stream.read() == content
command = shlex.join(['flock', '-n', environment['resource_limits']['gpu_lock'], python, '-B', '-u', root+'/controller.py'])
inner = command+' > '+shlex.quote(root+'/controller.log')+' 2>&1; code=$?; printf "%s\\n" "$code" > '+shlex.quote(root+'/controller.exit')+'; exit "$code"'
screen = 'pvg_whole_mask_fit_20261003'
_, stdout, stderr = client.exec_command('screen -dmS '+screen+' bash -c '+shlex.quote(inner), timeout=30)
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
pattern = '^'+python+' -B -u '+root+'/controller.py$'
_, stdout, stderr = client.exec_command('pgrep -af '+shlex.quote(pattern), timeout=30)
process = stdout.read().decode().strip()
assert stdout.channel.recv_exit_status() == 0 and len(process.splitlines()) == 1, stderr.read().decode()
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), root=root, process=process,
    screen=screen, resources=resources, order=['local_range', 'whole_range'],
    fresh_optimizer=True, base='original_g_5615_4495', seed=2027, batch_size=8,
    steps_per_arm=3723, fit_rows_per_arm=29778, formal_rows_per_arm=9508,
    formal_training_started=True, formal_result_available=False, estimate_hours_pair=[6, 8],
    estimate_basis='historical fit/evaluation estimate, not measured new throughput', poll_seconds=240,
    retention='metric-best plus protected parents and one active recovery; verified nonbest automatically deleted',
    teacher=False, quality_loss=False, contrastive_expansion=False, p2=False, goal_achieved=False)
content = (json.dumps(record, indent=2)+'\n').encode()
(local/'launch.json').write_bytes(content)
with sftp.open(root+'/launch.json', 'wx') as stream:
    stream.write(content)
sftp.close()
client.close()
print(json.dumps(record), flush=True)
