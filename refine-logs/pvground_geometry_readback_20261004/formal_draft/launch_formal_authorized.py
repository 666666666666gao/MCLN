"""Launch only after successful V2 runtime preflight and this exact formal source gate."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

local = Path(__file__).resolve().parent
preflight = local.parent / 'revision2'
assert not (local / 'launch.json').exists()
review = json.loads((local / 'READBACK_FORMAL_SOURCE_REVIEW.json').read_bytes())
assert review['verdict'] in ('PASS', 'WARN') and not review['blocking_findings']
assert review['execution_scope'] == 'SOURCE_ONLY'
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
wait = json.loads((preflight / 'readback_preflight_wait.json').read_bytes())
assert wait['observer_closed'] and not wait['terminal']['controller_alive']
assert wait['terminal']['exitcode'] == 0 and wait['terminal']['status']['status'] == 'complete'
source = json.loads((local / 'FORMAL_DRAFT_SOURCE_CHECK.json').read_bytes())
specs = {arm: json.loads((local / (arm + '_fit_spec.json')).read_bytes())
         for arm in ('evidence_hidden', 'evidence_visible')}
root = '/root/autodl-tmp/pvground_readback_fit_20261004'
reserve = 512 * 1024**2
for arm, spec in specs.items():
    assert spec['root'] == root + '/' + arm and spec['updates'] == 3723 and spec['fit_passes'] == 1
    assert spec['batch_size'] == 8 and spec['seed'] == 2027
    proof = json.loads((preflight / 'complete_preflight' / arm / 'preflight.json').read_bytes())
    assert proof['status'] == 'pass' and proof['optimizer_steps'] == 2 and proof['batch_size'] == 8
    assert proof['geometry_provider_and_g_states_exact'] and proof['zero_residual_native_semantic_exact']
    assert proof['use_geometry_evidence'] == spec['use_geometry_evidence'] == (arm == 'evidence_visible')
    assert proof['optimizer_exact_check']['all_keys_moments_steps_and_groups_exact']
    reserve += 2 * proof['serialization_bytes']
    tested = json.loads((preflight / (arm + '_preflight_spec.json')).read_bytes())
    for name, digest in source['runner_files'].items():
        raw = (local / 'runtime_bundle' / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest
        if name != 'run_readback_fit.py':
            assert tested['runner_files'][name] == digest
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
    password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
spec = specs['evidence_hidden']
with sftp.open(spec['runtime'] + '/env_spec.json', 'rb') as stream:
    environment = json.loads(stream.read())
python = spec['runtime'] + '/venv/bin/python'
probe = '''
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);closed=Path(sys.argv[2]);reserve=int(sys.argv[3]);spec=json.loads(sys.argv[4])
assert not root.exists()
assert json.loads((closed/'status.json').read_bytes())['status']=='complete'
assert (closed/'controller.exit').read_text().strip()=='0'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
for name in ('base_terminal','geometry_terminal'):
    assert hashlib.sha256(Path(spec[name]).read_bytes()).hexdigest()==spec[name+'_sha256']
assert hashlib.sha256(Path(spec['source_port']).read_bytes()).hexdigest()==spec['source_port_sha256']
free=shutil.disk_usage(root.parent).free
assert free>=reserve
print(json.dumps(dict(GPU_idle=True,prior_controller_closed=True,directory_free_bytes=free,
    system_free_bytes=shutil.disk_usage('/').free,required_reserve_bytes=reserve,protected_best=[5616,4506])))
'''
_, stdout, stderr = client.exec_command(shlex.join([python, '-c', probe, root,
    '/root/autodl-tmp/pvground_readback_preflight_20261004_v2', str(reserve), json.dumps(spec)]), timeout=60)
raw = stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
resources = json.loads(raw)
(local / 'resource_check.json').write_bytes(raw)
sftp.mkdir(root)
for arm in specs:
    sftp.mkdir(root + '/' + arm)
payloads = {name: (local / name).read_bytes() for name in ('controller.py', 'EXPERIMENT_PLAN_READBACK_FIT.md',
    'READBACK_FORMAL_SOURCE_REVIEW.json', 'FORMAL_DRAFT_SOURCE_CHECK.json')}
for arm, spec in specs.items():
    payloads[arm + '/spec.json'] = (local / (arm + '_fit_spec.json')).read_bytes()
    for name in source['runner_files']:
        payloads[arm + '/' + name] = (local / 'runtime_bundle' / name).read_bytes()
for name, raw in payloads.items():
    with sftp.open(root + '/' + name, 'wx') as stream:
        stream.write(raw)
    with sftp.open(root + '/' + name, 'rb') as stream:
        assert stream.read() == raw
command = shlex.join(['flock', '-n', environment['resource_limits']['gpu_lock'], python,
    '-B', '-u', root + '/controller.py'])
inner = command + ' > ' + shlex.quote(root + '/controller.log') + ' 2>&1; code=$?; printf "%s\\n" "$code" > ' + shlex.quote(root + '/controller.exit') + '; exit "$code"'
screen = 'pvg_readback_fit_20261004'
_, stdout, stderr = client.exec_command('screen -dmS ' + screen + ' bash -c ' + shlex.quote(inner), timeout=30)
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
pattern = '^' + python + ' -B -u ' + root + '/controller.py$'
_, stdout, stderr = client.exec_command('pgrep -af ' + shlex.quote(pattern), timeout=30)
process = stdout.read().decode().strip()
assert stdout.channel.recv_exit_status() == 0 and len(process.splitlines()) == 1, stderr.read().decode()
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), root=root, process=process,
    screen=screen, resources=resources, order=list(specs), primary_mode='bbs', primary_threshold=.5,
    fit_rows_per_arm=29778, batch_size=8, effective_batch=8, tail_batch=2, updates_per_arm=3723,
    completed=False, accuracy_result=False, first_check_seconds=6380, later_poll_seconds=240,
    estimate_seconds=16200, estimate_basis='previous actual face fit6561seconds+formal1478seconds per arm; small readback/evaluation overhead')
(local / 'launch.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
with sftp.open(root + '/launch.json', 'wx') as stream:
    stream.write((local / 'launch.json').read_bytes())
sftp.close()
client.close()
print(json.dumps(record), flush=True)
