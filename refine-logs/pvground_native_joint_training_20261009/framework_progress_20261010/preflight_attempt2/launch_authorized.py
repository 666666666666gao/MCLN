"""Isolated second engineering run after the closed first failure."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
project = root.parent
assert not (root / 'NATIVE_PREFLIGHT_LAUNCH.json').exists()
failed = json.loads((project / 'native_preflight_failed_attempt1/INTAKE.json').read_bytes())
assert failed['controller_exit'] == 1 and failed['new_neural_calls'] == 0
closed = json.loads((project / 'NATIVE_PREFLIGHT_WAIT.json').read_bytes())
assert closed['observer_closed'] and not closed['terminal']['controller_alive']
review = json.loads((project / 'source_review/NATIVE_M0_SOURCE_REVIEW_R8_1.json').read_bytes())
assert review['execution_scope'] == 'SOURCE_ONLY' and review['verdict'] in ('PASS', 'WARN')
assert not review['blocking_findings']
manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_bytes())
files = {name: (root / name).read_bytes() for name in manifest['files']}
for name, raw in files.items():
    assert hashlib.sha256(raw).hexdigest() == manifest['files'][name]['sha256']
    assert review['audited_input_hashes'][str((root / name).resolve())] == hashlib.sha256(raw).hexdigest()
assert review['audited_input_hashes'][str(Path(__file__).resolve())] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
port = json.loads((project / 'NATIVE_SOURCE_PORT.json').read_bytes())
remote_root = '/root/autodl-tmp/pvground_native_joint_training_20261009/preflight_attempt2'
code = r'''import base64,datetime,hashlib,json,shutil,subprocess,sys
from pathlib import Path
b=json.load(sys.stdin);root=Path(b['root']);prior=root.parent/'preflight'
assert root==Path('/root/autodl-tmp/pvground_native_joint_training_20261009/preflight_attempt2')
assert root.parent.resolve()==root.parent and not root.exists()
assert (prior/'controller.exit').read_text().strip()=='1'
assert not Path('/proc/'+str(b['prior_pid'])).exists()
environment=json.loads(Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest()==b['env_spec_sha256']
for name,row in b['port']['files'].items():
 assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
source=Path(b['port']['model_source'])
for mode in ('whole_support','extremal_support'):
 spec=json.loads((source/'init_manifests'/(mode+'.json')).read_bytes())
 for key in ('official_checkpoint','g_checkpoint','support_checkpoint'):
  h=hashlib.sha256()
  with Path(spec[key]).open('rb') as f:
   for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
  assert h.hexdigest()==spec[key+'_sha256']
assert shutil.disk_usage(root.parent).free>b['reserve']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
capacity=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total','--format=csv,noheader,nounits']).decode().strip().splitlines()
assert len(capacity)==1
index,name,used,total=[v.strip() for v in capacity[0].split(',')]
assert int(index)==0 and 'A100' in name and int(used)<500 and 40000<=int(total)<=45000
root.mkdir()
for name,data in b['files'].items():
 p=root/name;assert p.parent==root
 raw=base64.b64decode(data);p.write_bytes(raw);assert hashlib.sha256(p.read_bytes()).hexdigest()==b['helper_hashes'][name]
admission=dict(status='NATIVE_GPU_PREFLIGHT_ADMITTED_AFTER_ORIGINAL_SPAN_CLOSED',
 time_cst=datetime.datetime.now().astimezone().isoformat(),prior_span_controller_exit_code=0,
 original_span_terminal_verified=True,prior_actual_m0_exit=1,env_spec_sha256=b['env_spec_sha256'],
 helper_files=b['helper_hashes'],data_free_bytes=shutil.disk_usage(root).free,reserve=b['reserve'],GPU_idle=True,
 formal_accuracy=None,normal_epoch_training_started=False,original_failure_preserved=True)
(root/'admission.json').write_text(json.dumps(admission,indent=2)+'\n')
argv=[b['runtime'],'-B','-u',str(root/'native_preflight_controller.py')]
import shlex
command=' '.join(shlex.quote(value) for value in ['flock','-n',environment['resource_limits']['gpu_lock']]+argv)
shell=command+' > '+shlex.quote(str(root/'controller.log'))+' 2>&1; code=$?; printf "%s\\n" "$code" > '+shlex.quote(str(root/'controller.exit'))+'; exit "$code"'
subprocess.run(['screen','-dmS',b['screen'],'bash','-c',shell],check=True)
found=subprocess.check_output(['pgrep','-af','^'+' '.join(shlex.quote(value) for value in argv)+'$']).decode().strip().splitlines()
assert len(found)==1
print(json.dumps(dict(admission=admission,controller_pid=int(found[0].split()[0]),controller_argv=argv)))
'''
witness = json.loads((project / 'SCP_TRANSPORT_WITNESS.json').read_bytes())
environment = dict(os.environ)
environment.update(SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
                   SSH_ASKPASS_REQUIRE='force', DISPLAY='codex-byte-transfer')
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
screen = 'pvg_native_joint_preflight_attempt2_20261010'
command = ['C:/Windows/System32/OpenSSH/ssh.exe', '-T', '-p', '33476', '-o', 'StrictHostKeyChecking=yes',
           '-o', 'UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
           '-o', 'HostKeyAlgorithms=' + witness['negotiated_host_key_algorithm'], '-o', 'NumberOfPasswordPrompts=1',
           'root@region-9.autodl.pro', shlex.join([runtime, '-B', '-u', '-c', code])]
prior_launch = json.loads((project / 'NATIVE_PREFLIGHT_LAUNCH.json').read_bytes())
payload = dict(root=remote_root, port=port, env_spec_sha256=port['env_spec_sha256'], prior_pid=prior_launch['controller_pid'],
               reserve=2605151860, runtime=runtime, screen=screen,
               files={name: base64.b64encode(raw).decode() for name, raw in files.items()},
               helper_hashes={name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()})
result = subprocess.run(command, env=environment, input=json.dumps(payload).encode(),
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
(root / 'LAUNCH_STDOUT.json').write_bytes(result.stdout)
(root / 'LAUNCH_STDERR.txt').write_bytes(result.stderr)
(root / 'LAUNCH_EXIT.json').write_text(json.dumps(dict(exit_code=result.returncode)) + '\n')
assert result.returncode == 0, result.stderr.decode()
record = json.loads(result.stdout)
started = datetime.datetime.now().astimezone()
record.update(status='NATIVE_ENGINEERING_PREFLIGHT_STARTED_NOT_COMPLETED', time_cst=started.isoformat(),
              root=remote_root, screen=screen, estimated_seconds=1800, first_check_seconds=1500,
              first_observation_cst=(started + datetime.timedelta(seconds=1500)).isoformat(), later_poll_seconds=240,
              formal_accuracy=None, normal_epoch_training_started=False,
              attempt=2, computational_model_changes=False, previous_failed_run_preserved=True)
(root / 'NATIVE_PREFLIGHT_LAUNCH.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
