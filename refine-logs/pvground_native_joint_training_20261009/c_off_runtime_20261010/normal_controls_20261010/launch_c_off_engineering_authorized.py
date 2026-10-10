"""Admit the already-planned single-variable C-off native engineering check."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
previous = root.parent
project = previous.parent
assert not (root / 'ENGINEERING_LAUNCH.json').exists()
review_path = root / 'engineering_source_review' / 'SOURCE_REVIEW.json'
review = json.loads(review_path.read_bytes())
assert review['execution_scope'] == 'SOURCE_ONLY'
assert review['verdict'] in ('PASS', 'WARN') and not review['blocking_findings']
names = ['init.json', 'NORMAL_NATIVE_RUN_PROTOCOL.json', 'normal_joint_controller.py',
         'native_c_off_preflight.py', 'c_off_engineering_controller.py']
files = {name: (root / name).read_bytes() for name in names}
for path in [root / name for name in names] + [Path(__file__)]:
    assert review['audited_input_hashes'][str(path.resolve())] == hashlib.sha256(path.read_bytes()).hexdigest()
panel = json.loads((previous / 'normal_degradation_assessment_20261010' / 'actual_panel' /
                    'results' / 'PANEL_RECEIPT.json').read_bytes())
assert panel['new_optimizer_steps'] == 0 and panel['actual_neural_forwards'] == 16
port = json.loads((previous / 'NATIVE_SOURCE_PORT.json').read_bytes())
code = r'''import base64,datetime,hashlib,json,shutil,subprocess,sys
from pathlib import Path
b=json.load(sys.stdin); source=Path(b['port']['model_source'])
root=Path('/root/autodl-tmp/pvground_native_joint_controls_20261010/selected_mask_off')
assert not root.exists()
for name,row in b['port']['files'].items():
 assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
environment=json.loads(Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest()==b['port']['env_spec_sha256']
spec=json.loads(base64.b64decode(b['files']['init.json']['base64']))
assert not spec['use_selected_mask_supervision'] and spec['use_g_supervision']
for key in ('official_checkpoint','g_checkpoint','support_checkpoint','span_checkpoint'):
 h=hashlib.sha256()
 with Path(spec[key]).open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
 assert h.hexdigest()==spec[key+'_sha256']
initial=Path('/root/autodl-tmp/pvground_native_joint_training_20261010/normal/logs/scanrefer/extremal_support/1791573559/best.pth')
assert initial.stat().st_size==615023752
h=hashlib.sha256()
with initial.open('rb') as stream:
 for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
assert h.hexdigest()=='ed8455ddc67e4d17018e9cf499197140e15e35db3f5daee55e0eec6846faf0c4'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
capacity=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total','--format=csv,noheader,nounits']).decode().strip().splitlines()
assert len(capacity)==1
index,name,used,total=[value.strip() for value in capacity[0].split(',')]
assert int(index)==0 and 'A100' in name and int(used)<500 and 40000<=int(total)<=45000
free=shutil.disk_usage(source).free
reserve=846014332+67108864
assert free>reserve
root.mkdir(parents=True); engineering=root/'engineering'; engineering.mkdir()
for name,row in b['files'].items():
 raw=base64.b64decode(row['base64']); assert hashlib.sha256(raw).hexdigest()==row['sha256']
 target=(engineering if name in ('native_c_off_preflight.py','c_off_engineering_controller.py') else root)/name
 target.write_bytes(raw)
admission=dict(status='C_OFF_NATIVE_ENGINEERING_ADMITTED',time_cst=datetime.datetime.now().astimezone().isoformat(),
 env_spec_sha256=b['port']['env_spec_sha256'],runtime=b['runtime'],data_free_bytes=free,reserve_bytes=reserve,
 helpers={name:b['files'][name]['sha256'] for name in ('native_c_off_preflight.py','c_off_engineering_controller.py')},
 source_review_sha256=b['source_review_sha256'],normal_E0_checkpoint=str(initial),
 normal_E0_sha256=h.hexdigest(),normal_training_started=False,formal_accuracy=None)
(engineering/'admission.json').write_text(json.dumps(admission,indent=2)+'\n')
command=['flock','--no-fork','-n',environment['resource_limits']['gpu_lock'],b['runtime'],'-B','-u',str(engineering/'c_off_engineering_controller.py')]
with (engineering/'controller.log').open('wb') as stream:
 process=subprocess.Popen(command,cwd=str(source),stdin=subprocess.DEVNULL,stdout=stream,
  stderr=subprocess.STDOUT,start_new_session=True)
assert process.poll() is None
print(json.dumps(dict(admission=admission,controller_pid=process.pid,controller_argv=command,
 root=str(engineering),model_source=str(source))))
'''
witness = json.loads((project / 'SCP_TRANSPORT_WITNESS.json').read_bytes())
variables = dict(os.environ)
variables.update(SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
                 SSH_ASKPASS_REQUIRE='force', DISPLAY='codex-byte-transfer')
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
argv = ['C:/Windows/System32/OpenSSH/ssh.exe', '-T', '-p', '33476', '-o', 'ProxyCommand=none',
        '-o', 'StrictHostKeyChecking=yes', '-o',
        'UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
        '-o', 'HostKeyAlgorithms=' + witness['negotiated_host_key_algorithm'], '-o', 'NumberOfPasswordPrompts=1',
        'root@region-9.autodl.pro', shlex.join([runtime, '-B', '-u', '-c', code])]
payload = dict(port=port, runtime=runtime, source_review_sha256=hashlib.sha256(review_path.read_bytes()).hexdigest(),
               files={name: dict(base64=base64.b64encode(raw).decode(), sha256=hashlib.sha256(raw).hexdigest())
                      for name, raw in files.items()})
result = subprocess.run(argv, env=variables, input=json.dumps(payload).encode(), stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
(root / 'ENGINEERING_LAUNCH_STDOUT.json').write_bytes(result.stdout)
(root / 'ENGINEERING_LAUNCH_STDERR.txt').write_bytes(result.stderr)
(root / 'ENGINEERING_LAUNCH_EXIT.json').write_text(json.dumps(dict(exit_code=result.returncode)) + '\n')
assert result.returncode == 0, 'See preserved ENGINEERING_LAUNCH_STDERR.txt'
record = json.loads(result.stdout)
started = datetime.datetime.now().astimezone()
record.update(status='C_OFF_NATIVE_ENGINEERING_LAUNCHED_NOT_COMPLETED',time_cst=started.isoformat(),
              first_observation_cst=(started + datetime.timedelta(seconds=300)).isoformat(),
              first_check_seconds=300, later_poll_seconds=240, normal_training_started=False)
(root / 'ENGINEERING_LAUNCH.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
