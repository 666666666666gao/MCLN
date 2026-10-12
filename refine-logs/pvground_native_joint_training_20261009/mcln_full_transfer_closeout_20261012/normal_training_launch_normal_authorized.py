"""Start the reviewed normal loop only after actual native training checks."""
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
assert not (root / 'LAUNCH.json').exists()
review = json.loads((root / 'source_review' / 'SOURCE_REVIEW.json').read_bytes())
assert review['verdict'] in ('PASS', 'WARN') and not review['blocking_findings']
for path, digest in review['audited_input_hashes'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
files = {name: base64.b64encode((root / name).read_bytes()).decode()
         for name in ('train_native_mcln_portable.py', 'native_decline_policy.py', 'normal_controller.py')}
body = dict(files=files, source_manifest=json.loads((root.parent / 'SOURCE_MANIFEST.json').read_bytes()),
            storage_budget=json.loads((root / 'STORAGE_BUDGET.json').read_bytes()))
code = r'''import base64,datetime,hashlib,json,shutil,subprocess,sys
from pathlib import Path
body=json.load(sys.stdin)
parent=Path('/root/autodl-tmp/mcln_portable_best_transfer_20261012')
check_root=parent/'native_training_check'
check=json.loads((check_root/'status.json').read_bytes())
assert check['status']=='complete' and check['exit_code']==0
assert not Path('/proc/'+str(check['child_pid'])).exists()
result=check['result']
assert result['status']=='ACTUAL_NATIVE_MCLN_TWO_UPDATES_AND_SAME_PROCESS_FULL_RESTORE_PASS'
assert result['actual_optimizer_updates']==2 and result['same_process_saved_model_optimizer_scheduler_rng_exact']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
source=parent/'MCLN'
for name,row in body['source_manifest'].items():
 path=source/name
 assert path.stat().st_size==row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
root=parent/'normal_training'
assert not root.exists()
environment=json.loads(Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest()=='966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c'
engineering=check_root/'check'/'engineering_full_state.pth'
assert engineering==Path(result['engineering_checkpoint']) and engineering.stat().st_size==result['engineering_checkpoint_bytes']
projected_free=shutil.disk_usage(str(parent)).free+engineering.stat().st_size
assert projected_free>=body['storage_budget']['required_free_bytes'], 'Insufficient room for best, latest and atomic latest replacement'
root.mkdir()
for name,encoded in body['files'].items(): (root/name).write_bytes(base64.b64decode(encoded))
retirement=dict(path=str(engineering),bytes=engineering.stat().st_size,
 engineering_only=True,no_formal_initialization_dependency=True,completed_saved_restore_pass=True)
engineering.unlink()
retirement['removed_cst']=datetime.datetime.now().astimezone().isoformat()
(root/'ENGINEERING_WEIGHT_RETIREMENT.json').write_text(json.dumps(retirement,indent=2)+'\n')
disk=shutil.disk_usage(str(parent))
admission=dict(source=str(source),spec=str(parent/'TRANSFER_SPEC.json'),
 engineering_check=result,engineering_state_carried=False,max_epochs=8,patience_consecutive_strict_declines=2,
 disk_free_bytes=disk.free,storage_budget=body['storage_budget'],core_lr=1e-6,AB_lr=1e-5,batch_size=8,seed=2027)
(root/'admission.json').write_text(json.dumps(admission,indent=2)+'\n')
runtime='/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
command=['flock','--no-fork','-n',environment['resource_limits']['gpu_lock'],runtime,'-B','-u',str(root/'normal_controller.py')]
with (root/'controller.log').open('wb') as stream:
 process=subprocess.Popen(command,cwd=str(source),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
assert process.poll() is None
print(json.dumps(dict(controller_pid=process.pid,root=str(root),command=command,
 launched_cst=datetime.datetime.now().astimezone().isoformat(),scope='FULL_NATIVE_MCLN_TRAINING_MAX8_DECLINE2')))
'''
witness = json.loads(Path('C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/SCP_TRANSPORT_WITNESS.json').read_bytes())
environment = dict(os.environ, SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
                   SSH_ASKPASS_REQUIRE='force', DISPLAY='codex-byte-transfer')
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
command = ['C:/Windows/System32/OpenSSH/ssh.exe', '-T', '-p', '33476', '-o', 'ProxyCommand=none',
           '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
           '-o', 'HostKeyAlgorithms='+witness['negotiated_host_key_algorithm'], '-o', 'NumberOfPasswordPrompts=1',
           'root@region-9.autodl.pro', shlex.join([runtime, '-B', '-u', '-c', code])]
process = subprocess.run(command, input=json.dumps(body).encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         env=environment, timeout=300)
(root / 'LAUNCH_STDOUT.txt').write_bytes(process.stdout)
(root / 'LAUNCH_STDERR.txt').write_bytes(process.stderr)
assert process.returncode == 0, process.stderr.decode(errors='replace')
result = json.loads(process.stdout)
(root / 'LAUNCH.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
