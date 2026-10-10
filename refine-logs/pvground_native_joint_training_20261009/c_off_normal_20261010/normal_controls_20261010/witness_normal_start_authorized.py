"""One startup ownership check; no accuracy polling or model execution."""
import base64
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
project = root.parent.parent
launch = json.loads((root / 'NORMAL_LAUNCH.json').read_bytes())
assert not (root / 'NORMAL_START_WITNESS.json').exists()
witness = json.loads((project / 'SCP_TRANSPORT_WITNESS.json').read_bytes())
environment = dict(os.environ)
environment.update(SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
    SSH_ASKPASS_REQUIRE='force', DISPLAY='codex-byte-transfer')
code = r'''import base64,datetime,json,subprocess,sys
from pathlib import Path
b=json.load(sys.stdin);root=Path(b['root']);pid=b['controller_pid']
assert root==Path('/root/autodl-tmp/pvground_native_joint_controls_20261010/selected_mask_off')
assert Path('/proc/'+str(pid)).exists()
state=json.loads((root/'normal_status.json').read_bytes())
assert state['status']=='running' and state['source_mode']=='extremal_support'
child=state['child_pid'];proc=Path('/proc/'+str(child));assert proc.exists()
argv=proc.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
assert argv==state['child_argv'] and argv[3]==b['model_source']+'/train_dist_mod.py'
assert '--checkpoint_path' not in argv and '--frozen' not in argv
assert argv[argv.index('--max_epoch')+1]=='3' and argv[argv.index('--batch_size')+1]=='8'
parent=[line for line in proc.joinpath('status').read_text().splitlines() if line.startswith('PPid:')]
assert len(parent)==1 and int(parent[0].split()[1])==pid
log=root/'train.log';size=log.stat().st_size
with log.open('rb') as stream:
 head=stream.read(12000);stream.seek(max(0,size-16000));tail=stream.read()
assert b'Traceback (most recent call last)' not in tail
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits']).decode()
print(json.dumps(dict(status='ORDINARY_NATIVE_ENTRY_ALIVE',time_cst=datetime.datetime.now().astimezone().isoformat(),
 controller_pid=pid,child_pid=child,child_argv=argv,remote_status=state,gpu_processes=gpu,
 train_log_bytes=size,log_head_base64=base64.b64encode(head).decode(),log_tail_base64=base64.b64encode(tail).decode(),
 native_entry_not_engineering_probe=True,neural_calls_by_witness=0,accuracy_files_read=0)))
'''
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
argv = ['C:/Windows/System32/OpenSSH/ssh.exe', '-T', '-p', '33476', '-o', 'ProxyCommand=none',
    '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
    '-o', 'HostKeyAlgorithms='+witness['negotiated_host_key_algorithm'], '-o', 'NumberOfPasswordPrompts=1',
    'root@region-9.autodl.pro', shlex.join([runtime, '-B', '-u', '-c', code])]
response = subprocess.run(argv, env=environment, input=json.dumps(launch).encode(),
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
(root/'NORMAL_START_WITNESS_STDOUT.json').write_bytes(response.stdout)
(root/'NORMAL_START_WITNESS_STDERR.txt').write_bytes(response.stderr)
(root/'NORMAL_START_WITNESS_EXIT.json').write_text(json.dumps(dict(exit_code=response.returncode))+'\n')
assert response.returncode == 0, 'Inspect private preserved startup stderr; do not restart training'
value=json.loads(response.stdout)
for source, name in [('log_head_base64','NORMAL_START_LOG_HEAD.txt'), ('log_tail_base64','NORMAL_START_LOG_TAIL.txt')]:
    (root/name).write_bytes(base64.b64decode(value.pop(source)))
value.update(formal_accuracy=None,initial_evaluation_or_fit_phase_requires_log_read=True,
    ordinary_training_completed=False,three_contributions_proven=False,full_goal_complete=False)
(root/'NORMAL_START_WITNESS.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(dict(status=value['status'],child_pid=value['child_pid'],time_cst=value['time_cst'],
    log_bytes=value['train_log_bytes'],gpu_processes=value['gpu_processes'])),flush=True)
