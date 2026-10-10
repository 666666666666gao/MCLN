"""Complete the due epoch resource check; no metric or model computation."""
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
assert not (root/'EPOCH1_RESOURCE_ATTEMPT2_RECEIPT.json').exists()
observation = json.loads((root/'NORMAL_FIRST_OBSERVATION.json').read_bytes())
assert [row['epoch'] for row in observation['metrics']] == [0,1]
assert datetime.datetime.now().astimezone() >= datetime.datetime.fromisoformat(observation['first_due_cst'])
launch = json.loads((root/'NORMAL_LAUNCH.json').read_bytes())
start = json.loads((root/'NORMAL_START_WITNESS.json').read_bytes())
code = r'''import datetime,json,shutil,subprocess,sys
from pathlib import Path
b=json.load(sys.stdin);root=Path(b['root'])
assert root==Path('/root/autodl-tmp/pvground_native_joint_controls_20261010/selected_mask_off')
processes=[]
for pid,args in [(b['controller_pid'],b['controller_argv'][4:]),(b['child_pid'],b['child_argv'])]:
 p=Path('/proc')/str(pid);alive=p.exists()
 if alive:assert p.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')==args
 processes.append(dict(pid=pid,alive=alive))
disks={path:dict(zip(['total','used','free'],shutil.disk_usage(path))) for path in ['/', '/root/autodl-tmp']}
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits']).decode().strip()
weights=[]
for p in (root/'logs').rglob('*.pth'):
 weights.append(dict(path=str(p),bytes=p.stat().st_size))
print(json.dumps(dict(status='DUE_EPOCH1_RESOURCE_ATTEMPT2_CHECK_ONLY',observed_cst=datetime.datetime.now().astimezone().isoformat(),
 processes=processes,disks=disks,gpu_csv=gpu,weight_sizes=weights,neural_calls=0,metric_file_reads=0,
 model_source_changes=0,training_restart=False,checkpoint_deletions=0)))
'''
witness = json.loads((root.parent.parent/'SCP_TRANSPORT_WITNESS.json').read_bytes())
env = dict(os.environ,SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',SSH_ASKPASS_REQUIRE='force',DISPLAY='codex-byte-transfer')
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
argv = ['C:/Windows/System32/OpenSSH/ssh.exe','-T','-p','33476','-o','ProxyCommand=none','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts','-o','HostKeyAlgorithms='+witness['negotiated_host_key_algorithm'],'-o','NumberOfPasswordPrompts=1','root@region-9.autodl.pro',shlex.join([runtime,'-B','-u','-c',code])]
body=dict(root=launch['root'],controller_pid=launch['controller_pid'],controller_argv=launch['controller_argv'],child_pid=start['child_pid'],child_argv=start['child_argv'])
result=subprocess.run(argv,env=env,input=json.dumps(body).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
(root/'EPOCH1_RESOURCE_ATTEMPT2_STDOUT.json').write_bytes(result.stdout)
(root/'EPOCH1_RESOURCE_ATTEMPT2_STDERR.txt').write_bytes(result.stderr)
(root/'EPOCH1_RESOURCE_ATTEMPT2_EXIT.json').write_text(json.dumps(dict(exit_code=result.returncode))+'\n')
assert result.returncode==0,'Inspect the preserved private resource-check stderr; do not restart training'
value=json.loads(result.stdout)
(root/'EPOCH1_RESOURCE_ATTEMPT2_RECEIPT.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value),flush=True)

