import datetime,hashlib,json,os,shlex
from pathlib import Path
import paramiko
repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
local=repo/'refine-logs/scanrefer_native_box_transfer_probe_20260907_v1'
local.mkdir(exist_ok=False);(local/'scripts').mkdir();(local/'tests').mkdir()
remote='/root/autodl-tmp/mcln_scanrefer_native_box_transfer_probe_20260907_v1'
base=json.loads((repo/'refine-logs/scanrefer_frozen_readout_probe_20260907_v1/input_manifest.json').read_bytes())
files={}
paths=['scripts/probe_scanrefer_native_box_transfer.py','scripts/native_teacher_box_transfer.py','scripts/scanrefer_data_contract.py','scripts/scanrefer_joint_readout.py','tests/test_native_teacher_box_transfer.py']
for name in paths:
    raw=(repo/name).read_bytes();(local/name).write_bytes(raw);files[name]=hashlib.sha256(raw).hexdigest()
(local/'scripts/__init__.py').write_bytes(b'');files['scripts/__init__.py']=hashlib.sha256(b'').hexdigest()
manifest={k:base[k] for k in ['model_source','source_manifest_sha256','artifacts','data_root','train_superpoint_files','split_salt','split_protocol','split_protocol_sha256','environment_reuse']}
receipt=repo/'refine-logs/scanrefer_mesh_teacher_transfer_20260907_v1/receipt.json'
manifest.update(schema='mcln-native-box-transfer-probe-v1',learning_rate=1e-6,auxiliary_weight=1.,formal_rows=0,checkpoint_writes=0,files=files,
    teacher_audit_receipt='/root/autodl-tmp/mcln_scanrefer_mesh_teacher_transfer_20260907_v1/receipt.json',teacher_audit_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
    real_train_rows=16,disposable_optimizer_steps_per_arm=2,purpose='Verify direct teacher geometry targets in existing final box heads,with original GT matching and unchanged semantic/Mask outputs.')
(local/'input_manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n',encoding='utf-8')
source=manifest['model_source']
controller='''#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
cd {remote}
flock -n /root/autodl-tmp/mcln_v99_backbone_gpu0.lock /root/miniconda3/envs/bdetr/bin/python -u scripts/probe_scanrefer_native_box_transfer.py --manifest {remote}/input_manifest.json
status=$?
printf '%s\\n' "$status" > controller.exit
exit "$status"
'''.format(remote=remote)
(local/'controller.sh').write_bytes(controller.encode())
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();s.mkdir(remote);s.mkdir(remote+'/scripts');s.mkdir(remote+'/tests')
for path in local.rglob('*'):
    if path.is_file():s.put(str(path),remote+'/'+path.relative_to(local).as_posix())
command='cd '+shlex.quote(source)+' && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH='+shlex.quote(remote+':'+source)+' /root/miniconda3/envs/bdetr/bin/python -m pytest -q '+shlex.quote(remote+'/tests/test_native_teacher_box_transfer.py')+' 2>&1'
_,o,e=c.exec_command(command);output=o.read().decode();(local/'cpu_tests.txt').write_text(output,encoding='utf-8');print(output,flush=True)
assert o.channel.recv_exit_status()==0
_,o,e=c.exec_command('nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits',timeout=30)
gpu=int(o.read().decode().strip());assert gpu<500
command='screen -L -Logfile '+shlex.quote(remote+'/run.log')+' -dmS mcln_native_box_transfer_probe bash '+shlex.quote(remote+'/controller.sh')
_,o,e=c.exec_command(command,timeout=30);assert o.channel.recv_exit_status()==0,e.read().decode()
_,o,e=c.exec_command("screen -ls\nps -eo pid,ppid,stat,etime,args | grep '[p]robe_scanrefer_native_box_transfer'",timeout=30)
live=o.read().decode();assert 'mcln_native_box_transfer_probe' in live
proof={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'screen':'mcln_native_box_transfer_probe','remote_directory':remote,'launch_command':command,'live':live,'estimated_runtime_seconds':180,'formal_rows':0,'checkpoint_writes':0}
(local/'launch.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8');(local/'deploy_from_local.py').write_bytes(Path(__file__).read_bytes())
s.close();c.close();print(json.dumps(proof),flush=True)
