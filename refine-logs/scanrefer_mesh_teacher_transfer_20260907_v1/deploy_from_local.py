import datetime,hashlib,json,os,shlex
from pathlib import Path
import paramiko
repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
local=repo/'refine-logs/scanrefer_mesh_teacher_transfer_20260907_v1'
local.mkdir(exist_ok=False);(local/'scripts').mkdir()
remote='/root/autodl-tmp/mcln_scanrefer_mesh_teacher_transfer_20260907_v1'
base=json.loads((repo/'refine-logs/scanrefer_frozen_readout_probe_20260907_v1/input_manifest.json').read_bytes())
old_protocol=json.loads((repo/'refine-logs/scanrefer_teacher_transfer_20260906_v4/protocol.json').read_bytes())
files={}
for name in ['audit_scanrefer_mesh_teacher_transfer.py','scanrefer_data_contract.py','scanrefer_joint_readout.py','scanrefer_rec_evaluation.py']:
    raw=(repo/'scripts'/name).read_bytes();(local/'scripts'/name).write_bytes(raw)
    files['scripts/'+name]=hashlib.sha256(raw).hexdigest()
manifest={k:base[k] for k in ['model_source','source_manifest_sha256','artifacts','data_root','train_superpoint_files','split_salt','environment_reuse']}
manifest.update(schema='mcln-scanrefer-mesh-teacher-transfer-input-v1',rows=512,optimizer_steps=0,formal_rows=0,checkpoint_writes=0,
    files=files,selected_row_ids=old_protocol['selected_row_ids'],
    purpose='Rebuild teacher Query/Variant transfer evidence on correct mesh inputs;fixed historical512 fit identities;no training or formal evaluation.',
    historical_protocol_sha256=hashlib.sha256((repo/'refine-logs/scanrefer_teacher_transfer_20260906_v4/protocol.json').read_bytes()).hexdigest())
(local/'input_manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n',encoding='utf-8')
controller='''#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
cd {remote}
flock -n /root/autodl-tmp/mcln_v99_backbone_gpu0.lock /root/miniconda3/envs/bdetr/bin/python -u scripts/audit_scanrefer_mesh_teacher_transfer.py --manifest {remote}/input_manifest.json
status=$?
printf '%s\\n' "$status" > controller.exit
exit "$status"
'''.format(remote=remote)
(local/'controller.sh').write_bytes(controller.encode())
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();s.mkdir(remote);s.mkdir(remote+'/scripts')
for path in local.rglob('*'):
    if path.is_file():s.put(str(path),remote+'/'+path.relative_to(local).as_posix())
probe="import ast,json,os,subprocess;from pathlib import Path;p=Path("+repr(remote)+");m=json.loads((p/'input_manifest.json').read_text());[ast.parse((p/f).read_text()) for f in m['files']];gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits']).decode().strip();assert int(gpu)<500;free=os.statvfs('/root/autodl-tmp').f_bavail*os.statvfs('/root/autodl-tmp').f_frsize;assert free>2*1024**3;print(json.dumps({'syntax_files':len(m['files']),'gpu_used_mib':int(gpu),'free_bytes':free}))"
_,o,e=c.exec_command('/root/miniconda3/envs/bdetr/bin/python -c '+shlex.quote(probe),timeout=30)
raw=o.read();assert o.channel.recv_exit_status()==0,e.read().decode();(local/'launch_precheck.json').write_bytes(raw)
command='screen -L -Logfile '+shlex.quote(remote+'/run.log')+' -dmS mcln_mesh_teacher_audit bash '+shlex.quote(remote+'/controller.sh')
_,o,e=c.exec_command(command,timeout=30);assert o.channel.recv_exit_status()==0,e.read().decode()
_,o,e=c.exec_command("screen -ls\nps -eo pid,ppid,stat,etime,args | grep '[a]udit_scanrefer_mesh_teacher_transfer'",timeout=30)
live=o.read().decode();assert 'mcln_mesh_teacher_audit' in live
proof={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'remote_directory':remote,'screen':'mcln_mesh_teacher_audit','launch_command':command,'live':live,'precheck':json.loads(raw),'estimated_runtime_seconds':180,'optimizer_steps':0,'formal_rows':0}
(local/'launch.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
(local/'deploy_from_local.py').write_bytes(Path(__file__).read_bytes())
s.close();c.close();print(json.dumps(proof),flush=True)
