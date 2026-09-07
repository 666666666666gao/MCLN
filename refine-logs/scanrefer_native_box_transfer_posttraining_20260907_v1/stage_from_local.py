import datetime,hashlib,json,os,shlex
from pathlib import Path
import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
local=repo/'refine-logs/scanrefer_native_box_transfer_posttraining_20260907_v1'
remote='/root/autodl-tmp/mcln_scanrefer_native_box_transfer_posttraining_20260907_v1'
train='/root/autodl-tmp/mcln_scanrefer_native_box_transfer_pair_20260907_v1'
prep='/root/autodl-tmp/mcln_scanrefer_native_box_transfer_official_preparation_20260907_v1'
formal='/root/autodl-tmp/mcln_scanrefer_native_box_transfer_official_20260907_v1'
local.mkdir(exist_ok=False);(local/'scripts').mkdir();(local/'tests').mkdir()
sha=lambda raw:hashlib.sha256(raw).hexdigest()
raw=(repo/'scripts/queue_scanrefer_native_box_transfer_posttraining.py').read_bytes()
(local/'posttraining_queue.py').write_bytes(raw)
(local/'scripts/queue_scanrefer_native_box_transfer_posttraining.py').write_bytes(raw)
(local/'scripts/__init__.py').write_bytes(b'')
for name in ['scripts/evaluate_scanrefer_native_box_transfer_official.py','tests/test_native_box_transfer_queue.py','tests/test_native_box_transfer_promotion.py']:
 (local/name).write_bytes((repo/name).read_bytes())
train_manifest=json.loads((repo/'refine-logs/scanrefer_native_box_transfer_pair_20260907_v1/input_manifest.json').read_bytes())
old=json.loads((repo/'refine-logs/scanrefer_frozen_readout_pair_20260907_v1/input_manifest.json').read_bytes())
assert old['data_root']==train_manifest['data_root']
assert old['train_superpoint_files']==train_manifest['train_superpoint_files']
assert old['artifacts']==train_manifest['artifacts']
spec={'schema':'mcln-native-box-transfer-posttraining-queue-v1','training_directory':train,
 'formal_directory':formal,'formal_preparation_directory':prep,
 'training_manifest_sha256':sha((repo/'refine-logs/scanrefer_native_box_transfer_pair_20260907_v1/input_manifest.json').read_bytes()),
 'formal_preparation_sha256':sha((repo/'refine-logs/scanrefer_native_box_transfer_official_preparation_20260907_v1/preparation.json').read_bytes()),
 'queue_script_sha256':sha(raw),'interval_seconds':240,'training_screen_pid':58020,'training_python_pid':58023,
 'candidate_predeclared':'gt_teacher_box','data_root':train_manifest['data_root'],
 'first_check_cst':'2026-09-07T13:40:00+08:00','val_superpoint_files':old['val_superpoint_files'],
 'formal_only_after_fixed_module_screen':True,'training_controller_already_runs_independent_cpu_audit':True,
 'do_not_rerun_training_audit':True,'no_nr3d_sr3d_training_before_scanrefer_promotion':True}
(local/'input_manifest.json').write_bytes((json.dumps(spec,sort_keys=True,indent=2)+'\n').encode())
controller='''#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
cd {remote}
/root/miniconda3/envs/bdetr/bin/python -u posttraining_queue.py --manifest {remote}/input_manifest.json
status=$?
printf '%s\\n' "$status" > controller.exit
exit "$status"
'''.format(remote=remote)
(local/'controller.sh').write_bytes(controller.encode())
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();s.mkdir(remote);s.mkdir(remote+'/scripts');s.mkdir(remote+'/tests')
for p in local.rglob('*'):
 if p.is_file():s.put(str(p),remote+'/'+p.relative_to(local).as_posix())
_,o,e=c.exec_command('cd '+shlex.quote(remote)+' && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/bdetr/bin/python -m pytest -q tests',timeout=45)
tests=o.read().decode()+e.read().decode();assert o.channel.recv_exit_status()==0,tests
(local/'cpu_tests.txt').write_bytes(tests.encode());print(tests,flush=True)
check='''import ast,datetime,hashlib,json,subprocess
from pathlib import Path
base=Path(BASE)
m=json.loads((base/'input_manifest.json').read_text())
assert hashlib.sha256((base/'posttraining_queue.py').read_bytes()).hexdigest()==m['queue_script_sha256']
for p in base.rglob('*.py'):ast.parse(p.read_text())
training=Path(m['training_directory']);prep=Path(m['formal_preparation_directory'])
assert hashlib.sha256((training/'input_manifest.json').read_bytes()).hexdigest()==m['training_manifest_sha256']
assert hashlib.sha256((prep/'preparation.json').read_bytes()).hexdigest()==m['formal_preparation_sha256']
prepared=json.loads((prep/'preparation.json').read_text())
for name,digest in prepared['files'].items():assert hashlib.sha256((prep/name).read_bytes()).hexdigest()==digest,name
point_dir=Path(m['data_root'])/'superpoints/val'
assert len(m['val_superpoint_files'])==312
for name,digest in m['val_superpoint_files'].items():assert hashlib.sha256((point_dir/name).read_bytes()).hexdigest()==digest,name
p=subprocess.run(['ps','-p','58020,58023','-o','pid,ppid,comm,stat,etime,args'],stdout=subprocess.PIPE)
assert p.returncode==0 and b'58023' in p.stdout and b'run_scanrefer_native_box_transfer_pair.py' in p.stdout
print(json.dumps({'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
'original_python_ast_and_imports':True,'queue_and_formal_source_bytes_verified':True,
'val_superpoints_verified':312,'live_training_processes':p.stdout.decode(),
'new_gpu_forwards':0,'training_manifest_unchanged':True,'queue_started':False}))
'''.replace('BASE',repr(remote))
_,o,e=c.exec_command('/root/miniconda3/envs/bdetr/bin/python -c '+shlex.quote(check)+' && bash -n '+shlex.quote(remote+'/controller.sh'),timeout=45)
body=o.read().decode();err=e.read().decode();assert o.channel.recv_exit_status()==0,err
proof=json.loads(body);(local/'preparation.json').write_bytes((json.dumps(proof,indent=2)+'\n').encode())
(local/'source_input_check.py').write_bytes(check.encode())
(local/'stage_from_local.py').write_bytes(Path(__file__).read_bytes())
for name in ['preparation.json','cpu_tests.txt','source_input_check.py']:
 s.put(str(local/name),remote+'/'+name)
_,o,e=c.exec_command('tail -n 7 '+train+'/run.log',timeout=30)
print(o.read().decode(),flush=True)
s.close();c.close();print(json.dumps(proof),flush=True)
