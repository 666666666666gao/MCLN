import datetime,hashlib,json,os,shlex
from pathlib import Path
import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
train='/root/autodl-tmp/mcln_scanrefer_mask_geometry_pair_20260907_v1'
prep='/root/autodl-tmp/mcln_scanrefer_mask_geometry_official_preparation_20260907_v1'
remote='/root/autodl-tmp/mcln_scanrefer_mask_geometry_posttraining_20260907_v1'
formal='/root/autodl-tmp/mcln_scanrefer_mask_geometry_official_20260907_v1'
local=repo/'refine-logs/scanrefer_mask_geometry_posttraining_20260907_v1'
localprep=repo/'refine-logs/scanrefer_mask_geometry_official_preparation_20260907_v1'
for p in [local,localprep]:p.mkdir(exist_ok=False);(p/'scripts').mkdir();(p/'tests').mkdir()
sha=lambda raw:hashlib.sha256(raw).hexdigest()
prep_names=['scripts/evaluate_scanrefer_mask_geometry_official.py','scripts/audit_scanrefer_mask_geometry_official.py',
    'scripts/audit_scanrefer_joint_readout_pair.py','scripts/scanrefer_data_contract.py',
    'scripts/scanrefer_joint_readout.py','scripts/scanrefer_rec_evaluation.py','tests/test_mask_geometry_promotion.py']
files={}
for name in prep_names:
    raw=(repo/name).read_bytes();(localprep/name).write_bytes(raw);files[name]=sha(raw)
(localprep/'scripts/__init__.py').write_bytes(b'');files['scripts/__init__.py']=sha(b'')
prepared={'schema':'mcln-mask-geometry-official-preparation-v1','files':files,
    'native_checkpoint_restore':'E71 plus exact84 allowed core tensors; actual native parameter names checked',
    'candidate_predeclared':'native_gt_mask_geometry_v99','control':'native_gt_v99','formal_rows_executed':0,
    'training_files_not_modified':True,'formal_manifest_not_yet_bound':'Requires actual2482 terminal,module pass and independent CPU audit'}
(localprep/'preparation.json').write_bytes((json.dumps(prepared,indent=2)+'\n').encode())
raw=(repo/'scripts/queue_scanrefer_mask_geometry_posttraining.py').read_bytes()
(local/'posttraining_queue.py').write_bytes(raw)
(local/'scripts/queue_scanrefer_mask_geometry_posttraining.py').write_bytes(raw)
(local/'scripts/__init__.py').write_bytes(b'')
for name in ['scripts/evaluate_scanrefer_mask_geometry_official.py','tests/test_mask_geometry_queue.py','tests/test_mask_geometry_promotion.py']:
    (local/name).write_bytes((repo/name).read_bytes())
train_local=repo/'refine-logs/scanrefer_mask_geometry_pair_20260907_v1'
train_manifest=json.loads((train_local/'input_manifest.json').read_bytes())
old=json.loads((repo/'refine-logs/scanrefer_frozen_readout_pair_20260907_v1/input_manifest.json').read_bytes())
assert old['data_root']==train_manifest['data_root'] and old['artifacts']==train_manifest['artifacts']
spec={'schema':'mcln-mask-geometry-gt-posttraining-queue-v1','training_directory':train,'formal_directory':formal,
    'formal_preparation_directory':prep,'training_manifest_sha256':sha((train_local/'input_manifest.json').read_bytes()),
    'formal_preparation_sha256':sha((localprep/'preparation.json').read_bytes()),'queue_script_sha256':sha(raw),
    'interval_seconds':240,'training_screen_pid':62966,'training_python_pid':62969,'candidate_predeclared':'native_gt_mask_geometry',
    'data_root':train_manifest['data_root'],'first_check_cst':'2026-09-07T19:10:00+08:00',
    'val_superpoint_files':old['val_superpoint_files'],'formal_only_after_fixed_module_screen':True,
    'training_controller_already_runs_independent_cpu_audit':True,'no_nr3d_sr3d_training_before_scanrefer_promotion':True}
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
s=c.open_sftp()
for localdir,remotedir in [(localprep,prep),(local,remote)]:
    s.mkdir(remotedir);s.mkdir(remotedir+'/scripts');s.mkdir(remotedir+'/tests')
    for p in localdir.rglob('*'):
        if p.is_file():s.put(str(p),remotedir+'/'+p.relative_to(localdir).as_posix())
_,o,e=c.exec_command('cd '+shlex.quote(remote)+' && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/bdetr/bin/python -m pytest -q tests',timeout=45)
tests=o.read().decode()+e.read().decode();assert o.channel.recv_exit_status()==0,tests
(local/'cpu_tests.txt').write_bytes(tests.encode());print(tests,flush=True)
check='''import ast,datetime,hashlib,json,subprocess
from pathlib import Path
base=Path(BASE);m=json.loads((base/'input_manifest.json').read_text())
assert hashlib.sha256((base/'posttraining_queue.py').read_bytes()).hexdigest()==m['queue_script_sha256']
for root in [base,Path(m['formal_preparation_directory'])]:
 for p in root.rglob('*.py'):ast.parse(p.read_text())
training=Path(m['training_directory']);prep=Path(m['formal_preparation_directory'])
assert hashlib.sha256((training/'input_manifest.json').read_bytes()).hexdigest()==m['training_manifest_sha256']
t=json.loads((training/'input_manifest.json').read_text())
for name,digest in t['files'].items():assert hashlib.sha256((training/name).read_bytes()).hexdigest()==digest,name
assert hashlib.sha256((prep/'preparation.json').read_bytes()).hexdigest()==m['formal_preparation_sha256']
prepared=json.loads((prep/'preparation.json').read_text())
for name,digest in prepared['files'].items():assert hashlib.sha256((prep/name).read_bytes()).hexdigest()==digest,name
for name,digest in m['val_superpoint_files'].items():assert hashlib.sha256((Path(m['data_root'])/'superpoints/val'/name).read_bytes()).hexdigest()==digest,name
p=subprocess.run(['ps','-p','62966,62969','-o','pid,ppid,comm,stat,etime,args'],stdout=subprocess.PIPE)
assert p.returncode==0 and b'62969' in p.stdout and b'run_scanrefer_mask_geometry_pair.py' in p.stdout
print(json.dumps({'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
'source_ast_and_bytes_verified':True,'val_superpoints_verified':312,'live_training':p.stdout.decode(),
'new_gpu_forwards':0,'training_files_unchanged':True,'queue_started':False}))
'''.replace('BASE',repr(remote))
_,o,e=c.exec_command('/root/miniconda3/envs/bdetr/bin/python -c '+shlex.quote(check)+' && bash -n '+shlex.quote(remote+'/controller.sh'),timeout=45)
body=o.read().decode();assert o.channel.recv_exit_status()==0,e.read().decode()
proof=json.loads(body);(local/'preparation.json').write_bytes((json.dumps(proof,indent=2)+'\n').encode())
(local/'source_input_check.py').write_bytes(check.encode());(local/'stage_from_local.py').write_bytes(Path(__file__).read_bytes())
s.close();c.close();print(json.dumps(proof),flush=True)
