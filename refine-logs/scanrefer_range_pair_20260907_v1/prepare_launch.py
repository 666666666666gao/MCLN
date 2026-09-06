import datetime
import hashlib
import json
import os
from pathlib import Path

import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
preflight=repo/'refine-logs/scanrefer_range_preflight_20260907_v1'
local=repo/'refine-logs/scanrefer_range_pair_20260907_v1'
remote='/root/autodl-tmp/mcln_scanrefer_range_pair_20260907_v1'
remote_probe='/root/autodl-tmp/mcln_scanrefer_range_preflight_20260907_v1'
assert not local.exists()
old=json.loads((repo/'refine-logs/scanrefer_local_visual_mesh_pair_20260906_v1/input_manifest.json').read_bytes())
probe=json.loads((preflight/'receipt.json').read_bytes())
assert (preflight/'controller.exit').read_text().strip()=='0'
assert probe['status']=='pass' and probe['formal_rows']==probe['checkpoint_writes']==0
assert hashlib.sha256((preflight/'coverage_rows.json').read_bytes()).hexdigest()==probe['coverage_rows_sha256']
assert probe['manifest_sha256']==hashlib.sha256((preflight/'input_manifest.json').read_bytes()).hexdigest()
for a,b in zip(probe['observations'][::2],probe['observations'][1::2]):
    assert a['point_sha256']==b['point_sha256'] and a['loss']==b['loss']
    assert a['zero_start_native_and_v99_parity'] and b['zero_start_native_and_v99_parity']
local.mkdir()
names=['models/candidate_range_visual.py','models/candidate_local_visual.py',
    'scripts/run_scanrefer_range_pair.py','scripts/audit_scanrefer_range_pair.py',
    'scripts/evaluate_scanrefer_range_official.py','scripts/audit_scanrefer_range_official.py',
    'scripts/audit_scanrefer_joint_readout_pair.py','scripts/scanrefer_joint_readout.py',
    'scripts/scanrefer_data_contract.py','scripts/scanrefer_rec_evaluation.py']
files={}
for name in names:
    raw=(repo/name).read_bytes()
    target=local/name
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(raw)
    files[name]=hashlib.sha256(raw).hexdigest()
(local/'scripts/__init__.py').write_bytes(b'')
files['scripts/__init__.py']=hashlib.sha256(b'').hexdigest()
split=(repo/'refine-logs/scanrefer_local_visual_mesh_pair_20260906_v1/split_protocol.json').read_bytes()
assert hashlib.sha256(split).hexdigest()==old['split_protocol_sha256']
(local/'split_protocol.json').write_bytes(split)
plan={'schema':'mcln-scanrefer-range-plan-v1','question':'Does distributed spatial evidence improve REC compared with center sampling under the same regional readout?',
    'arms':{'control':'center nearest64 within common RoI, grouped by octants','local':'extent up to8 unique points per octant,64 slots'},
    'reader_per_arm':{'trainable_parameters':145008,'regions':8,'point_budget_slots':64,'window_half_extent_multiplier':1.5,
        'minimum_half_extent_m':.05,'readout':'point attention within region then query-conditioned region attention'},
    'pretrained_start':'protected E71;identical fresh zero-output region modules,not failed endpoints',
    'fit_rows':29778,'module_holdout_rows':6887,'module_holdout_backbone_seen':True,
    'steps_per_arm':2482,'epochs':1,'batch_size':12,'core_learning_rate':1e-6,'reader_learning_rate':1e-4,
    'loss':'native_gt_only','frozen_readouts':['Parent','Geometry','V99'],'checkpoint_policy':'one final endpoint per arm',
    'formal_evaluation':'one fixed9508-row triplet:protected_v99,center_v99,local_v99;native and full-system REC plus Mask',
    'candidate_predeclared':'local_v99 extent;no validation arm or epoch selection',
    'promotion':{'historical_scan_rec_hits':[5572,4797],'also_no_lower_than_same_run_protected':True,
        'scan_mask_paper_percent':[58.70,50.70,44.72],'nr3d_sr3d_mask_gate':False,
        'stretch59_51_not_required':True,'advance_to_nr_sr_when_protected_gates_pass':True},
    'exclusions':['no new Gate','no quality distribution','no BCT loss','no parser change','no validation-specific rules'],
    'preflight_receipt_sha256':hashlib.sha256((preflight/'receipt.json').read_bytes()).hexdigest()}
(local/'experiment_plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n',encoding='utf-8')
manifest={key:old[key] for key in ['artifacts','batch_size','clip_norm','core_learning_rate','data_root','epochs','local_learning_rate',
    'loss','model_source','readouts_frozen','source_manifest_sha256','split_protocol_sha256','split_salt','steps_per_arm','superpoint_files','weight_decay']}
manifest.update({'schema':'mcln-scanrefer-range-pair-input-v1','run_directory':remote,'mode':'train','formal_rows':0,'files':files,
    'split_protocol':remote+'/split_protocol.json','native_probe_receipt':remote_probe+'/receipt.json',
    'native_probe_receipt_sha256':plan['preflight_receipt_sha256'],'native_probe_manifest':remote_probe+'/input_manifest.json',
    'plan_sha256':hashlib.sha256((local/'experiment_plan.json').read_bytes()).hexdigest(),
    'arm_meanings':plan['arms'],'formal_candidate_predeclared':'local_v99','formal_include_center_control':True})
(local/'input_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')
controller=("#!/usr/bin/env bash\nset -u\nexport CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1\ncd '"+remote+"'\nflock -x /root/autodl-tmp/mcln_v99_backbone_gpu0.lock /root/miniconda3/envs/bdetr/bin/python -u scripts/run_scanrefer_range_pair.py --manifest '"+remote+"/input_manifest.json'\nstatus=$?\nprintf '%s\\n' \"$status\" > controller.exit\nexit \"$status\"\n").encode()
(local/'controller.sh').write_bytes(controller)
client=paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
sftp.mkdir(remote)
for name in ['models','scripts']: sftp.mkdir(remote+'/'+name)
for name in list(files)+['split_protocol.json','experiment_plan.json','input_manifest.json','controller.sh']:
    raw=(local/name).read_bytes()
    with sftp.open(remote+'/'+name,'wx') as stream: stream.write(raw)
    with sftp.open(remote+'/'+name,'rb') as stream: assert stream.read()==raw
code="""import ast,datetime,hashlib,json,shutil,subprocess
from pathlib import Path
root=Path('/root/autodl-tmp/mcln_scanrefer_range_pair_20260907_v1')
manifest=json.loads((root/'input_manifest.json').read_text())
assert not (root/'run.log').exists()
for name,digest in manifest['files'].items():
 raw=(root/name).read_bytes()
 assert hashlib.sha256(raw).hexdigest()==digest,name
 if name.endswith('.py'): ast.parse(raw)
assert hashlib.sha256(Path(manifest['native_probe_receipt']).read_bytes()).hexdigest()==manifest['native_probe_receipt_sha256']
assert json.loads(Path(manifest['native_probe_receipt']).read_text())['status']=='pass'
disk=shutil.disk_usage('/root/autodl-tmp')
assert disk.free>3*1024**3,disk
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader']).decode().strip()
assert not gpu,gpu
session='mcln_scanrefer_range_pair_v1'
subprocess.run(['screen','-dmS',session,'bash','-c','exec bash '+str(root/'controller.sh')+' > '+str(root/'run.log')+' 2>&1'],check=True)
listing=subprocess.check_output(['screen','-ls']).decode()
sessions=[line.split()[0] for line in listing.splitlines() if line.split() and line.split()[0].endswith('.'+session)]
assert len(sessions)==1,sessions
value={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'screen_session':sessions,'manifest_sha256':hashlib.sha256((root/'input_manifest.json').read_bytes()).hexdigest(),'plan_sha256':manifest['plan_sha256'],'steps_per_arm':2482,'formal_rows':0,'disk_before':disk._asdict(),'started':True,'preflight_passed':True,'controller_sha256':hashlib.sha256((root/'controller.sh').read_bytes()).hexdigest()}
with (root/'launch.json').open('x') as stream: json.dump(value,stream,indent=2,sort_keys=True)
print(json.dumps(value))
"""
_,out,err=client.exec_command('/root/miniconda3/envs/bdetr/bin/python - <<\'PY\'\n'+code+'\nPY',timeout=30)
raw=out.read()
assert out.channel.recv_exit_status()==0,err.read().decode()
value=json.loads(raw)
with sftp.open(remote+'/launch.json','rb') as stream: (local/'launch.json').write_bytes(stream.read())
sftp.close()
client.close()
print(json.dumps(value))
