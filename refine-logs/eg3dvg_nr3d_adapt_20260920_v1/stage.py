import datetime,hashlib,json,os,shutil,subprocess
from pathlib import Path
import numpy as np
r=Path(__file__).resolve().parent;old=Path('/root/autodl-tmp/mcln_eg3dvg_nr3d_transfer_20260920_v1');source=r/'source'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8388608),b''):h.update(block)
 return h.hexdigest()
assert not (r/'spec.json').exists()
fix=json.loads((r/'view_fix.json').read_text());assert fix['status']=='pass' and fix['rotate_allowed_to_blocked']==325 and fix['blocked_to_allowed']==0
assert sha(source/'src/joint_det_dataset.py')==fix['after_sha256']
base=json.loads((old/'spec.json').read_text());assert base['base_environment_sha256']=='81a835e8144f7070d66feb0afa460911dec610052ced6b96c98097352cd022fd'
state_root=Path('/root/mcln_eg3dvg_nr3d_adapt_states_20260920_v1');assert shutil.disk_usage('/root').free>4000000000;state_root.mkdir()
order=np.random.RandomState(2027).permutation(44909);np.save(str(r/'training_order.npy'),order)
input_hashes={name:value for name,value in base['input_hashes'].items() if '/superpoints/val/' not in name and not name.endswith('val_v3scans.pkl')}
data=Path(base['data_root']);input_hashes[str(data/'train_v3scans.pkl')]=sha(data/'train_v3scans.pkl')
for p in sorted((data/'superpoints/train').glob('*_superpoint.pth')):input_hashes[str(p)]=sha(p)
for name in ['nr3d','scannet']:
 p=old/'train_input_cache'/(name+'_annotations.pkl');input_hashes[str(p)]=sha(p)
input_hashes[str(r/'training_order.npy')]=sha(r/'training_order.npy')
spec={'source':str(source),'source_files':{name:sha(source/name) for name in base['source_files']},'checkpoint':base['checkpoint'],'checkpoint_sha256':base['checkpoint_sha256'],'data_root':base['data_root'],'input_hashes':input_hashes,'seed':2027,'batch_size':8,'fit_rows':44909,'fit_updates':5614,'preflight_indices':[0,4702,10973,16459,21946,24689,32917,32918,163,165,238,356,32919,33518,34117,34118],'order_path':str(r/'training_order.npy'),'lr':1e-4,'lr_backbone':1e-5,'text_encoder_lr':1e-5,'schedule':'constant one complete pass; no validation selection','clip_norm':.1,'weight_decay':.0005,'trainer_sha256':sha(r/'train.py'),'controller_sha256':sha(r/'controller.py'),'post_evaluator_sha256':sha(r/'evaluate_adapted.py'),'post_auditor_sha256':sha(r/'audit_adapted.py'),'state_root':str(state_root),'transfer_root':str(old),'environment_sha256':base['base_environment_sha256'],'runtime':base['runtime'],'upstream_commit':base['upstream_commit'],'scope':'native EG adaptation control; train-only view-word normalization; no new network module','mask_acceptance_gate':False}
(r/'spec.json').write_text(json.dumps(spec,indent=2))
env=dict(os.environ);env.update({'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'})
with (r/'controller.log').open('xb') as f:proc=subprocess.Popen([spec['runtime'],'-u',str(r/'controller.py')],cwd=str(source),env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
launch={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'pid':proc.pid,'waiting_for':str(old/'controller.exit'),'spec_sha256':sha(r/'spec.json'),'fit_updates_planned':5614,'fit_rows_planned':44909,'preflight_optimizer_steps_planned':2,'training_started':False}
(r/'launch.json').write_text(json.dumps(launch,indent=2));print(json.dumps(launch),flush=True)
