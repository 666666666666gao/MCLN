import datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8388608),b''):h.update(block)
 return h.hexdigest()
ins=json.loads((r/'checkpoint_inspection.json').read_text())
assert (r/'checkpoint_inspection.exit').read_text().strip()=='0'
assert not ins['shape_mismatches'] and not ins['unexpected']
assert ins['missing'] in [[],['text_encoder.embeddings.position_ids']],ins
assert json.loads((r/'data_receipt.json').read_text())['rows']==9508
source=r/'source';up=json.loads((r/'upstream_manifest.json').read_text())
source_files={p:sha(source/p) for p in up['files']}
assert all(source_files[p]==v for p,v in up['files'].items() if p!='src/joint_det_dataset.py')
data=Path('/root/autodl-tmp/DATA_ROOT_mcln_meshsp')
files=[data/'val_v3scans.pkl',data/'ScanRefer/ScanRefer_filtered_val.json',data/'ScanRefer/ScanRefer_filtered_val.txt',r/'annotations.pkl']
scenes=sorted({a['scan_id'] for a in json.loads((r/'annotation_manifest.json').read_text())})
assert len(scenes)==312
files+=[data/'superpoints/val'/(s+'_superpoint.pth') for s in scenes]
files+=[data/'group_free_pred_bboxes/group_free_pred_bboxes_val'/(s+'.npy') for s in scenes]
files+=sorted(p for p in (data/'roberta-base').iterdir() if p.is_file())
input_hashes={str(p):sha(p) for p in files}
spec={'source':str(source),'upstream_commit':up['commit'],'source_files':source_files,'checkpoint':str(r/'official_scanrefer.pth'),'checkpoint_sha256':ins['checkpoint_sha256'],'checkpoint_key_prefix':ins['key_prefix'],'nonpersistent_position_ids':bool(ins['missing']),'data_root':str(data)+'/', 'input_hashes':input_hashes,'seed':2027,'batch_size':8,'formal_rows':9508,'primary_mode':'bbs','mask_gate':False,'evaluator_sha256':sha(r/'evaluate.py'),'auditor_sha256':sha(r/'audit.py'),'runtime':'/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python','base_environment_sha256':'966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c','native_box_rule':'mean of regressed and mask-derived boxes','training_steps':0}
with (r/'spec.json').open('x') as f:json.dump(spec,f,indent=2)
env=dict(os.environ);env.update({'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'})
with (r/'controller.log').open('xb') as f:
 p=subprocess.Popen([spec['runtime'],'-u',str(r/'controller.py')],cwd=str(source),env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
launch={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'pid':p.pid,'controller':str(r/'controller.py'),'spec_sha256':sha(r/'spec.json'),'preflight_rows':8,'formal_rows_planned':9508,'training_steps':0}
(r/'launch.json').write_text(json.dumps(launch,indent=2));print(json.dumps(launch),flush=True)
