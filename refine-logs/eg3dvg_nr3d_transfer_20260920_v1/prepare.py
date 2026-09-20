import datetime,hashlib,json,os,pickle,shutil,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent
parent=Path('/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8388608),b''):h.update(block)
 return h.hexdigest()
old=json.loads((parent/'spec.json').read_text());terminal=json.loads((parent/'formal/audit.json').read_text())
assert terminal['integrity_pass'] and terminal['rows']==9508
assert (parent/'controller.exit').read_text().strip()=='0'
cpupre=json.loads((parent/'referit_dataset_preflight.json').read_text());assert cpupre['status']=='complete'
envspec=json.loads((parent/'env_torch112.json').read_text())
envhash=hashlib.sha256(json.dumps(envspec,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert envhash==old['base_environment_sha256']=='81a835e8144f7070d66feb0afa460911dec610052ced6b96c98097352cd022fd'
assert json.loads((parent/'independent_torch112_witness.json').read_text())['env_spec_sha256']==envhash
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
assert shutil.disk_usage(str(r)).free>500000000
source=r/'source';assert not source.exists()
shutil.copytree(str(parent/'referit_input_source'),str(source),ignore=shutil.ignore_patterns('__pycache__'))
assert sha(source/'src/joint_det_dataset.py')==cpupre['dataset_source_sha256']
for name,digest in old['source_files'].items():
 if name!='src/joint_det_dataset.py':assert sha(source/name)==digest,name
cache=parent/'referit_input_cache/nr3d_annotations.pkl'
with cache.open('rb') as f:annos=pickle.load(f)
assert len(annos)==7899 and len({a['scan_id'] for a in annos})==130
manifest=[{k:a[k] for k in ['scan_id','target_id','utterance']} for a in annos]
(r/'annotation_manifest.json').write_text(json.dumps(manifest))
inputs={name:digest for name,digest in old['input_hashes'].items() if 'group_free_pred_bboxes' not in name and '/ScanRefer/' not in name and name!=str(parent/'annotations.pkl')}
inputs[str(cache)]=sha(cache);inputs[str(source/'data/cls_results.json')]=sha(source/'data/cls_results.json')
spec=dict(old);spec.update({'source':str(source),'source_files':{name:sha(source/name) for name in old['source_files']},'formal_rows':7899,'dataset':'nr3d','initialization_dataset':'scanrefer','experiment_type':'cross_dataset_transfer','primary_mode':'bbs','input_hashes':inputs,'evaluator_sha256':sha(r/'evaluate.py'),'auditor_sha256':sha(r/'audit.py'),'annotation_manifest_sha256':sha(r/'annotation_manifest.json'),'object_input':'GT scene instance boxes with predicted class ids','primary_filter':'score multiplied by IoU(native_box, any scene object)>0.25','unfiltered_modes':'same-forward diagnostic only','no_checkpoint_selection':True})
with (r/'spec.json').open('x') as f:json.dump(spec,f,indent=2)
env=dict(os.environ);env.update({'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'})
with (r/'controller.log').open('xb') as f:
 proc=subprocess.Popen([spec['runtime'],'-u',str(r/'controller.py')],cwd=str(source),env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
launch={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'pid':proc.pid,'root':str(r),'spec_sha256':sha(r/'spec.json'),'dataset':'nr3d','formal_rows_planned':7899,'training_steps':0,'environment_warm_reused':True}
(r/'launch.json').write_text(json.dumps(launch,indent=2));print(json.dumps(launch),flush=True)
