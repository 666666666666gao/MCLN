import datetime,hashlib,json,os,shutil,subprocess
from pathlib import Path
import numpy as np

r=Path(__file__).resolve().parent
sr=Path('/root/autodl-tmp/mcln_eg3dvg_sr3d_transfer_20260920_v1')
nr=Path('/root/autodl-tmp/mcln_eg3dvg_nr3d_adapt_20260920_v3')
acceptance=Path('/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8388608),b''):h.update(chunk)
    return h.hexdigest()

assert not (r/'spec.json').exists()
pre=json.loads((sr/'train_dataset_preflight.json').read_text())
assert pre['status']=='complete' and pre['rows']==77836 and pre['sr3d_rows']==65846
assert (sr/'train_dataset_preflight.exit').read_text().strip()=='0'
source=r/'source'
shutil.copytree(str(sr/'train_input_source'),str(source),ignore=shutil.ignore_patterns('__pycache__'))
assert sha(source/'src/joint_det_dataset.py')==pre['source_sha256']
model=source/'models/eg.py';before=model.read_text()
marker='        # STEP 5. Query Points Generation\n'
assert before.count(marker)==1 and "end_points['super_xyz_list']" not in before
model.write_text(before.replace(marker,"        end_points['super_xyz_list'] = super_xyz_list\n\n"+marker))
assert sha(model)==json.loads((nr/'spec.json').read_text())['source_files']['models/eg.py']
base=json.loads((nr/'spec.json').read_text())
eval_base=json.loads((sr/'spec.json').read_text())
env=json.loads((acceptance/'env_torch112.json').read_text())
env_sha=hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert env_sha==base['environment_sha256']=='81a835e8144f7070d66feb0afa460911dec610052ced6b96c98097352cd022fd'
assert base['checkpoint_sha256']==eval_base['checkpoint_sha256']
assert shutil.disk_usage('/root').free>3000000000
assert shutil.disk_usage(str(r)).free>1000000000
state=Path('/root/mcln_eg3dvg_sr3d_adapt_states_20260920_v1');state.mkdir()
order=np.random.RandomState(2027).permutation(77836)
np.save(str(r/'training_order.npy'),order)
inputs={key:value for key,value in base['input_hashes'].items() if not key.endswith('nr3d_annotations.pkl') and not key.endswith('training_order.npy')}
for path in [sr/'train_input_cache/sr3d_annotations.pkl',r/'training_order.npy',source/'data/meta_data/sr3d_train_scans.txt']:
    inputs[str(path)]=sha(path)
spec=dict(base)
for key in ['replaces_failed_preflight','repair']:spec.pop(key)
spec.update({'source':str(source),'source_files':{name:sha(source/name) for name in eval_base['source_files']},
             'input_hashes':inputs,'fit_rows':77836,'fit_updates':9730,
             'preflight_indices':[0,9406,21948,32923,43897,49384,65844,65845,100,1000,10000,30000,65846,66445,67044,67045],
             'order_path':str(r/'training_order.npy'),'trainer_sha256':sha(r/'train.py'),'controller_sha256':sha(r/'controller.py'),
             'post_evaluator_sha256':sha(r/'evaluate_adapted.py'),'post_auditor_sha256':sha(r/'audit_adapted.py'),
             'paired_analysis_sha256':sha(r/'analyze_pair.py'),'state_root':str(state),'transfer_root':str(sr),
             'nr_adaptation_root':str(nr),'required_nr_hits25':4726,'required_nr_hits50':4059,
             'protected_sr_hits25':12139,'protected_sr_hits50':10335,
             'scope':'Conditional native Sr adaptation control; author augmentation unchanged; no new network module',
             'entry_rule':'First require Nr endpoint to exceed baseline at both thresholds; then skip if Sr zero-update already meets protected counts; otherwise one fixed pass.',
             'initialization_dataset':'scanrefer','checkpoint_choice':'original author ScanRefer epoch69 regardless of Nr weights',
             'repair':'Only export existing super_xyz_list required by author loss; no forward numeric or loss change.'})
(r/'spec.json').write_text(json.dumps(spec,indent=2))
runtime_env=dict(os.environ)
runtime_env.update({'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'})
with (r/'controller.log').open('xb') as log:
    process=subprocess.Popen([spec['runtime'],'-u',str(r/'controller.py')],cwd=str(source),env=runtime_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'pid':process.pid,'root':str(r),'spec_sha256':sha(r/'spec.json'),'fit_updates_if_needed':9730,
        'fit_rows_if_needed':77836,'training_started':False,'gpu_preflight_started':False,
        'source_dataset_sha256':sha(source/'src/joint_det_dataset.py'),'source_model_sha256':sha(model),
        'environment_warm_reused':True,'state':'waiting_for_fixed_entry_conditions'}
(r/'launch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
