import datetime, hashlib, json, os, pickle, shutil, subprocess
from pathlib import Path

r = Path(__file__).resolve().parent
acceptance = Path('/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1')
nr = Path('/root/autodl-tmp/mcln_eg3dvg_nr3d_transfer_20260920_v1')
source = acceptance / 'referit_input_source'
previous = Path('/root/autodl-tmp/mcln_eg3dvg_nr3d_adapt_20260920_v3')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8388608), b''): h.update(chunk)
    return h.hexdigest()

assert not (r/'spec.json').exists()
assert json.loads((nr/'formal/audit.json').read_text())['integrity_pass']
assert json.loads((previous/'preflight/receipt.json').read_text())['status'] == 'complete'
check = json.loads((acceptance/'referit_dataset_preflight.json').read_text())
assert check['status'] == 'complete' and check['datasets']['sr3d']['rows'] == 17726
assert sha(source/'src/joint_det_dataset.py') == check['dataset_source_sha256']
base = json.loads((nr/'spec.json').read_text())
env = json.loads((acceptance/'env_torch112.json').read_text())
env_sha = hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert env_sha == base['base_environment_sha256'] == '81a835e8144f7070d66feb0afa460911dec610052ced6b96c98097352cd022fd'
assert shutil.disk_usage(str(r)).free > 1000000000
cache = acceptance/'referit_input_cache/sr3d_annotations.pkl'
with cache.open('rb') as f: annotations=pickle.load(f)
assert len(annotations) == 17726 and len({a['scan_id'] for a in annotations}) == 255
(r/'annotation_manifest.json').write_text(json.dumps([{k:a[k] for k in ['scan_id','target_id','utterance']} for a in annotations]))
inputs={key:value for key,value in base['input_hashes'].items() if not key.endswith('nr3d_annotations.pkl') and not key.endswith('cls_results.json')}
inputs[str(cache)]=sha(cache)
inputs[str(source/'data/cls_results.json')]=sha(source/'data/cls_results.json')
spec=dict(base)
spec.update({'source':str(source),'source_files':{name:sha(source/name) for name in base['source_files']},
             'formal_rows':17726,'dataset':'sr3d','input_hashes':inputs,
             'evaluator_sha256':sha(r/'evaluate.py'),'auditor_sha256':sha(r/'audit.py'),
             'annotation_manifest_sha256':sha(r/'annotation_manifest.json'),
             'wait_for_nr_adaptation':str(previous),'protected_rec_hits25':12139,'protected_rec_hits50':10335})
(r/'spec.json').write_text(json.dumps(spec,indent=2))
runtime_env=dict(os.environ)
runtime_env.update({'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'})
with (r/'controller.log').open('xb') as log:
    proc=subprocess.Popen([spec['runtime'],'-u',str(r/'controller.py')],cwd=str(source),env=runtime_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
cpu_python='/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
runner="import subprocess,sys\nfrom pathlib import Path\nr=Path(__file__).resolve().parent\nwith (r/'train_inputs.log').open('xb') as f:code=subprocess.call([sys.executable,'-u',str(r/'prepare_train_inputs.py')],stdout=f,stderr=subprocess.STDOUT)\n(r/'train_inputs.exit').write_text(str(code)+'\\n')\nraise SystemExit(code)\n"
(r/'train_inputs_controller.py').write_text(runner)
cpu_env=dict(runtime_env);cpu_env['CUDA_VISIBLE_DEVICES']=''
cpu=subprocess.Popen([cpu_python,'-u',str(r/'train_inputs_controller.py')],cwd=str(source),env=cpu_env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
record={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'root':str(r),'pid':proc.pid,'cpu_input_pid':cpu.pid,'wait_for':str(previous/'controller.exit'),
        'spec_sha256':sha(r/'spec.json'),'dataset':'sr3d','formal_rows_planned':17726,
        'training_steps':0,'model_forwards_started':False,'shared_readonly_source':str(source),'environment_warm_reused':True}
(r/'launch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
