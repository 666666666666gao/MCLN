import os,subprocess,time
from pathlib import Path
r=Path(__file__).resolve().parent
while not (r/'train_input_prep.exit').exists():time.sleep(180)
if (r/'train_input_prep.exit').read_text().strip()!='0':raise SystemExit('Training annotation preparation did not pass')
env=dict(os.environ);env.update({'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})
with (r/'train_dataset_preflight.log').open('xb') as f: code=subprocess.call(['/root/mcln_eg3dvg_torch112_20260920_v1/venv/bin/python','-u',str(r/'preflight_train_dataset.py')],cwd=str(r/'source'),env=env,stdout=f,stderr=subprocess.STDOUT)
(r/'train_dataset_preflight.exit').write_text(str(code)+'\n')
raise SystemExit(code)
