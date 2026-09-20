import os,subprocess,sys,time
from pathlib import Path
r=Path(__file__).resolve().parent
while not (r/'train_inputs.exit').exists():time.sleep(180)
assert (r/'train_inputs.exit').read_text().strip()=='0'
with (r/'train_dataset_preflight.log').open('xb') as f:code=subprocess.call([sys.executable,'-u',str(r/'preflight_train_dataset.py')],stdout=f,stderr=subprocess.STDOUT)
(r/'train_dataset_preflight.exit').write_text(str(code)+'\n')
raise SystemExit(code)
