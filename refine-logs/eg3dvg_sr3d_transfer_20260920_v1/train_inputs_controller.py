import subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent
with (r/'train_inputs.log').open('xb') as f:code=subprocess.call([sys.executable,'-u',str(r/'prepare_train_inputs.py')],stdout=f,stderr=subprocess.STDOUT)
(r/'train_inputs.exit').write_text(str(code)+'\n')
raise SystemExit(code)
