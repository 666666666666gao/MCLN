import subprocess,json,sys
from pathlib import Path
r=Path('/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1'); e=Path('/root/mcln_eg3dvg_torch112_20260920_v1')
for cmd in [['df','-h','/','/tmp','/root/autodl-tmp'],['df','-i','/'],['du','-h','--max-depth=2',str(e)],['tail','-30',str(r/'torch112_install.log')]]:
 print(json.dumps(cmd),flush=True);subprocess.run(cmd)
import torch
print('OLD_TORCH',torch.__version__,torch.__file__)
print('NEW_CONTENTS', [str(p) for p in (e/'venv/lib/python3.7/site-packages').iterdir()])
