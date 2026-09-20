import os,subprocess,sys
from pathlib import Path
r=Path('/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1')
e=Path('/root/mcln_eg3dvg_torch112_20260920_v1')
import torch
assert torch.__file__.startswith(str(e/'venv')+'/'),torch.__file__
env=dict(os.environ,TMPDIR=str(e/'tmp'))
code=subprocess.call([sys.executable,'-m','pip','uninstall','-y','torch'],env=env)
assert code==0
code=subprocess.call([sys.executable,'-m','pip','install','--disable-pip-version-check','--progress-bar','off','--ignore-installed','torch==1.12.0+cu116','--no-deps','--no-cache-dir','--index-url','https://download.pytorch.org/whl/cu116'],env=env)
(r/'torch112_install.exit').write_text(str(code)+'\n')
if code:raise SystemExit(code)
with (r/'pointnet112_build.log').open('wb') as f:code=subprocess.call([sys.executable,str(r/'build_pointnet112.py')],env=env,stdout=f,stderr=subprocess.STDOUT)
raise SystemExit(code)
