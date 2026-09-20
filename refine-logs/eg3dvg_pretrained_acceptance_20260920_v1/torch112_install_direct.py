import subprocess,os
from pathlib import Path
r=Path('/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1')
e=Path('/root/mcln_eg3dvg_torch112_20260920_v1')
env=dict(os.environ,TMPDIR=str(e/'tmp'))
cmd=[str(e/'venv/bin/python'),'-m','pip','install','--disable-pip-version-check','--progress-bar','off','--ignore-installed','torch==1.12.0+cu113','--no-deps','--no-cache-dir','--index-url','https://download.pytorch.org/whl/cu113']
print(cmd,flush=True)
code=subprocess.call(cmd,env=env)
(r/'torch112_install.exit').write_text(str(code)+'\n')
raise SystemExit(code)
