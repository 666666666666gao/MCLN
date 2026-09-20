import subprocess,os,json
from pathlib import Path
r=Path(__file__).resolve().parent;e=Path('/root/mcln_eg3dvg_torch112_20260920_v1');env=dict(os.environ,TMPDIR=str(e/'tmp'))
(r/'torch112_install_first.exit').write_text((r/'torch112_install.exit').read_text())
(r/'torch112_install_first.log').write_bytes((r/'torch112_install.log').read_bytes())
cmd=[str(e/'venv/bin/python'),'-m','pip','install','--disable-pip-version-check','--no-cache-dir','--no-deps','pip==23.1.2','--index-url','https://pypi.org/simple']
subprocess.run(cmd,env=env,check=True)
subprocess.run([str(e/'venv/bin/python'),'-c','import pip,inspect; from pip._internal.operations.install.wheel import install_wheel; print(pip.__version__); print(inspect.getsource(install_wheel))'],env=env,check=True)
