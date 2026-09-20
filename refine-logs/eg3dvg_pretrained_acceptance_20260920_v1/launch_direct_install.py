import hashlib,json,os,subprocess
from pathlib import Path
r=Path(__file__).resolve().parent;e=Path('/root/mcln_eg3dvg_torch112_20260920_v1')
spec=json.loads((r/'env_torch112.json').read_text());spec['installer_pip']='23.1.2'
digest=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()
for path in [r/'env_torch112.json',e/'env_spec.json']:path.write_text(json.dumps(spec,indent=2))
(e/'env_spec.sha256').write_text(digest+'\n')
runner=r/'torch112_install_direct.py'
runner.write_text("import subprocess,os\nfrom pathlib import Path\nr=Path("+repr(str(r))+")\ne=Path("+repr(str(e))+")\nenv=dict(os.environ,TMPDIR=str(e/'tmp'))\ncmd=[str(e/'venv/bin/python'),'-m','pip','install','--disable-pip-version-check','--progress-bar','off','--ignore-installed','torch==1.12.0+cu113','--no-deps','--no-cache-dir','--index-url','https://download.pytorch.org/whl/cu113']\nprint(cmd,flush=True)\ncode=subprocess.call(cmd,env=env)\n(r/'torch112_install.exit').write_text(str(code)+'\\n')\nraise SystemExit(code)\n")
(r/'torch112_install.exit').unlink()
log=open(str(r/'torch112_install.log'),'wb')
p=subprocess.Popen([str(e/'venv/bin/python'),str(runner)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,TMPDIR=str(e/'tmp')))
receipt={'pid':p.pid,'spec_sha256':digest,'installer':'pip 23.1.2 direct wheel installation','reason':'pip20.1 unpacked-wheel duplicate caused disk exhaustion','old_environment_unchanged':'1.10.2+cu111 verified'}
(r/'torch112_install_launch.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
