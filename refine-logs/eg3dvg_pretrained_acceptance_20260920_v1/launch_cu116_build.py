import hashlib,json,os,subprocess
from pathlib import Path
r=Path(__file__).resolve().parent;e=Path('/root/mcln_eg3dvg_torch112_20260920_v1')
for n in ['env_torch112.json','torch112_install.log','torch112_install.exit','pointnet112_build.log','pointnet112_build.exit']:
 (r/(n+'.cu113')).write_bytes((r/n).read_bytes())
spec=json.loads((r/'env_torch112.json').read_text());spec['pip_phases']=[[v.replace('cu113','cu116') for v in phase] for phase in spec['pip_phases']];spec['required_versions']['torch']='1.12.0+cu116';spec['reason']+=' CUDA build aligned to installed nvcc11.6 after actual Torch1.12 compiler mismatch failure; author cu113 deviation recorded.'
digest=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()
for p in [r/'env_torch112.json',e/'env_spec.json']:p.write_text(json.dumps(spec,indent=2))
(e/'env_spec.sha256').write_text(digest+'\n')
runner=r/'install_cu116_and_build.py'
runner.write_text("import os,subprocess,sys\nfrom pathlib import Path\nr=Path("+repr(str(r))+")\ne=Path("+repr(str(e))+")\nimport torch\nassert torch.__file__.startswith(str(e/'venv')+'/'),torch.__file__\nenv=dict(os.environ,TMPDIR=str(e/'tmp'))\ncode=subprocess.call([sys.executable,'-m','pip','uninstall','-y','torch'],env=env)\nassert code==0\ncode=subprocess.call([sys.executable,'-m','pip','install','--disable-pip-version-check','--progress-bar','off','--ignore-installed','torch==1.12.0+cu116','--no-deps','--no-cache-dir','--index-url','https://download.pytorch.org/whl/cu116'],env=env)\n(r/'torch112_install.exit').write_text(str(code)+'\\n')\nif code:raise SystemExit(code)\nwith (r/'pointnet112_build.log').open('wb') as f:code=subprocess.call([sys.executable,str(r/'build_pointnet112.py')],env=env,stdout=f,stderr=subprocess.STDOUT)\nraise SystemExit(code)\n")
for n in ['torch112_install.exit','pointnet112_build.exit']:(r/n).unlink()
f=open(str(r/'torch112_install.log'),'wb');p=subprocess.Popen([str(e/'venv/bin/python'),str(runner)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
rec={'pid':p.pid,'env_spec_sha256':digest,'torch':'1.12.0+cu116','reason':'Match real CUDA11.6 compiler; cu113 failed exact version check','original_environment_modified':False};(r/'torch112_install_launch.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec))
