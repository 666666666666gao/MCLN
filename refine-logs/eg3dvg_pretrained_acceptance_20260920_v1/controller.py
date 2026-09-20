import fcntl,json,os,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent
lock=open('/root/autodl-tmp/mcln_v99_backbone_gpu0.lock','a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
code=0
for stage in ['preflight','formal']:
 with (r/(stage+'.log')).open('xb') as f:
  code=subprocess.call([sys.executable,'-u',str(r/'evaluate.py'),'--spec',str(r/'spec.json'),'--stage',stage],cwd=str(r/'source'),stdout=f,stderr=subprocess.STDOUT)
 (r/(stage+'.exit')).write_text(str(code)+'\n')
 if code:break
if code==0:
 with (r/'audit.log').open('xb') as f:
  code=subprocess.call([sys.executable,'-u',str(r/'audit.py'),'--root',str(r)],cwd=str(r/'source'),stdout=f,stderr=subprocess.STDOUT)
 (r/'audit.exit').write_text(str(code)+'\n')
(r/'controller.exit').write_text(str(code)+'\n')
raise SystemExit(code)
