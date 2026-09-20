import fcntl, json, subprocess, sys, time
from pathlib import Path
r = Path(__file__).resolve().parent
spec = json.loads((r/'spec.json').read_text())
prior = Path(spec['wait_for_nr_adaptation'])
while not (prior/'controller.exit').exists():
    time.sleep(300)
assert (prior/'controller.exit').read_text().strip() == '0', 'Nr adaptation controller did not complete cleanly'
assert json.loads((prior/'evaluation/formal/audit.json').read_text())['integrity_pass']
lock = open('/root/autodl-tmp/mcln_v99_backbone_gpu0.lock', 'a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
code = 0
for stage in ['preflight','formal']:
    with (r/(stage+'.log')).open('xb') as f:
        code = subprocess.call([sys.executable,'-u',str(r/'evaluate.py'),'--spec',str(r/'spec.json'),'--stage',stage],cwd=spec['source'],stdout=f,stderr=subprocess.STDOUT)
    (r/(stage+'.exit')).write_text(str(code)+'\n')
    if code:
        break
if code == 0:
    with (r/'audit.log').open('xb') as f:
        code = subprocess.call([sys.executable,'-u',str(r/'audit.py'),'--root',str(r)],cwd=spec['source'],stdout=f,stderr=subprocess.STDOUT)
    (r/'audit.exit').write_text(str(code)+'\n')
(r/'controller.exit').write_text(str(code)+'\n')
raise SystemExit(code)
