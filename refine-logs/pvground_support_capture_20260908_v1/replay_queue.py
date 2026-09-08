"""Wait for the existing Scan formal queue, then run one bounded observation."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


root=Path(__file__).parent
spec=json.loads((root/'spec.json').read_bytes())
dependency=Path(spec['dependency_root'])
(root/'queue.pid').write_text(str(os.getpid())+'\n')
first=datetime.datetime.fromisoformat(spec['first_check_cst']).timestamp()
time.sleep(max(0,first-time.time()))
while not (dependency/'controller.exit').exists():
    process=Path('/proc')/str(spec['dependency_pid'])/'cmdline'
    assert process.exists() and str(dependency).encode() in process.read_bytes()
    print('SUPPORT_WAIT '+datetime.datetime.now().isoformat(),flush=True)
    time.sleep(300)
assert (dependency/'controller.exit').read_text().strip()=='0'
for name,digest in spec['files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
runtime=Path(spec['runtime'])
env=os.environ.copy();env.update(json.loads((runtime/'env_spec.json').read_bytes())['env'])
env['CUDA_VISIBLE_DEVICES']='0'
with open(spec['gpu_lock'],'a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (root/'capture.log').open('xb') as log:
        result=subprocess.run([str(runtime/'venv/bin/python'),'-u',str(root/'capture.py'),
            '--spec',str(root/'spec.json'),'--output',str(root/'observed_replay')],
            env=env,stdout=log,stderr=subprocess.STDOUT)
    (root/'capture.exit').write_text(str(result.returncode)+'\n')
    assert result.returncode==0,'Capture failed; no automatic retry'
print('SUPPORT_QUEUE_COMPLETE',flush=True)
