"""Serial read-only pilot and full fixed-candidate diagnostic."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
spec_path = Path('/root/pvground_g_p2_20261002/g_p2/spec.json')
spec = json.loads(spec_path.read_bytes())
environment = json.loads((Path(spec['runtime'])/'env_spec.json').read_bytes())
env = dict(os.environ, **environment['env'])
env['PYTHONPATH'] = '/root/pvground_g_p2_20261002:'+env['PYTHONPATH']
record = {'status':'running','checkpoint':'P2 step3723','optimizer_updates':0,
          'started_cst':datetime.datetime.now().astimezone().isoformat(),'completed':[]}

def save():
    temporary = root/'status.json.tmp'
    temporary.write_text(json.dumps(record,indent=2)+'\n')
    os.replace(str(temporary),str(root/'status.json'))

for phase, count in [('pilot',8),('full',9508)]:
    if phase == 'full':
        pilot = json.loads((root/'pilot/receipt.json').read_bytes())
        assert pilot['status']=='complete' and pilot['rows']==8
        assert pilot['historical_actual_mismatches']==0
        # The disk is known to be tight. Use the measured compressed pilot,
        # plus a small output/log reserve; no checkpoint copies are written.
        required = 2*pilot['rows_bytes']*9508//8+128*1024**2
        assert shutil.disk_usage(root).free >= required
    command = ['flock','-n',environment['resource_limits']['gpu_lock'],sys.executable,'-u',
               str(root/'run.py'),'--spec',str(spec_path),'--output',str(root/phase),'--rows',str(count)]
    begin = time.time()
    with (root/(phase+'.log')).open('x') as log:
        child = subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        record.update(stage=phase,process_pid=child.pid)
        save()
        code = child.wait()
    (root/(phase+'.exit')).write_text(str(code)+'\n')
    if code:
        record.update(status='failed',exit_code=code)
        save()
        raise SystemExit(code)
    receipt = json.loads((root/phase/'receipt.json').read_bytes())
    assert receipt['status']=='complete' and receipt['rows']==count
    record['completed'].append({'stage':phase,'seconds':time.time()-begin,
                                'receipt_sha256':hashlib.sha256((root/phase/'receipt.json').read_bytes()).hexdigest()})
    save()
record.update(status='complete',finished_cst=datetime.datetime.now().astimezone().isoformat())
save()
