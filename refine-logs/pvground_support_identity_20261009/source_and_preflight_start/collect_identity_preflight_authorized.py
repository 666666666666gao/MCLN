"""Wait until the planned milestone, then collect the original sanity job."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import time
import paramiko

root = Path(__file__).resolve().parent
launch = json.loads((root/'PREFLIGHT_LAUNCH.json').read_bytes())
parent = json.loads((root.parent/'pvground_compressed_geometry_support_20261008/pair_spec.json').read_bytes())
deadline = datetime.datetime.fromisoformat(launch['first_observation_cst'])
assert not (root/'actual/INTAKE.json').exists()
remaining = (deadline-datetime.datetime.now().astimezone()).total_seconds()
if remaining>0:time.sleep(remaining)
while True:
    stamp=datetime.datetime.now().astimezone()
    assert stamp>=deadline
    client=paramiko.SSHClient();client.load_system_host_keys()
    client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
    code="""import datetime,hashlib,json,sys
from pathlib import Path
b=json.load(sys.stdin);r=Path(b['root']);p=r/'preflight.exit'
assert r==Path('/root/autodl-tmp/pvground_support_identity_20261009')
files=[r/name for name in ('preflight.exit','preflight.log','preflight.json')]
print(json.dumps(dict(exitcode=int(p.read_text()) if p.exists() else None,process_live=Path('/proc/'+str(b['pid'])).exists(),exit_marker_mtime_cst=datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone(datetime.timedelta(hours=8))).isoformat() if p.exists() else None,files=[dict(name=v.name,bytes=v.stat().st_size,sha256=hashlib.sha256(v.read_bytes()).hexdigest()) for v in files if v.exists()])))
"""
    stdin,stdout,stderr=client.exec_command(shlex.join([parent['runtime']+'/venv/bin/python','-B','-c',code]),timeout=60)
    stdin.write(json.dumps(dict(root=launch['root'],pid=launch['pid'])));stdin.channel.shutdown_write()
    observed=json.loads(stdout.read());assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
    observed['observed_cst']=stamp.isoformat()
    with (root/'OBSERVATIONS.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(observed)+'\n')
    if observed['exitcode'] is not None:
        destination=root/'actual';destination.mkdir()
        sftp=client.open_sftp()
        for item in observed['files']:
            assert item['name'] in ('preflight.exit','preflight.log','preflight.json')
            path=destination/item['name'];sftp.get(launch['root']+'/'+item['name'],str(path))
            assert path.stat().st_size==item['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        sftp.close();client.close()
        (destination/'INTAKE.json').write_text(json.dumps(observed,indent=2)+'\n',encoding='utf-8')
        if observed['exitcode']==0:
            report=json.loads((destination/'preflight.json').read_bytes())
            assert report['status']=='PASS_ACTUAL_B8_TWO_STEP_SUPPORT_IDENTITY_PREFLIGHT'
            print(json.dumps({k:report[k] for k in ('status','optimizer_steps_per_arm','head_parameters','head_state_tensors','peak_allocated_bytes','peak_reserved_bytes','formal_accuracy_result','weight_files_created')}),flush=True)
        else:
            print((destination/'preflight.log').read_text(encoding='utf-8')[-4000:],flush=True)
        print(json.dumps(observed),flush=True)
        break
    client.close()
    assert observed['process_live'],'Original process absent without exit marker; inspect authoritative state, do not restart.'
    time.sleep(240)
