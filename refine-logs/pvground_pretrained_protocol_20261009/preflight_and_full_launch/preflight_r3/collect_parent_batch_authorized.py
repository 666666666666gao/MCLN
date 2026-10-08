"""Observe and collect the original read-only evaluation after its planned time."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

parser=argparse.ArgumentParser()
parser.add_argument('--phase',choices=('preflight','formal'),required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parent
launch=json.loads((root/(args.phase.upper()+'_LAUNCH.json')).read_bytes())
stamp=datetime.datetime.now().astimezone()
assert stamp>=datetime.datetime.fromisoformat(launch['first_outcome_check_cst'])
remote=launch['remote_root']
process_id=int(launch['process'].split()[0])
code="""import hashlib,json
from pathlib import Path
r=Path(%r);phase=%r
p=r/(phase+'.exit')
files=[r/(phase+'.exit'),r/(phase+'.log'),r/(phase+'_receipt.json'),r/(phase+'_imports.json')]
directory=r/phase
if directory.exists():files += sorted(v for v in directory.rglob('*') if v.is_file())
print(json.dumps(dict(exitcode=int(p.read_text()) if p.exists() else None,process_live=Path('/proc/%s').exists(),files=[dict(name=str(f.relative_to(r)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files if f.exists()])))
""" % (remote,args.phase,process_id)
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
_,stdout,stderr=client.exec_command(shlex.join(['/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python','-B','-c',code]),timeout=60)
observation=json.loads(stdout.read())
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
observation['observed_cst']=stamp.isoformat()
if observation['exitcode'] is not None:
    destination=root/(args.phase+'_actual');destination.mkdir()
    sftp=client.open_sftp()
    for entry in observation['files']:
        path=destination/entry['name'];path.parent.mkdir(parents=True,exist_ok=True)
        sftp.get(remote+'/'+entry['name'],str(path))
        data=path.read_bytes()
        assert len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256']
    sftp.close()
    (destination/'INTAKE.json').write_text(json.dumps(observation,indent=2)+'\n',encoding='utf-8')
    if observation['exitcode']==0:
        print((destination/(args.phase+'_receipt.json')).read_text(encoding='utf-8'),flush=True)
    else:
        print((destination/(args.phase+'.log')).read_text(encoding='utf-8')[-5000:],flush=True)
client.close()
(root/(args.phase.upper()+'_OBSERVATION_'+stamp.strftime('%H%M%S')+'.json')).write_text(json.dumps(observation,indent=2)+'\n',encoding='utf-8')
print(json.dumps(observation),flush=True)
