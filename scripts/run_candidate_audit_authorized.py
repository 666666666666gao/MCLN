"""Deploy a read-only probe after the support experiment finishes; stream arrays."""
import argparse
import base64
import datetime
import json
import os
from pathlib import Path
import shlex
import paramiko

parser=argparse.ArgumentParser()
parser.add_argument('--limit',type=int,choices=(8,9508),required=True)
args=parser.parse_args()
local=Path(r'C:\Users\gb\.codex\tmp\pvground_fused_support_20261002')
name='sanity' if args.limit==8 else 'full'
destination=local/'candidate_audit'/name
assert not destination.exists()
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',
    password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
experiment='/root/autodl-tmp/pvground_tail_support_fused_retry_20261002'
remote=experiment+'/candidate_audit_'+name
with sftp.open(experiment+'/tail_fused_status.json','rb') as stream:status=json.loads(stream.read())
assert status['status']=='complete' and status['support_arm']=='tail_fused'
with sftp.open(experiment+'/tail_fused/formal/receipt.json','rb') as stream:formal=json.loads(stream.read())
assert formal['status']=='pass' and formal['rows']==9508
if args.limit==9508:
    sanity=json.loads((local/'candidate_audit/sanity/receipt.json').read_bytes())
    assert sanity['status']=='pass' and sanity['rows']==8 and sanity['model_state_unchanged']
    assert sanity['all_candidates_retained']==256
sftp.mkdir(remote)
destination.mkdir(parents=True)
source=local/'run_full_candidate_audit.py'
sftp.put(str(source),remote+'/run.py')
(destination/'run.py').write_bytes(source.read_bytes())
with sftp.open(experiment+'/tail_fused/spec.json','rb') as stream:spec=json.loads(stream.read())
with sftp.open(spec['runtime']+'/env_spec.json','rb') as stream:environment=json.loads(stream.read())
python=spec['runtime']+'/venv/bin/python'
command=['flock','-n',environment['resource_limits']['gpu_lock'],python,'-u',remote+'/run.py',
    '--spec',experiment+'/tail_fused/spec.json','--mode','formal',
    '--audit-output',remote+'/results','--limit',str(args.limit)]
variables=dict(environment['env'])
variables['PYTHONPATH']=spec['root']+':'+variables['PYTHONPATH']
shell='env '+shlex.join([key+'='+value for key,value in variables.items()])+' '+shlex.join(command)
# stderr is merged so a failed process cannot fill a second unread SSH pipe.
_,stdout,_=client.exec_command(shell+' 2>&1',timeout=60)
stdout.channel.settimeout(None)
chunks=[];receipt=None
with (destination/'run.log').open('x',encoding='utf-8') as log:
    for line in stdout:
        if line.startswith('CANDIDATE_CHUNK '):
            _,index,encoded=line.rstrip('\n').split(' ',2)
            assert int(index)==len(chunks)
            raw=base64.b64decode(encoded,validate=True)
            chunk=destination/(f'candidates_{int(index):04d}.npz')
            chunk.write_bytes(raw)
            chunks.append(dict(name=chunk.name,bytes=len(raw)))
        else:
            log.write(line);log.flush()
            if line.startswith('CANDIDATE_AUDIT_COMPLETE '):
                receipt=json.loads(line.split(' ',1)[1])
            if line.startswith('CANDIDATE_AUDIT_PROGRESS '):print(line.strip(),flush=True)
    exit_code=stdout.channel.recv_exit_status()
(destination/'run.exit').write_text(str(exit_code)+'\n',encoding='utf-8')
if exit_code:
    print(json.dumps(dict(status='failed',exit_code=exit_code,log=str(destination/'run.log'))),flush=True)
    raise SystemExit(exit_code)
assert receipt and receipt['status']=='pass' and receipt['rows']==args.limit
assert receipt['chunks']==len(chunks)
for name in ('imports.json','load.json','rows.jsonl','receipt.json'):
    with sftp.open(remote+'/results/'+name,'rb') as stream:raw=stream.read()
    (destination/name).write_bytes(raw)
client.close()
intake=dict(status='pass',time_cst=datetime.datetime.now().astimezone().isoformat(),
    limit=args.limit,remote=remote,remote_arrays_written=False,optimizer_updates=0,
    chunks=chunks,total_streamed_array_bytes=sum(c['bytes'] for c in chunks),receipt=receipt)
(destination/'INTAKE.json').write_text(json.dumps(intake,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status='pass',directory=str(destination),rows=args.limit,
    chunks=len(chunks),rec_hits=receipt['rec_hits'],elapsed_seconds=receipt['elapsed_seconds'],
    total_streamed_array_bytes=intake['total_streamed_array_bytes'])),flush=True)
