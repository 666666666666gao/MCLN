"""Collect the original evidence run once; inspect its actual process if unfinished."""
import hashlib,json,os,shlex
from pathlib import Path
import paramiko

repo=Path(__file__).resolve().parents[1]
name='pvground_rec_score_evidence_20260917_v1'
root='/root/autodl-tmp/mcln_'+name
archive=repo/'refine-logs'/name
c=paramiko.SSHClient();c.load_system_host_keys()
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp()
names=s.listdir(root)
for name in ['controller.pid','controller.exit','run.log','diagnostic.json','rows.json','batches.json','input_selection.json','candidate_values.npz']:
    if name in names:
        with s.open(root+'/'+name,'rb') as f:raw=f.read()
        assert len(raw)<10_000_000
        (archive/name).write_bytes(raw)
if 'controller.exit' not in names:
    launch=json.loads((archive/'launch.json').read_bytes());pid=int(launch['process'].split()[0])
    _,out,err=c.exec_command('ps -p '+str(pid)+' -o pid=,etime=,args=',timeout=30)
    process=out.read().decode();assert process and out.channel.recv_exit_status()==0
    print(json.dumps(dict(status='running',process=process,tail=(archive/'run.log').read_text()[-2500:])))
else:
    code=int((archive/'controller.exit').read_text())
    assert code==0,(archive/'run.log').read_text()[-5000:]
    result=json.loads((archive/'diagnostic.json').read_bytes())
    for name,key in [('rows.json','rows_sha256'),('batches.json','batches_sha256'),('candidate_values.npz','arrays_sha256')]:
        assert hashlib.sha256((archive/name).read_bytes()).hexdigest()==result[key]
    print(json.dumps(result))
s.close();c.close()
