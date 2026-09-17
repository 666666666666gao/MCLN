"""Collect the original annotation-only partition controller and its outputs."""
import datetime,hashlib,json,os,shlex
from pathlib import Path
import paramiko
repo=Path(__file__).resolve().parents[1];root='/root/autodl-tmp/mcln_pvground_referit_fit_partitions_20260917_v1'
archive=repo/'refine-logs'/Path(root).name.replace('mcln_','',1)
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30);s=c.open_sftp();names=s.listdir(root)
result=dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
for name in ['controller.pid','controller.exit','run.log','receipt.json']:
 if name in names:
  with s.open(root+'/'+name,'rb') as f:raw=f.read()
  (archive/name).write_bytes(raw)
  result[name]=json.loads(raw) if name.endswith('.json') else raw.decode()[-3000:]
if 'receipt.json' in result:
 for dataset,info in result['receipt.json']['datasets'].items():
  name=dataset+'_partition.json';s.get(root+'/'+name,str(archive/name))
  assert hashlib.sha256((archive/name).read_bytes()).hexdigest()==info['partition_sha256']
launch=json.loads((archive/'launch.json').read_bytes());pid=int(launch['process'].split()[0])
_,out,err=c.exec_command('ps -p '+str(pid)+' -o pid=,etime=,args=',timeout=30);result['original_process']=out.read().decode()
(archive/'observation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));s.close();c.close()