import datetime,json,os,shlex
from pathlib import Path
import paramiko
repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();result={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'stages':{}}
for stage in ['finetune','endpoint_audit','formal']:
 name='pvground_scanrefer_'+stage+'_20260917_rec_competition_v1';root='/root/autodl-tmp/mcln_'+name;local=repo/'refine-logs'/name
 names=s.listdir(root);entry={}
 for item in ['launch.json','controller.exit','preflight.exit','preflight_receipt.json','training_start.json','capacity.json','receipt.json','decision.json','audit.json']:
  if item in names:
   with s.open(root+'/'+item,'rb') as f:raw=f.read()
   (local/item).write_bytes(raw)
   if item.endswith('.json'):entry[item]=json.loads(raw)
   else:entry[item]=raw.decode().strip()
 if 'launch.json' in entry:
  pid=int(entry['launch.json']['process'].split()[0]);_,out,err=c.exec_command('ps -p '+str(pid)+' -o pid=,etime=,args=',timeout=30);entry['process']=out.read().decode().strip()
 _,out,err=c.exec_command('tail -n 5 '+shlex.quote(root+'/run.log'),timeout=30);entry['log_tail']=out.read().decode()
 result['stages'][stage]=entry
_,out,err=c.exec_command("df -B1 /root/autodl-tmp; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader",timeout=30)
result['resources']=out.read().decode()
raw=(json.dumps(result,indent=2)+'\n').encode();(repo/'refine-logs/pvground_scanrefer_finetune_20260917_rec_competition_v1/live_status.json').write_bytes(raw)
print(json.dumps(result));s.close();c.close()