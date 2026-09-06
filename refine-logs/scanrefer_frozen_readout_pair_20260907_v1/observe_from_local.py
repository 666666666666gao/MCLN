import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
name='scanrefer_frozen_readout_pair_20260907_v1'
local=repo/'refine-logs'/name
remote='/root/autodl-tmp/mcln_'+name
queue='/root/autodl-tmp/mcln_scanrefer_frozen_readout_posttraining_20260907_v1'
client=paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
code='''import datetime,json,shutil,subprocess
from pathlib import Path
training=Path('''+repr(remote)+''')
queue=Path('''+repr(queue)+''')
def tail(path,count=64000):
 with path.open('rb') as stream:
  stream.seek(max(0,path.stat().st_size-count))
  return stream.read().decode(errors='replace')
launch=json.loads((training/'launch.json').read_text())
ql=json.loads((queue/'launch.json').read_text())
pids=[x['screen_session'][0].split('.')[0] for x in [launch,ql]]
p=subprocess.run(['ps','-p',','.join(pids),'-o','pid,ppid,stat,etime,args'],stdout=subprocess.PIPE)
assert p.returncode in [0,1]
lines=tail(training/'run.log').splitlines()
progress={}
for line in lines:
 for prefix in ['SCANREFER FROZEN READOUT EVAL ','SCANREFER FROZEN READOUT EVAL COMPLETE ','SCANREFER FROZEN READOUT TRAIN ']:
  if line.startswith(prefix+'{'):progress[prefix.strip()]=json.loads(line[len(prefix):])
gpu=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],stdout=subprocess.PIPE)
assert gpu.returncode==0
result={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 'processes':p.stdout.decode(),'gpu':gpu.stdout.decode().strip(),'progress':progress,
 'disk_free':shutil.disk_usage('/root/autodl-tmp').free,'training_files':sorted(x.name for x in training.iterdir()),
 'queue_files':sorted(x.name for x in queue.iterdir()),'last_log_lines':lines[-6:],
 'controller_exit':(training/'controller.exit').read_text().strip() if (training/'controller.exit').exists() else None,
 'queue_controller_exit':(queue/'controller.exit').read_text().strip() if (queue/'controller.exit').exists() else None}
if (queue/'decision.json').exists():result['queue_decision']=json.loads((queue/'decision.json').read_text())
if (queue/'training_observations.jsonl').exists():result['last_queue_observation']=json.loads(tail(queue/'training_observations.jsonl').splitlines()[-1])
print(json.dumps(result))
'''
_,output,error=client.exec_command('/root/miniconda3/envs/bdetr/bin/python -c '+shlex.quote(code),timeout=30)
result=json.loads(output.read())
assert output.channel.recv_exit_status()==0,error.read().decode()
sftp=client.open_sftp()
for root,directory,names in [
    (remote,local,['controller.exit','protocol.json','baseline_metrics.json','baseline_native_metrics.json','fit_complete.json','terminal_metrics.json','terminal_native_metrics.json','receipt.json','independent_audit.json']),
    (queue,repo/'refine-logs/scanrefer_frozen_readout_posttraining_20260907_v1',['controller.exit','observation_schedule.json','decision.json'])]:
    available=sftp.listdir(root)
    for item in names:
        if item in available and not (directory/item).exists():
            size=sftp.stat(root+'/'+item).st_size
            with sftp.open(root+'/'+item,'rb') as stream:
                stream.prefetch(file_size=size)
                raw=stream.read()
            assert len(raw)==size
            (directory/item).write_bytes(raw)
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
snapshot='observation_'+now.strftime('%H%M%S')+'.json'
raw=(json.dumps(result,indent=2,sort_keys=True)+'\n').encode()
(local/snapshot).write_bytes(raw)
with sftp.open(remote+'/'+snapshot,'wx') as stream:stream.write(raw)
sftp.close()
client.close()
result['local_observation_file']=str(local/snapshot)
print(json.dumps(result),flush=True)
