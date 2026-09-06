import datetime
import json
import os
from pathlib import Path
import shlex

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
probe = """import datetime,json,shutil,subprocess
from pathlib import Path
root=Path('/root/autodl-tmp/mcln_scanrefer_range_official_20260907_v1')
launch=json.loads((root/'launch.json').read_text())
result={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'launch':launch,'processes':{}}
for name,pid in [('formal',launch['process_pid']),('posttraining_queue',launch['parent_queue_pid']),('native_queue_screen',48128)]:
 p=subprocess.run(['ps','-p',str(pid),'-o','pid,ppid,stat,etime,args'],stdout=subprocess.PIPE)
 result['processes'][name]={'returncode':p.returncode,'output':p.stdout.decode()}
p=subprocess.run(['ps','--ppid',str(launch['process_pid']),'-o','pid,ppid,stat,etime,args'],stdout=subprocess.PIPE)
result['formal_children']={'returncode':p.returncode,'output':p.stdout.decode()}
with (root/'run.log').open('rb') as stream:
 stream.seek(max(0,(root/'run.log').stat().st_size-24000))
 tail=stream.read().decode()
prefix='SCANREFER RANGE OFFICIAL '
progress=[json.loads(line[len(prefix):]) for line in tail.splitlines() if line.startswith(prefix+'{')]
result['latest_progress']=progress[-1] if progress else None
result['log_tail']=tail.splitlines()[-10:]
for name,folder in [('formal',root),('posttraining',Path('/root/autodl-tmp/mcln_scanrefer_range_posttraining_20260907_v1')),('native_queue',Path('/root/autodl-tmp/mcln_native_range_preflight_queue_20260907_v1'))]:
 result[name+'_outputs']={}
 for file in ['controller.exit','decision.json','result/receipt.json','result/independent_audit.json']:
  path=folder/file
  if path.exists():result[name+'_outputs'][file]=path.read_text()
result['disk']=dict(zip(['total','used','free'],shutil.disk_usage('/root/autodl-tmp')))
p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],stdout=subprocess.PIPE)
result['gpu_processes']={'returncode':p.returncode,'output':p.stdout.decode()}
print(json.dumps(result))
"""
_, output, error = client.exec_command('/root/miniconda3/envs/bdetr/bin/python -c ' + shlex.quote(probe), timeout=30)
raw = output.read()
assert output.channel.recv_exit_status() == 0, error.read().decode()
state = json.loads(raw)
stamp = datetime.datetime.fromisoformat(state['time_cst']).strftime('%Y%m%d_%H%M%S')
path = repo / 'refine-logs/scanrefer_range_official_20260907_v1' / ('observation_' + stamp + '.json')
with path.open('xb') as stream:
    stream.write((json.dumps(state, indent=2, sort_keys=True) + '\n').encode())
client.close()
print(json.dumps(state), flush=True)
