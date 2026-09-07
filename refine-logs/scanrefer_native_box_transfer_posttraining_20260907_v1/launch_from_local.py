import datetime,hashlib,json,os,shlex
from pathlib import Path
import paramiko
repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
local=repo/'refine-logs/scanrefer_native_box_transfer_posttraining_20260907_v1'
remote='/root/autodl-tmp/mcln_scanrefer_native_box_transfer_posttraining_20260907_v1'
spec=json.loads((local/'input_manifest.json').read_bytes())
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();assert not set(['launch.json','run.log','controller.exit']).intersection(s.listdir(remote))
for name in ['input_manifest.json','posttraining_queue.py','controller.sh']:
 with s.open(remote+'/'+name,'rb') as stream:assert stream.read()==(local/name).read_bytes(),name
_,o,e=c.exec_command('bash -n '+shlex.quote(remote+'/controller.sh'),timeout=30);assert o.channel.recv_exit_status()==0,e.read().decode()
command='screen -dmS mcln_native_box_transfer_post_v1 bash -c '+shlex.quote('exec bash '+remote+'/controller.sh > '+remote+'/run.log 2>&1')
_,o,e=c.exec_command(command,timeout=30);assert o.channel.recv_exit_status()==0,e.read().decode()
_,o,e=c.exec_command("screen -ls\nps -eo pid,ppid,comm,stat,etime,args | grep '[p]osttraining_queue.py'",timeout=30)
live=o.read().decode();assert 'mcln_native_box_transfer_post_v1' in live and 'python' in live
value={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 'screen':'mcln_native_box_transfer_post_v1','live':live,'command':command,
 'manifest_sha256':hashlib.sha256((local/'input_manifest.json').read_bytes()).hexdigest(),
 'queue_script_sha256':spec['queue_script_sha256'],'training_pid_unchanged':58023,'first_check_cst':spec['first_check_cst'],
 'interval_seconds':240,'formal_evaluation_started':False,'training_files_modified':False}
(local/'launch.json').write_bytes((json.dumps(value,indent=2)+'\n').encode())
(local/'launch_from_local.py').write_bytes(Path(__file__).read_bytes())
for name in ['launch.json','launch_from_local.py']:
 with s.open(remote+'/'+name,'wx') as stream:stream.write((local/name).read_bytes())
s.close();c.close();print(json.dumps(value),flush=True)
