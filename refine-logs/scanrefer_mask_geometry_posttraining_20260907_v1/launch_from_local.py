import datetime,hashlib,json,os,shlex
from pathlib import Path
import paramiko
repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
local=repo/'refine-logs/scanrefer_mask_geometry_posttraining_20260907_v1'
remote='/root/autodl-tmp/mcln_scanrefer_mask_geometry_posttraining_20260907_v1'
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();assert not {'run.log','controller.exit','launch.json'}.intersection(s.listdir(remote))
for name in ['controller.sh','input_manifest.json','posttraining_queue.py']:
    with s.open(remote+'/'+name,'rb') as f:assert f.read()==(local/name).read_bytes()
command='screen -dmS mcln_mask_geometry_queue_v1 bash -c '+shlex.quote('exec bash '+remote+'/controller.sh > '+remote+'/run.log 2>&1')
_,o,e=c.exec_command(command);assert o.channel.recv_exit_status()==0,e.read().decode()
_,o,e=c.exec_command("screen -ls\nps -eo pid,ppid,comm,etime,args | grep '[p]osttraining_queue.py --manifest /root/autodl-tmp/mcln_scanrefer_mask_geometry_posttraining_20260907_v1'",timeout=30)
live=o.read().decode();assert 'mcln_mask_geometry_queue_v1' in live
assert any(len(line.split())>=5 and line.split()[2]=='python' and 'posttraining_queue.py --manifest '+remote in line for line in live.splitlines()),live
record={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'screen':'mcln_mask_geometry_queue_v1','remote_directory':remote,'command':command,'live_processes':live,
    'manifest_sha256':hashlib.sha256((local/'input_manifest.json').read_bytes()).hexdigest(),
    'first_check_cst':'2026-09-07T19:10:00+08:00','interval_seconds':240,'formal_evaluation_started':False,
    'nr3d_sr3d_started':False,'training_restart_count':0}
(local/'launch.json').write_bytes((json.dumps(record,indent=2)+'\n').encode())
(local/'launch_from_local.py').write_bytes(Path(__file__).read_bytes())
for name in ['launch.json','launch_from_local.py','preparation.json','cpu_tests.txt','source_input_check.py']:
    with s.open(remote+'/'+name,'wb') as f:f.write((local/name).read_bytes())
s.close();c.close();print(json.dumps(record),flush=True)
