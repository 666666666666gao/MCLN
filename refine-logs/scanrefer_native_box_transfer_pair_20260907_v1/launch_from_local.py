import datetime,hashlib,json,os,shlex,subprocess
from pathlib import Path
import paramiko
repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
local=repo/'refine-logs/scanrefer_native_box_transfer_pair_20260907_v1'
remote='/root/autodl-tmp/mcln_scanrefer_native_box_transfer_pair_20260907_v1'
manifest=json.loads((local/'input_manifest.json').read_bytes())
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=repo,text=True).split()[0]==head
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();assert not set(['launch.json','run.log','training.exit','controller.exit']).intersection(s.listdir(remote))
for name in ['input_manifest.json','controller.sh']+list(manifest['files']):
 with s.open(remote+'/'+name,'rb') as f:assert f.read()==(local/name).read_bytes(),name
_,o,e=c.exec_command('nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits\ndf -B1 /root/autodl-tmp\nscreen -ls',timeout=30)
before=o.read().decode();print(before,flush=True)
assert int(before.splitlines()[0].split(',')[2].strip())<500
assert int(before.splitlines()[2].split()[3])>2*1024**3
assert 'mcln_native_box_transfer_pair_v1' not in before
command='screen -dmS mcln_native_box_transfer_pair_v1 bash -c '+shlex.quote('exec bash '+remote+'/controller.sh > '+remote+'/run.log 2>&1')
_,o,e=c.exec_command(command,timeout=30);assert o.channel.recv_exit_status()==0,e.read().decode()
_,o,e=c.exec_command("screen -ls\nps -eo pid,ppid,comm,etime,args | grep '[r]un_scanrefer_native_box_transfer_pair'",timeout=30)
live=o.read().decode();assert 'mcln_native_box_transfer_pair_v1' in live
proof={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 'remote_directory':remote,'screen':'mcln_native_box_transfer_pair_v1','command':command,'published_source_commit':head,
 'manifest_sha256':hashlib.sha256((local/'input_manifest.json').read_bytes()).hexdigest(),'resource_precheck':before,'live_processes':live,
 'steps_per_arm':2482,'candidate':'gt_teacher_box','control':'gt_only','formal_rows':0,
 'estimated_initial_baseline_minutes':35,'estimated_total_hours':4,'poll_interval_seconds':240,
 'terminal_cpu_audit_chained':True,'quality_result_available':False,'nr3d_sr3d_started':False}
raw=(json.dumps(proof,indent=2)+'\n').encode()
(local/'launch.json').write_bytes(raw);(local/'launch_from_local.py').write_bytes(Path(__file__).read_bytes())
for name in ['launch.json','launch_from_local.py']:
 with s.open(remote+'/'+name,'wx') as f:f.write((local/name).read_bytes())
s.close();c.close();print(json.dumps(proof),flush=True)
