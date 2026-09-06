import datetime, hashlib, json, os
from pathlib import Path
import time
import paramiko

zone=datetime.timezone(datetime.timedelta(hours=8))
local=Path('C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/scanrefer_local_visual_official_20260906_v3')
remote='/root/autodl-tmp/mcln_scanrefer_local_visual_official_20260906_v3'
first_terminal_check=datetime.datetime(2026,9,6,21,2,tzinfo=zone)
client=paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
_,out,err=client.exec_command('ps -eo pid,ppid,comm,args',timeout=30)
lines=out.read().decode().splitlines()[1:]
assert out.channel.recv_exit_status()==0,err.read().decode()
processes={}
for line in lines:
    parts=line.split(None,3)
    if len(parts)==4:
        processes[int(parts[0])]=parts
matches=[p for p in processes.values() if p[2]=='python' and remote+'/input_manifest.json' in p[3]
         and processes[int(p[1])][2]=='flock']
assert len(matches)==1,matches
pid=int(matches[0][0])
sftp=client.open_sftp()
with sftp.open(remote+'/run.log','rb') as stream:
    stream.seek(max(0,stream.stat().st_size-20000))
    startup_tail=stream.read().decode('utf-8')
startup={'time_cst':datetime.datetime.now(zone).isoformat(),'process_live':True,'remote_pid':pid,
         'remote_process':matches[0],'local_observer_pid':os.getpid(),
         'first_terminal_check_cst':first_terminal_check.isoformat(),'interval_seconds':240,
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'eta_basis':'Actual old-root paired9508 evaluation took1508.56s plus startup;corrected run launched20:37:39,estimated finish21:05. Startup checked once,then wait until21:02.'}
raw=(json.dumps(startup,indent=2,sort_keys=True)+'\n').encode()
(local/'startup_log.txt').write_text(startup_tail,encoding='utf-8')
with (local/'observation_schedule.json').open('xb') as stream:
    stream.write(raw)
with sftp.open(remote+'/observation_schedule.json','wx') as stream:
    stream.write(raw)
sftp.close()
client.close()
print(json.dumps(startup),flush=True)
print('STARTUP LOG TAIL\n'+'\n'.join(startup_tail.splitlines()[-10:]),flush=True)
delay=(first_terminal_check-datetime.datetime.now(zone)).total_seconds()
if delay>0:
    time.sleep(delay)
while True:
    client=paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
    sftp=client.open_sftp()
    _,out,err=client.exec_command('ps -p '+str(pid)+' -o pid,stat,etime,args',timeout=30)
    process=out.read().decode()
    code=out.channel.recv_exit_status()
    assert code in (0,1),err.read().decode()
    live=code==0 and remote+'/input_manifest.json' in process
    with sftp.open(remote+'/run.log','rb') as stream:
        stream.seek(max(0,stream.stat().st_size-64000))
        tail=stream.read().decode('utf-8')
    progress={}
    for line in tail.splitlines():
        for label in ['SCANREFER LOCAL VISUAL OFFICIAL ','SCANREFER LOCAL VISUAL OFFICIAL COMPLETE ']:
            if line.startswith(label+'{'):
                progress[label.strip()]=json.loads(line[len(label):])
    names=sftp.listdir(remote)
    downloaded={}
    result_names=sftp.listdir(remote+'/result') if 'result' in names else []
    targets=[]
    if 'controller.exit' in names:
        targets.append('controller.exit')
    if 'receipt.json' in result_names:
        targets.extend('result/'+name for name in ['receipt.json','rows.json','native_rows.json','protocol.json'])
    for name in targets:
        with sftp.open(remote+'/'+name,'rb') as stream:
            stream.prefetch(file_size=stream.stat().st_size)
            data=stream.read()
        path=local/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
        downloaded[name]=hashlib.sha256(data).hexdigest()
    _,out,err=client.exec_command('df -B1 /root/autodl-tmp',timeout=30)
    disk=out.read().decode()
    assert out.channel.recv_exit_status()==0,err.read().decode()
    sftp.close()
    client.close()
    now=datetime.datetime.now(zone)
    result={'time_cst':now.isoformat(),'process_live':live,'process':process,'progress':progress,
            'downloaded':downloaded,'disk_bytes_report':disk}
    name='progress_'+now.strftime('%Y%m%d_%H%M%S')
    (local/(name+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    (local/(name+'.log')).write_text(tail,encoding='utf-8')
    print(json.dumps(result),flush=True)
    if not live or ('controller.exit' in downloaded and 'result/receipt.json' in downloaded):
        break
    time.sleep(240)

