import datetime,json,os,shlex
from pathlib import Path
import paramiko
b=Path(__file__).resolve().parent
r='/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1'
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();assert 'launch_queue.json' not in s.listdir(r)
for name in ['prepare_evaluation.py','launch_when_inspected.py']:
 compile((b/name).read_bytes(),name,'exec');s.put(str(b/name),r+'/'+name)
python='/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
code="import json,os,subprocess;from pathlib import Path;r=Path(%r);f=(r/'launch_queue.log').open('xb');p=subprocess.Popen([%r,'-u',str(r/'launch_when_inspected.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);v={'pid':p.pid,'inspection_required':True,'preflight_rows':8,'formal_rows':9508,'training_steps':0,'poll_seconds':180};(r/'launch_queue.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))"%(r,python)
_,out,err=c.exec_command(python+' -c '+shlex.quote(code),timeout=30)
stdout=out.read().decode();stderr=err.read().decode();status=out.channel.recv_exit_status()
rec={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'stdout':stdout,'stderr':stderr,'exit':status}
(b/'evaluation_queue.json').write_text(json.dumps(rec,indent=2),encoding='utf-8');print(json.dumps(rec));s.close();c.close();raise SystemExit(status)
