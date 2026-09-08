import datetime,json,os,shlex
from pathlib import Path
import paramiko
root='/root/autodl-tmp/mcln_scanrefer_detection_augmentation_20260908_v1'
archive=Path('C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/scanrefer_detection_augmentation_20260908_v1')
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();names=s.listdir(root)
record=dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),files=names)
for name in ['controller.pid','controller.exit','run.log','receipt.json','rows.json']:
    if name in names:
        s.get(root+'/'+name,str(archive/name))
        if name=='run.log':record['tail']=(archive/name).read_text(encoding='utf-8').splitlines()[-4:]
        elif name.endswith('.json') and name!='rows.json':record[name]=json.loads((archive/name).read_bytes())
        elif name!='rows.json':record[name]=(archive/name).read_text().strip()
_,out,err=c.exec_command("ps -eo pid,ppid,etimes,args",timeout=30)
record['processes']=[line for line in out.read().decode().splitlines() if root in line]
assert out.channel.recv_exit_status()==0 and not err.read()
(archive/'observation.json').write_bytes((json.dumps(record,indent=2)+'\n').encode())
print(json.dumps(record));s.close();c.close()
