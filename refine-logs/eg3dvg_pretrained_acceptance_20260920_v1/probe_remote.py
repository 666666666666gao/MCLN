import json,os,datetime
from pathlib import Path
import paramiko

base=Path(__file__).resolve().parent
c=paramiko.SSHClient();c.load_system_host_keys()
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
commands={
 'resources':'nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader; df -B1 /root/autodl-tmp /tmp; free -b',
 'existing':'find /root/autodl-tmp -maxdepth 3 -iname "*eg3d*" -print',
 'environments':'ls /root/miniconda3/envs; cat /root/autodl-tmp/mcln_pvground_runtime_20260908_v1/env_spec.json',
 'data':'ls /root/autodl-tmp/DATA_ROOT_mcln_meshsp; ls /root/autodl-tmp/DATA_ROOT_mcln_meshsp/ScanRefer',
}
record={'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}
for key,cmd in commands.items():
 _,o,e=c.exec_command(cmd,timeout=45)
 record[key]={'stdout':o.read().decode(),'stderr':e.read().decode(),'exit':o.channel.recv_exit_status()}
c.close()
(base/'remote_probe.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
