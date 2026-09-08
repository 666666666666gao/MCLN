"""Read transfer/inventory state without restarting either process."""
import datetime
import json
import os
from pathlib import Path
import paramiko

root='/root/autodl-tmp/mcln_pvground_nr_checkpoint_inspection_20260908_v1'
archive=Path(__file__).resolve().parents[1]/'refine-logs/pvground_nr_checkpoint_inspection_20260908_v1'
c=paramiko.SSHClient();c.load_system_host_keys()
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();files=s.listdir(root)
record=dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),files=files)
for name in ['PV-Ground_NR3D.pth.part','PV-Ground_NR3D.pth']:
    if name in files:record[name]={'bytes':s.stat(root+'/'+name).st_size}
for name in ['inventory.exit','inventory.log','receipt.json','state_inventory.json','strict_load.json','strict_load.exit','strict_load.log','strict_load.stderr']:
    if name in files:
        s.get(root+'/'+name,str(archive/name))
        if name.endswith('.json') and name!='state_inventory.json':record[name]=json.loads((archive/name).read_bytes())
        if name.endswith('.exit'):record[name]=(archive/name).read_text().strip()
s.close();c.close()
(archive/'observation_latest.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record),flush=True)
