"""Collect the fixed loss/score diagnostic; never restart the remote handle."""
import hashlib
import json
import os
from pathlib import Path
import paramiko

repo=Path(__file__).resolve().parents[1]
root='/root/autodl-tmp/mcln_pvground_parameter_direction_20260917_v1'
archive=repo/'refine-logs/pvground_parameter_direction_20260917_v1'
c=paramiko.SSHClient();c.load_system_host_keys()
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();names=s.listdir(root)
for name in ['controller.pid','controller.exit','run.log','input_selection.json','rows.json','diagnostic.json','candidate_values.npz','batches.json','parameter_groups.json']:
    if name in names:s.get(root+'/'+name,str(archive/name))
print((archive/'run.log').read_text(encoding='utf-8')[-7000:])
if 'controller.exit' in names:print('controller.exit='+ (archive/'controller.exit').read_text().strip())
if 'diagnostic.json' in names:
    result=json.loads((archive/'diagnostic.json').read_bytes())
    for filename,key in [('candidate_values.npz','arrays_sha256'),('rows.json','rows_sha256'),('input_selection.json','input_selection_sha256'),('batches.json','batches_sha256')]:
        assert hashlib.sha256((archive/filename).read_bytes()).hexdigest()==result[key]
    print(json.dumps(result))
s.close();c.close()
