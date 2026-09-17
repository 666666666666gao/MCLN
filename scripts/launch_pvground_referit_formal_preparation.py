"""Run CPU endpoint-audit tests and native validation input binding alongside Scan training."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

repo=Path(__file__).resolve().parents[1]
root='/root/autodl-tmp/mcln_pvground_referit_formal_preparation_20260917_v1'
archive=repo/'refine-logs'/Path(root).name.replace('mcln_','',1)
part=json.loads((repo/'refine-logs/pvground_referit_fit_partitions_20260917_v1/spec.json').read_bytes())
part_receipt=json.loads((repo/'refine-logs/pvground_referit_fit_partitions_20260917_v1/receipt.json').read_bytes())
files={name:(repo/'scripts'/name).read_bytes() for name in ['prepare_pvground_referit_formal_inputs.py','audit_pvground_referit_endpoint.py','pvground_native_output_recount.py']}
files['tests/test_pvground_referit_endpoint_audit.py']=(repo/'tests/test_pvground_referit_endpoint_audit.py').read_bytes()
spec=dict(runtime=part['runtime'],selection_manifest=part['selection_root']+'/manifest.json',
    selection_manifest_sha256=part['selection_manifest_sha256'],full_point_manifest=part['scan_input_manifest'],
    full_point_manifest_sha256=part['scan_input_manifest_sha256'],
    partitions={name:dict(path='/root/autodl-tmp/mcln_pvground_referit_fit_partitions_20260917_v1/'+name+'_partition.json',sha256=item['partition_sha256']) for name,item in part_receipt['datasets'].items()},
    script_sha256=hashlib.sha256(files['prepare_pvground_referit_formal_inputs.py']).hexdigest())
files['spec.json']=(json.dumps(spec,indent=2)+'\n').encode()
files['controller.py']=br'''import json,os,subprocess
from pathlib import Path
root=Path(__file__).parent
(root/'controller.pid').write_text(str(os.getpid())+'\n')
spec=json.loads((root/'spec.json').read_bytes());runtime=Path(spec['runtime'])
env=dict(os.environ,**json.loads((runtime/'env_spec.json').read_bytes())['env'])
env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(root))
with (root/'tests.log').open('w') as log:
 test=subprocess.run([str(runtime/'venv/bin/python'),str(root/'tests/test_pvground_referit_endpoint_audit.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
(root/'tests.exit').write_text(str(test.returncode)+'\n')
if test.returncode!=0:
 (root/'controller.exit').write_text(str(test.returncode)+'\n')
 raise SystemExit(test.returncode)
with (root/'run.log').open('w') as log:
 result=subprocess.run([str(runtime/'venv/bin/python'),'-u',str(root/'prepare_pvground_referit_formal_inputs.py'),'--spec',str(root/'spec.json')],env=env,stdout=log,stderr=subprocess.STDOUT)
(root/'controller.exit').write_text(str(result.returncode)+'\n')
raise SystemExit(result.returncode)
'''
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();s.mkdir(root);s.mkdir(root+'/tests');archive.mkdir()
for name,raw in files.items():
    if name.endswith('.py'):compile(raw,name,'exec')
    path=archive/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    with s.open(root+'/'+name,'wx') as f:f.write(raw)
inner='exec /root/miniconda3/envs/bdetr/bin/python -u '+shlex.quote(root+'/controller.py')
_,out,err=c.exec_command('screen -dmS mcln_pvg_referit_formal_inputs_v1 bash -c '+shlex.quote(inner),timeout=30)
assert out.channel.recv_exit_status()==0,err.read().decode()
_,out,err=c.exec_command('pgrep -af '+shlex.quote('^/root/miniconda3/envs/bdetr/bin/python -u '+root+'/controller.py$'),timeout=30)
process=out.read().decode().strip();assert process and out.channel.recv_exit_status()==0
record=dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),process=process,
            scope='CPU audit tests and formal input binding only, no model or training')
raw=(json.dumps(record,indent=2)+'\n').encode();(archive/'launch.json').write_bytes(raw)
with s.open(root+'/launch.json','wx') as f:f.write(raw)
print(json.dumps(record));s.close();c.close()
