"""Install ReferIt formal evaluation/audit entries and test CPU restores, without GPU launch."""
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

repo=Path(__file__).resolve().parents[1]
root='/root/autodl-tmp/mcln_pvground_referit_formal_execution_preparation_20260917_v1'
archive=repo/'refine-logs'/Path(root).name.replace('mcln_','',1)
train_archive=repo/'refine-logs/pvground_referit_rec_training_preparation_20260917_v1'
input_archive=repo/'refine-logs/pvground_referit_formal_preparation_20260917_v1'
nr=json.loads((train_archive/'nr3d/spec.json').read_bytes())
common={
    'evaluate.py':(repo/'scripts/evaluate_pvground_referit_rec.py').read_bytes(),
    'audit.py':(repo/'scripts/audit_pvground_referit_formal.py').read_bytes(),
    'pvground_native_output_recount.py':(repo/'scripts/pvground_native_output_recount.py').read_bytes(),
}
for name in ['pvground_source_query.py','pvground_observation_query.py','pvground_task_observation_query.py']:
    common[name]=(repo/'models'/name).read_bytes()
probe_files=dict(common)
probe_files['check.py']=(repo/'scripts/check_pvground_referit_formal_preparation.py').read_bytes()
probe_files['controller.py']=br'''import json,os,subprocess
from pathlib import Path
root=Path(__file__).parent
spec=json.loads((root/'spec.json').read_bytes());runtime=Path(spec['runtime'])
env=dict(os.environ,**json.loads((runtime/'env_spec.json').read_bytes())['env'])
env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (root/'check.log').open('w') as log:
 result=subprocess.run([str(runtime/'venv/bin/python'),'-u',str(root/'check.py'),'--spec',str(root/'spec.json')],env=env,stdout=log,stderr=subprocess.STDOUT)
(root/'check.exit').write_text(str(result.returncode)+'\n')
raise SystemExit(result.returncode)
'''
probe=dict(runtime=nr['runtime'],model_source=nr['model_source'],source_port=nr['source_port'],
    source_port_sha256=nr['source_port_sha256'],env_spec_sha256=nr['env_spec_sha256'],
    data_root='/root/autodl-tmp/DATA_ROOT_mcln_meshsp/',datasets={},
    files={name:hashlib.sha256(raw).hexdigest() for name,raw in probe_files.items()})
for name,raw in probe_files.items():
    if name.endswith('.py'):compile(raw,name,'exec')
c=paramiko.SSHClient();c.load_system_host_keys();c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();s.mkdir(root);archive.mkdir()
for name,raw in probe_files.items():
    (archive/name).write_bytes(raw)
    with s.open(root+'/'+name,'wx') as f:f.write(raw)
for dataset,short in [('nr3d','nr'),('sr3d','sr')]:
    train_spec_path=train_archive/dataset/'spec.json'
    training=json.loads(train_spec_path.read_bytes())
    formal='/root/autodl-tmp/mcln_pvground_'+short+'_formal_20260917_rec_competition_v1'
    endpoint='/root/autodl-tmp/mcln_pvground_'+short+'_endpoint_audit_20260917_rec_competition_v1'
    s.mkdir(formal);s.mkdir(endpoint);local=archive/dataset;local.mkdir()
    files=dict(common)
    files['formal_input_contract.json']=(input_archive/(dataset+'_formal_input_contract.json')).read_bytes()
    contract=json.loads(files['formal_input_contract.json'])
    assert contract['expected_formal_rows']=={'nr3d':7899,'sr3d':17726}[dataset]
    for name,raw in files.items():
        with s.open(formal+'/'+name,'wx') as f:f.write(raw)
    spec=dict(dataset=dataset,training_root=training['root'],training_audit_root=endpoint,
        training_spec_sha256=hashlib.sha256(train_spec_path.read_bytes()).hexdigest(),
        seed=2027,batch_size=8,primary_mode='bbs',files={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()})
    raw=(json.dumps(spec,indent=2)+'\n').encode();(local/'formal_spec.json').write_bytes(raw)
    with s.open(formal+'/spec.json','wx') as f:f.write(raw)
    endpoint_files={'audit.py':(repo/'scripts/audit_pvground_referit_endpoint.py').read_bytes(),
                    'pvground_native_output_recount.py':common['pvground_native_output_recount.py']}
    for name,raw in endpoint_files.items():
        (local/('endpoint_'+name)).write_bytes(raw)
        with s.open(endpoint+'/'+name,'wx') as f:f.write(raw)
    probe['datasets'][dataset]=dict(parent=training['checkpoint']['path'],parent_sha256=training['checkpoint_sha256'],
        formal_root=formal,endpoint_root=endpoint,formal_spec_sha256=hashlib.sha256((local/'formal_spec.json').read_bytes()).hexdigest(),
        endpoint_files={name:hashlib.sha256(raw).hexdigest() for name,raw in endpoint_files.items()})
raw=(json.dumps(probe,indent=2)+'\n').encode();(archive/'spec.json').write_bytes(raw)
with s.open(root+'/spec.json','wx') as f:f.write(raw)
_,out,err=c.exec_command('/root/miniconda3/envs/bdetr/bin/python -u '+shlex.quote(root+'/controller.py'),timeout=60)
code=out.channel.recv_exit_status()
with s.open(root+'/check.exit','rb') as f:assert f.read().decode().strip()==str(code)
for name in ['check.log','check.exit','receipt.json']:
    if name in s.listdir(root):s.get(root+'/'+name,str(archive/name))
assert code==0,(archive/'check.log').read_text()
print((archive/'receipt.json').read_text(),flush=True)
s.close();c.close()
