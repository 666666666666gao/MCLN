"""Test the evidence-based VSA ordering fix with the unchanged frozen-batch protocol."""
import datetime,hashlib,json,os,shlex
from pathlib import Path
import paramiko
repo=Path(__file__).resolve().parents[1]
previous='/root/autodl-tmp/mcln_pvground_initial_replay_20260908_v2'
root='/root/autodl-tmp/mcln_pvground_vsa_order_replay_20260908_v1'
source='/root/autodl-tmp/mcln_pvground_vsa_order_source_20260908_v1'
archive=repo/'refine-logs/pvground_vsa_order_replay_20260908_v1'
c=paramiko.SSHClient();c.load_system_host_keys()
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp()
with s.open('/root/autodl-tmp/mcln_pvground_sparse_order_20260908_v1/controller.exit','rb') as f:assert f.read().strip()==b'0'
with s.open(previous+'/spec.json','rb') as f:spec=json.loads(f.read())
with s.open(source+'/preparation.json','rb') as f:preparation=json.loads(f.read())
assert preparation['original_runtime_unchanged'] and preparation['changed_files']==['models/pv_utils.py']
spec.update(model_source=preparation['model_source'],source_port=preparation['source_port'])
files={'replay.py':(repo/'scripts/replay_pvground_vsa_order.py').read_bytes()}
for name in ['replay_queue.py','controller.py']:
    with s.open(previous+'/'+name,'rb') as f:files[name]=f.read()
spec['files']={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}
spec['scope']='same frozen batch and seed controls, one VSA coordinate/feature ordering correction'
files['spec.json']=(json.dumps(spec,indent=2)+'\n').encode()
assert root.rsplit('/',1)[1] not in s.listdir('/root/autodl-tmp')
s.mkdir(root);archive.mkdir()
for name,raw in files.items():
    (archive/name).write_bytes(raw)
    with s.open(root+'/'+name,'wb') as f:f.write(raw)
    with s.open(root+'/'+name,'rb') as f:assert f.read()==raw
for name in ['source_port.json','preparation.json','PV-Ground/models/pv_utils.py']:
    with s.open(source+'/'+name,'rb') as f:raw=f.read()
    (archive/Path(name).name).write_bytes(raw)
s.symlink('/root/autodl-tmp/mcln_pvground_initial_replay_20260908_v1/inputs',root+'/inputs')
with s.open(root+'/input_controller.exit','wb') as f:f.write(b'0\n')
argv=['screen','-dmS','mcln_pvg_vsa_order_replay_v1','/root/miniconda3/bin/python',root+'/controller.py']
_,out,err=c.exec_command(' '.join(map(shlex.quote,argv)),timeout=30)
assert out.channel.recv_exit_status()==0 and not err.read()
record=dict(root=root,time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    argv=argv,model_source=spec['model_source'],source_sha256=preparation['after_sha256'],optimizer_steps=0,formal_rows=0)
(archive/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record));s.close();c.close()
