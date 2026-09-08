"""Stage the existing native loss/evaluator/backward check with the empty-pool control installed."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

repo=Path(__file__).resolve().parents[1]
root='/root/autodl-tmp/mcln_pvground_empty_pool_interface_20260908_v1'
archive=repo/'refine-logs/pvground_empty_pool_interface_20260908_v1'
runtime='/root/autodl-tmp/mcln_pvground_runtime_20260908_v1'
model_source='/root/autodl-tmp/mcln_pvground_vsa_order_source_20260908_v1/PV-Ground'
fixtures='/root/autodl-tmp/mcln_pvground_training_interface_20260908_v1/fixtures'
c=paramiko.SSHClient();c.load_system_host_keys()
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp()
for dependency in ['mcln_pvground_empty_pool_check_20260908_v1','mcln_pvground_empty_pool_replay_20260908_v1']:
    with s.open('/root/autodl-tmp/'+dependency+'/controller.exit') as f:assert f.read().strip()==b'0'
with s.open(runtime+'/env_spec.json','rb') as f:env=json.loads(f.read())
env_sha=hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert env_sha=='966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c'
with s.open(fixtures+'/receipt.json','rb') as f:fixture_receipt=json.loads(f.read())
assert fixture_receipt['status']=='pass' and fixture_receipt['separate_training_labels']
assert [r['training_row_id'] for r in fixture_receipt['rows']]==[0,173,237,455]
_,out,err=c.exec_command('nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader',timeout=30)
assert not out.read().strip() and out.channel.recv_exit_status()==0,err.read().decode()
original=(repo/'scripts/check_pvground_training_interface.py').read_bytes()
code=original.decode()
code=code.replace('root = args.runtime.resolve()', 'root = args.runtime.resolve()\n    model_source = Path('+repr(model_source)+')')
code=code.replace("root / 'source_port.json'", "model_source.parent / 'source_port.json'")
code=code.replace("root / 'PV-Ground'", 'model_source')
anchor='    embeddings = model.text_encoder.embeddings'
assert code.count(anchor)==1
code=code.replace(anchor,'    from pvground_empty_pool_mask import install_empty_pool_mask\n    install_empty_pool_mask(model,True)\n'+anchor)
anchor="    (args.output/'receipt.json').write_text"
assert code.count(anchor)==1
code=code.replace(anchor,"    receipt.update(empty_pool_mask=True,model_source=str(model_source),module_sha256=sha(Path(__file__).parent/'pvground_empty_pool_mask.py'))\n"+anchor)
code=code.replace('PVG_TRAINING_INTERFACE_PASS','PVG_EMPTY_POOL_TRAINING_INTERFACE_PASS')
argv=['flock','-n','/root/autodl-tmp/mcln_v99_backbone_gpu0.lock',runtime+'/venv/bin/python','-u',root+'/check.py',
      '--runtime',runtime,'--fixtures',fixtures,'--output',root+'/results']
controller=('import hashlib,json,os,subprocess\nfrom pathlib import Path\nroot=Path(__file__).parent\n'
            '(root/"controller.pid").write_text(str(os.getpid())+"\\n")\n'
            'spec=json.loads((root/"spec.json").read_bytes())\n'
            'for name,digest in spec["files"].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name\n'
            'env=os.environ.copy();env.update('+repr(env['env'])+')\n'
            'with (root/"run.log").open("xb") as log:\n'
            '    result=subprocess.run('+repr(argv)+',env=env,stdout=log,stderr=subprocess.STDOUT)\n'
            '(root/"controller.exit").write_text(str(result.returncode)+"\\n")\nraise SystemExit(result.returncode)\n')
files={'check.py':code.encode(),'pvground_empty_pool_mask.py':(repo/'models/pvground_empty_pool_mask.py').read_bytes(),
       'controller.py':controller.encode()}
spec=dict(root=root,runtime=runtime,model_source=model_source,fixtures=fixtures,env_spec_sha256=env_sha,
          base_check_sha256=hashlib.sha256(original).hexdigest(),empty_pool_mask=True,seed=2027,
          eval_forwards=2,train_forwards=2,optimizer_steps=2,new_checkpoints=0,formal_rows=0,
          files={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()})
files['spec.json']=(json.dumps(spec,indent=2)+'\n').encode()
assert root.rsplit('/',1)[1] not in s.listdir('/root/autodl-tmp')
s.mkdir(root);archive.mkdir()
for name,raw in files.items():
    if name.endswith('.py'):compile(raw,name,'exec')
    (archive/name).write_bytes(raw)
    with s.open(root+'/'+name,'wb') as f:f.write(raw)
    with s.open(root+'/'+name,'rb') as f:assert f.read()==raw
command=['screen','-dmS','mcln_pvg_empty_pool_interface_v1','/root/miniconda3/envs/bdetr/bin/python','-u',root+'/controller.py']
_,out,err=c.exec_command(' '.join(map(shlex.quote,command)),timeout=30)
assert out.channel.recv_exit_status()==0,err.read().decode()
pattern='^/root/miniconda3/envs/bdetr/bin/python -u '+root+'/controller.py$'
_,out,err=c.exec_command('pgrep -af '+shlex.quote(pattern),timeout=30)
process=out.read().decode().strip();assert out.channel.recv_exit_status()==0 and process,err.read().decode()
record=dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),process=process,
            root=root,expected_seconds=120,optimizer_steps_planned=2,new_checkpoints=0,formal_rows=0)
raw=(json.dumps(record,indent=2)+'\n').encode();(archive/'launch.json').write_bytes(raw)
with s.open(root+'/launch.json','wb') as f:f.write(raw)
s.close();c.close();print(json.dumps(record),flush=True)
