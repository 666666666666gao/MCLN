"""Read one atomically saved live checkpoint on CPU, without changing training."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

formal=Path('/root/autodl-tmp/mcln_pvground_scanrefer_formal_20260909_observation_v1')
training=Path('/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260909_observation_v1')
sys.path.insert(0,str(formal))
from evaluate import expanded_parent_state, terminal_state, sha, now

formal_spec=json.loads((formal/'spec.json').read_bytes())
for name,digest in formal_spec['files'].items():
    assert sha(formal/name)==digest,name
assert sha(training/'spec.json')==formal_spec['training_spec_sha256']
spec=json.loads((training/'spec.json').read_bytes())
runtime=Path(spec['runtime'])
environment=json.loads((runtime/'env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(environment,sort_keys=True,separators=(',',':')).encode()).hexdigest()==spec['env_spec_sha256']
assert sha(spec['source_port'])==spec['source_port_sha256']
source=Path(spec['model_source'])
for name,digest in json.loads(Path(spec['source_port']).read_bytes())['files'].items():
    assert sha(source/name)==digest,name
assert os.environ['CUDA_VISIBLE_DEVICES']==''
import torch
torch.set_num_threads(1);torch.manual_seed(2027)
assert not torch.cuda.is_initialized()
os.chdir(str(source));sys.path.insert(0,str(source))
from models.pv_ground import PVGround
from pcdet.config import cfg,cfg_from_yaml_file
checkpoint=environment['weight_dirs']['scanrefer']
assert sha(checkpoint['path'])==checkpoint['sha256']==spec['checkpoint_sha256']
parent_payload=torch.load(checkpoint['path'],map_location='cpu')
config=parent_payload['config']
assert all(name.startswith('module.') for name in parent_payload['model'])
parent={name[7:]:value for name,value in parent_payload['model'].items()}
cfg_from_yaml_file(str(runtime/'PV-Ground/wandb_config.yaml'),cfg)
model=PVGround(cfg,num_class=256,num_queries=256,num_decoder_layers=6,
    self_position_embedding=config.self_position_embedding,contrastive_align_loss=True,butd=True,
    pointnet_ckpt=None,data_path='/root/autodl-tmp/DATA_ROOT_mcln_meshsp/',self_attend=config.self_attend)
initial=expanded_parent_state(model,parent)
added=set(initial)-set(parent)
begin=time.time()
# The live writer uses tmp + os.replace. One open descriptor pins a complete inode.
with (training/'latest.pth').open('rb') as stream:
    size=os.fstat(stream.fileno()).st_size
    digest=hashlib.sha256()
    for block in iter(lambda:stream.read(8*1024*1024),b''):
        digest.update(block)
    stream.seek(0)
    data=torch.load(stream,map_location='cpu')
assert 0<data['step']<3723 and data['step']%512==0
assert data['parent_checkpoint_sha256']==checkpoint['sha256']
assert data['spec_sha256']==sha(training/'spec.json')
for key in ['source_query_read','source_query_module_sha256','source_port_sha256','observation_state','observation_module_sha256']:
    assert data[key]==spec[key],key
assert sha(formal/'pvground_source_query.py')==data['source_query_module_sha256']
assert sha(formal/'pvground_observation_query.py')==data['observation_module_sha256']
assert len(added)==36
state=terminal_state(model,initial,data['state_delta'])
model.load_state_dict(state,strict=True)
assert all(torch.equal(value,state[name]) for name,value in model.state_dict().items())
logs=[json.loads(line) for line in (training/'train.jsonl').read_text().splitlines()[:data['step']]]
assert [row['step'] for row in logs]==list(range(1,data['step']+1))
assert data['row_ids']==[row_id for row in logs for row_id in row['rows']]
assert len(data['row_ids'])==data['step']*8 and len(set(data['row_ids']))==len(data['row_ids'])
manifest=json.loads(Path(spec['input_manifest']).read_bytes())
partitions=json.loads(Path(manifest['split_protocol']).read_bytes())['row_ids']
assert set(data['row_ids'])<=set(partitions['fit'])
assert not set(data['row_ids']).intersection(partitions['holdout'])
optimizer=data['optimizer']
parameter_ids=[pid for group in optimizer['param_groups'] for pid in group['params']]
assert len(parameter_ids)==819 and len(set(parameter_ids))==819
assert set(optimizer['state'])<=set(parameter_ids)
assert all(float(entry['step'])<=data['step'] for entry in optimizer['state'].values())
assert all(torch.isfinite(entry[key]).all() for entry in optimizer['state'].values() for key in ['exp_avg','exp_avg_sq'])
changes={name:float((state[name]-initial[name]).abs().max()) for name in sorted(added)}
assert not torch.cuda.is_initialized()
result=dict(status='pass',time_cst=now(),step=data['step'],fit_rows=len(data['row_ids']),
    checkpoint_snapshot_sha256=digest.hexdigest(),checkpoint_snapshot_bytes=size,
    checkpoint_path=str(training/'latest.pth'),snapshot_open_descriptor=True,
    delta_tensors=len(data['state_delta']),expanded_tensors=len(state),added_tensors=len(added),
    added_max_absolute_change=changes,optimizer_parameter_ids=len(parameter_ids),
    optimizer_state_entries=len(optimizer['state']),optimizer_state_finite=True,
    exact_fit_log_prefix=True,strict_restore_exact=True,cpu_only=True,
    torch_cuda_initialized=False,model_forwards=0,optimizer_steps=0,new_checkpoint_files=0,formal_rows=0,
    elapsed_seconds=time.time()-begin,module_sha256=data['observation_module_sha256'],source_query_module_sha256=data['source_query_module_sha256'],
    source_port_sha256=data['source_port_sha256'],training_spec_sha256=data['spec_sha256'],
    scope='actual observed intermediate snapshot, not terminal quality or resume-equivalence test')
print('PVG_OBSERVATION_LATEST_CPU_PASS '+json.dumps(result),flush=True)
