"""Strict-load the prepared Nr3D parent into the current PV-Ground on CPU."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


root=Path(__file__).resolve().parent
runtime=Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1')
source=Path('/root/autodl-tmp/mcln_pvground_vsa_order_source_20260908_v1/PV-Ground')
plan=json.loads((root/'plan.json').read_bytes())
receipt=json.loads((root/'receipt.json').read_bytes())
assert receipt['status']=='complete' and (root/'inventory.exit').read_text().strip()=='0'
env=json.loads((runtime/'env_spec.json').read_bytes())
env_sha=hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert env_sha=='966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c'
upstream=json.loads((runtime/'source_bundle_receipt.json').read_bytes())['sources']['PV-Ground']
port=json.loads((source.parent/'source_port.json').read_bytes())
for name,entry in upstream['files'].items():
    expected=port['after_sha256'] if 'PV-Ground/'+name==port['file'] else entry['sha256']
    assert sha(source/name)==expected,name
assert os.environ['CUDA_VISIBLE_DEVICES']==''
import torch
torch.set_num_threads(1);torch.manual_seed(2027)
assert not torch.cuda.is_initialized()
os.chdir(str(source));sys.path.insert(0,str(source))
from models.pv_ground import PVGround
from pcdet.config import cfg,cfg_from_yaml_file
assert Path(sys.modules['models.pv_ground'].__file__).resolve()==source/'models/pv_ground.py'
parent=root/plan['filename'];assert sha(parent)==plan['sha256']
payload=torch.load(str(parent),map_location='cpu');config=payload['config']
assert all(name.startswith('module.') for name in payload['model'])
state={name[len('module.'):]:value for name,value in payload['model'].items()}
cfg_from_yaml_file(str(runtime/'PV-Ground/wandb_config.yaml'),cfg)
started=time.time()
model=PVGround(cfg,num_class=256,num_queries=256,num_decoder_layers=6,
    self_position_embedding=config.self_position_embedding,contrastive_align_loss=True,
    butd=config.butd or config.butd_gt or config.butd_cls,pointnet_ckpt=None,data_path='/root/autodl-tmp/DATA_ROOT_mcln_meshsp/',
    self_attend=config.self_attend)
# The Nr3D parent actually stores this deterministic buffer; keep it persistent.
position_ids=torch.arange(model.text_encoder.config.max_position_embeddings).expand((1,-1))
assert set(model.state_dict())==set(state)
assert torch.equal(model.text_encoder.embeddings.position_ids,position_ids)
assert torch.equal(state['text_encoder.embeddings.position_ids'],position_ids)
result=model.load_state_dict(state,strict=True)
assert not result.missing_keys and not result.unexpected_keys
assert all(value.device.type=='cpu' and torch.equal(value,state[name]) for name,value in model.state_dict().items())
assert not torch.cuda.is_initialized()
record=dict(status='pass',checkpoint_sha256=plan['sha256'],env_spec_sha256=env_sha,
    model_source=str(source),model_source_port_sha256=sha(source.parent/'source_port.json'),
    state_tensors=len(state),loaded_tensors_exact=True,missing_keys=[],unexpected_keys=[],
    model_instantiated=True,model_forwards=0,optimizer_steps=0,formal_rows=0,
    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
    trainable_tensors=sum(p.requires_grad for p in model.parameters()),
    frozen_tensors=sum(not p.requires_grad for p in model.parameters()),
    cpu_only=True,torch_cuda_initialized=False,elapsed_seconds=time.time()-started,
    script_sha256=sha(__file__))
(root/'strict_load.json').write_text(json.dumps(record,indent=2)+'\n')
print('NR_CPU_STRICT_LOAD_PASS '+json.dumps(record),flush=True)
