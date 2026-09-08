"""CPU inspection of the exact official Sr3D checkpoint, without a model."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import torch

root=Path(__file__).resolve().parent
plan=json.loads((root/'plan.json').read_bytes())
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized()
torch.set_num_threads(1)
path=root/plan['filename'];digest=hashlib.sha256()
with path.open('rb') as stream:
    for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
assert digest.hexdigest()==plan['sha256'] and path.stat().st_size==plan['bytes']
payload=torch.load(str(path),map_location='cpu')
state=payload['model'];config=payload['config']
inventory={}
for name,value in state.items():
    assert isinstance(value,torch.Tensor) and value.device.type=='cpu' and torch.isfinite(value).all(),name
    inventory[name]={'shape':list(value.shape),'dtype':str(value.dtype),'elements':value.numel()}
scan_path=Path('/root/autodl-tmp/mcln_pvground_scan_checkpoint_inspection_20260908_v1/state_inventory.json')
scan=json.loads(scan_path.read_bytes())
shared=set(scan)&set(inventory)
shape_differences={name:{'scan':scan[name],'sr':inventory[name]} for name in sorted(shared) if scan[name]!=inventory[name]}
config_names=['dataset','test_dataset','butd','butd_cls','butd_gt','num_queries','num_decoder_layers',
    'self_position_embedding','self_attend','use_soft_token_loss','use_contrastive_align',
    'joint_det','detect_intermediate','use_color','use_height','use_multiview','frozen','small_lr',
    'lr','lr_backbone','weight_decay','clip_norm','query_points_obj_topk']
selected={name:getattr(config,name) for name in config_names if hasattr(config,name)}
raw=(json.dumps(inventory,indent=2,sort_keys=True)+'\n').encode();(root/'state_inventory.json').write_bytes(raw)
receipt=dict(status='complete',time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    checkpoint_sha256=digest.hexdigest(),bytes=path.stat().st_size,epoch=payload['epoch'],
    payload_keys=sorted(payload),config=selected,model_tensor_count=len(state),
    inventory_sha256=hashlib.sha256(raw).hexdigest(),scan_inventory_sha256=hashlib.sha256(scan_path.read_bytes()).hexdigest(),
    same_state_schema_as_scan=(inventory==scan),shape_differences=shape_differences,
    effective_object_stream=bool(config.butd or config.butd_gt or config.butd_cls),
    object_protocol_matches_planned_butd_cls=bool(not config.butd and config.butd_cls and not config.butd_gt),
    sr_only_keys=sorted(set(inventory)-set(scan)),scan_only_keys=sorted(set(scan)-set(inventory)),
    model_instantiated=False,strict_load_tested=False,model_forwards=0,optimizer_steps=0,formal_rows=0,
    torch_version=torch.__version__,torch_cuda_initialized=torch.cuda.is_initialized())
assert not receipt['torch_cuda_initialized']
(root/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print('SR_CPU_INVENTORY_COMPLETE '+json.dumps(receipt),flush=True)
