"""CPU stage probe: the exact frozen MLP response to an observed empty group."""
import hashlib
import json
import os
from pathlib import Path
import sys
import torch

assert not torch.cuda.is_initialized()
torch.set_num_threads(1)
runtime = Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1')
source = Path('/root/autodl-tmp/mcln_pvground_vsa_order_source_20260908_v1/PV-Ground')
evidence = Path('/root/autodl-tmp/mcln_pvground_fit_support_20260908_v2')
assert (evidence / 'controller.exit').read_text().strip() == '0'
receipt = json.loads((evidence / 'receipt.json').read_bytes())
assert receipt['status'] == 'complete' and receipt['checkpoint_unchanged']
env = json.loads((runtime / 'env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(env, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == receipt['env_spec_sha256']
os.chdir(str(source)); sys.path.insert(0, str(source))
sys.path.insert(0, str(runtime / 'OpenPCDet'))
sys.path.insert(0, str(runtime / 'PV-Ground/pointnet2'))
from models.pv_ground import PVGround
from pcdet.config import cfg, cfg_from_yaml_file
parent = Path(env['weight_dirs']['scanrefer']['path'])
assert hashlib.sha256(parent.read_bytes()).hexdigest() == receipt['checkpoint_sha256']
payload = torch.load(str(parent), map_location='cpu')
config = payload['config']; state = {k[7:]: v for k, v in payload['model'].items()}
cfg_from_yaml_file(str(source / 'wandb_config.yaml'), cfg)
model = PVGround(cfg, num_class=256, num_queries=256, num_decoder_layers=6,
                 self_position_embedding=config.self_position_embedding, contrastive_align_loss=True,
                 butd=True, pointnet_ckpt=None, data_path='/root/autodl-tmp/DATA_ROOT_mcln_meshsp/', self_attend=config.self_attend)
assert set(model.state_dict()) - set(state) == {'text_encoder.embeddings.position_ids'}
model.text_encoder.embeddings.register_buffer('position_ids', model.text_encoder.embeddings.position_ids, persistent=False)
model.load_state_dict(state, strict=True); model.eval().requires_grad_(False)
vsa = model.backbone_net.vsa
sources = [('raw_points', vsa.SA_rawpoints)] + list(zip(vsa.SA_layer_names, vsa.SA_layers))
results = []
for name, module in sources:
    for grouper, mlp in zip(module.groupers, module.mlps):
        zeros = torch.zeros(1, mlp[0].in_channels, 1, grouper.nsample)
        with torch.no_grad(): value = mlp(zeros).max(dim=-1)[0].reshape(-1)
        assert torch.isfinite(value).all()
        results.append(dict(source=name, radius_m=grouper.radius, channels=len(value),
                            nonzero_channels=int((value != 0).sum()), max_abs=float(value.abs().max()),
                            mean_abs=float(value.abs().mean()), l2=float(value.norm()), values=value.tolist()))
assert all(torch.equal(value, state[name]) for name, value in model.state_dict().items())
result = dict(scope='CPU MLP-stage zero-input probe under frozen eval BN; not train-mode activation or a full scene/model forward',
              parent_sha256=receipt['checkpoint_sha256'], fit_receipt_sha256=hashlib.sha256((evidence / 'receipt.json').read_bytes()).hexdigest(),
              state_unchanged=True, full_model_forwards=0, mlp_stage_forwards=10, optimizer_steps=0,
              torch_cuda_initialized=torch.cuda.is_initialized(), sources=results)
print(json.dumps(result, allow_nan=False))
