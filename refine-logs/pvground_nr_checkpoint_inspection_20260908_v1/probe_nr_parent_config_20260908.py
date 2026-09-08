import json,os
from pathlib import Path
import torch
torch.set_num_threads(1)
assert not torch.cuda.is_initialized()
root=Path('/root/autodl-tmp/mcln_pvground_nr_checkpoint_inspection_20260908_v1')
p=torch.load(str(root/'PV-Ground_NR3D.pth'),map_location='cpu')
c=p['config']
names=['dataset','test_dataset','butd','butd_cls','butd_gt','num_queries','num_decoder_layers',
'self_position_embedding','self_attend','use_soft_token_loss','use_contrastive_align',
'joint_det','detect_intermediate','frozen','small_lr','lr','lr_backbone','weight_decay','clip_norm']
record={'saved_config':{n:getattr(c,n) for n in names if hasattr(c,n)},'epoch':p['epoch'],
'state_tensor_count':len(p['model']),'payload_keys':sorted(p),'torch_cuda_initialized':torch.cuda.is_initialized()}
print(json.dumps(record))
