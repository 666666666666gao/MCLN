"""Check real stored Nr3D target tensors against the pinned native loss input contract."""
import hashlib
import json
import os
from pathlib import Path
import time

started=time.time()
root=Path(__file__).parent
batch_root=Path('/root/autodl-tmp/mcln_pvground_nr_native_batch_20260908_v1')
source=Path('/root/autodl-tmp/mcln_pvground_vsa_order_source_20260908_v1/PV-Ground')
receipt=json.loads((batch_root/'receipt.json').read_bytes())
path=batch_root/'native_batch.pt'
assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['input_sha256']
assert os.environ['CUDA_VISIBLE_DEVICES']==''
import torch
torch.set_num_threads(1)
payload=torch.load(str(path),map_location='cpu')
batch=payload['native_batch']
required=['center_label','size_gts','sem_cls_label','gt_masks','positive_map',
          'modify_positive_map','pron_positive_map','other_entity_map','rel_positive_map',
          'box_label_mask','auxi_entity_positive_map','auxi_box','point_instance_label','language_dataset']
assert all(key in batch for key in required)
loss_text=(source/'models/losses.py').read_text()
assert "target = targets[bs]['masks'][idx1].float()" in loss_text
assert 'torch.cdist(out_masks, tgt_masks.float(), p=1)' in loss_text
assert 'if end_points["language_dataset"][0] == "scanrefer":' in loss_text
rows=[]
for i in range(8):
    valid=batch['box_label_mask'][i].bool()
    labels=batch['sem_cls_label'][i,valid]
    centers=batch['center_label'][i,valid,:3]
    sizes=batch['size_gts'][i,valid]
    assert len(labels)>0 and centers.shape==sizes.shape==(len(labels),3)
    assert labels.dtype==torch.long and ((labels>=0)&(labels<256)).all()
    assert torch.isfinite(centers).all() and torch.isfinite(sizes).all() and (sizes>0).all()
    masks=batch['gt_masks'][i,valid]
    assert masks.dtype==torch.bool and masks.shape==(len(labels),50000)
    assert batch['point_instance_label'][i].shape==(50000,)
    maps={key:list(batch[key][i,valid].shape) for key in required if key.endswith('_map') and key!='auxi_entity_positive_map'}
    assert all(shape==[len(labels),256] for shape in maps.values())
    rows.append(dict(index=i,dataset=batch['language_dataset'][i],valid_targets=len(labels),
                     semantic_label_min=int(labels.min()),semantic_label_max=int(labels.max()),
                     minimum_target_size=float(sizes.min()),
                     empty_target_masks=int((masks.sum(-1)==0).sum()),
                     target_mask_points=masks.sum(-1).tolist(),
                     auxiliary_box_shape=list(batch['auxi_box'][i].shape)))
assert not torch.cuda.is_initialized()
record=dict(status='pass',scope='real target selection, shape and finite-value contract only; no criterion execution or gradient evidence',
            rows=rows,input_sha256=receipt['input_sha256'],required_keys=required,
            sources={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ['models/losses.py','main_utils.py']},
            native_mask_conversion='selected target.float() before scatter_mean; matcher also casts before cdist',
            native_batch_classification_weight=0.5 if batch['language_dataset'][0]=='scanrefer' else 1.0,
            loss_prefixes=['proposal_','last_','0head_','1head_','2head_','3head_','4head_'],
            model_forwards=0,criterion_calls=0,optimizer_steps=0,formal_rows=0,weights_loaded=0,
            elapsed_seconds=time.time()-started,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
with (root/'loss_inputs.json').open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
print(json.dumps(record),flush=True)
