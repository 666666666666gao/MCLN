"""CPU binding checks on the stored real Nr3D batch; no neural forward."""
import hashlib
import json
import os
from pathlib import Path
import time

root=Path(__file__).parent
batch_root=Path('/root/autodl-tmp/mcln_pvground_nr_native_batch_20260908_v1')
inventory_root=Path('/root/autodl-tmp/mcln_pvground_nr_checkpoint_inspection_20260908_v2')
started=time.time()
batch_receipt=json.loads((batch_root/'receipt.json').read_bytes())
assert batch_receipt['status']=='pass' and (batch_root/'controller.exit').read_text().strip()=='0'
batch_path=batch_root/'native_batch.pt'
assert hashlib.sha256(batch_path.read_bytes()).hexdigest()==batch_receipt['input_sha256']
inventory=json.loads((inventory_root/'state_inventory.json').read_bytes())
strict=json.loads((inventory_root/'strict_load.json').read_bytes())
inventory_receipt=json.loads((inventory_root/'receipt.json').read_bytes())
assert hashlib.sha256((inventory_root/'state_inventory.json').read_bytes()).hexdigest()==inventory_receipt['inventory_sha256']
assert inventory_receipt['checkpoint_sha256']==strict['checkpoint_sha256']
assert strict['status']=='pass' and strict['state_tensors']==1235
import torch
from transformers import RobertaTokenizerFast
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized()
torch.set_num_threads(1)
payload=torch.load(str(batch_path),map_location='cpu')
inputs=payload['inputs'];batch=payload['native_batch']
classes=inputs['det_class_ids']
class_count=inventory['module.butd_class_embeddings.weight']['shape'][0]
assert classes.dtype==torch.long and classes.shape==(8,132)
assert ((classes>=0)&(classes<class_count)).all()
tokenizer=RobertaTokenizerFast.from_pretrained('/root/autodl-tmp/DATA_ROOT_mcln_meshsp/roberta-base/',local_files_only=True)
tokens=tokenizer(inputs['text'],padding='longest',return_tensors='pt',max_length=256,truncation=True)
vocabulary=inventory['module.text_encoder.embeddings.word_embeddings.weight']['shape'][0]
assert tokens.input_ids.shape[0]==8 and tokens.input_ids.shape[1]<=256
assert ((tokens.input_ids>=0)&(tokens.input_ids<vocabulary)).all()
maps=['positive_map','modify_positive_map','pron_positive_map','other_entity_map','rel_positive_map','auxi_entity_positive_map']
lengths=tokens.attention_mask.sum(-1).tolist()
map_records={}
for key in maps:
    value=batch[key]
    assert value.shape[0]==8 and value.shape[-1]==256 and torch.isfinite(value).all()
    assert (value>=0).all()
    outside=[]
    for index,length in enumerate(lengths):
        row=value[index].reshape(-1,256)
        outside.append(dict(nonzero_after_eos_before_slot255=int((row[:,length:255]!=0).sum()),
                            slot255_nonzero=int((row[:,255]!=0).sum())))
    map_records[key]=dict(shape=list(value.shape),outside_encoded_text=outside)
assert batch['gt_masks'].dtype==torch.bool and tuple(inputs['superpoint'].shape)==(8,50000)
assert not torch.cuda.is_initialized()
result=dict(status='pass',scope='real input index bounds and tokenizer/label shapes; not ontology correctness, loss/gradient or model accuracy',
    input_sha256=batch_receipt['input_sha256'],checkpoint_sha256=strict['checkpoint_sha256'],
    state_inventory_sha256=hashlib.sha256((inventory_root/'state_inventory.json').read_bytes()).hexdigest(),
    class_table_rows=class_count,class_id_min=int(classes.min()),class_id_max=int(classes.max()),
    text_vocabulary_rows=vocabulary,text_tensor_shape=list(tokens.input_ids.shape),text_lengths=lengths,
    map_records=map_records,gt_masks_shape=list(batch['gt_masks'].shape),
    model_forwards=0,optimizer_steps=0,formal_rows=0,weights_loaded=0,torch_cuda_initialized=False,
    elapsed_seconds=time.time()-started,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
with (root/'bindings.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
print(json.dumps(result),flush=True)
