"""Rehash the two completed initial exports without touching live training."""
import datetime
import hashlib
import json
from pathlib import Path


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


base=Path('/root/autodl-tmp')
roots=[base/'mcln_pvground_scanrefer_finetune_20260908_sourcequery_v1',
       base/'mcln_pvground_scanrefer_finetune_20260909_observation_v1']
receipts=[];hashes=[];specs=[]
for root in roots:
    receipt=json.loads((root/'initial/receipt.json').read_bytes())
    assert receipt['status']=='pass' and receipt['rows']==6887 and receipt['formal_rows']==0
    actual={name:sha(root/'initial'/name) for name in ['rows.jsonl','boxes.npy','scores.npy']}
    for name,suffix in [('rows.jsonl','rows'),('boxes.npy','boxes'),('scores.npy','scores')]:
        assert actual[name]==receipt[suffix+'_sha256'],name
    receipts.append(receipt);hashes.append(actual)
    specs.append(json.loads((root/'spec.json').read_bytes()))
for key in ['seed','batch_size','input_manifest','checkpoint_sha256','runtime','env_spec_sha256','primary_mode','fit_passes','lr','lr_backbone']:
    assert specs[0][key]==specs[1][key],key
assert hashes[0]==hashes[1]
assert receipts[0]['metrics']==receipts[1]['metrics']
result=dict(status='exact_export_match',time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    roots=[str(root) for root in roots],rows=6887,raw_candidate_boxes=6887*256,
    primary='bbs',metrics=receipts[1]['metrics'],export_sha256=hashes[1],
    initial_receipt_sha256=[sha(root/'initial/receipt.json') for root in roots],
    training_spec_sha256=[sha(root/'spec.json') for root in roots],
    model_forwards=0,optimizer_steps=0,formal_rows=0,new_checkpoint_files=0,
    claim='saved candidate boxes, bbs/bbf scores and per-row selected REC/Mask records exactly equal; not full internal tensors or per-point Mask probabilities')
print(json.dumps(result))
