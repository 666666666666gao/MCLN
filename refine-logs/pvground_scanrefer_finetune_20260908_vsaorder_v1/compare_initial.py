"""Compare fixed pre-update ScanRefer outputs before/after the VSA interface correction."""
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

old=Path('/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260908_detalign_v1')
new=Path('/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260908_vsaorder_v1')
roots=[old,new]
specs=[json.loads((p/'spec.json').read_bytes()) for p in roots]
for key in ['checkpoint_sha256','seed','batch_size','input_manifest','env_spec_sha256']:
    assert specs[0][key]==specs[1][key],key
receipts=[json.loads((p/'initial/receipt.json').read_bytes()) for p in roots]
rows=[]
for root,receipt in zip(roots,receipts):
    assert receipt['status']=='pass' and receipt['rows']==6887 and receipt['formal_rows']==0
    path=root/'initial/rows.jsonl'
    assert sha(path)==receipt['rows_sha256']
    rows.append([json.loads(line) for line in path.read_text().splitlines()])
identity={key:sum(a[key]==b[key] for a,b in zip(*rows)) for key in ['row_id','scan_id','target_id','point_sha256','root_box']}
assert all(count==6887 for count in identity.values()),identity
result=dict(status='complete',time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    roots=[str(p) for p in roots],initial_receipt_sha256=[sha(p/'initial/receipt.json') for p in roots],
    identical_row_fields=identity,metrics=[r['metrics'] for r in receipts],transitions={},oracle={})
for mode in ['bbs','bbf']:
    values=[]
    for threshold in [.25,.5]:
        a=np.array([r[mode]['iou']>threshold for r in rows[0]])
        b=np.array([r[mode]['iou']>threshold for r in rows[1]])
        values.append(dict(threshold=threshold,old_hits=int(a.sum()),new_hits=int(b.sum()),
            fixes=int((~a&b).sum()),breaks=int((a&~b).sum()),net=int(b.sum()-a.sum())))
    result['transitions'][mode]=values
for root,receipt,rowset in zip(roots,receipts,rows):
    path=root/'initial/boxes.npy';assert sha(path)==receipt['boxes_sha256']
    boxes=np.load(str(path),mmap_mode='r');assert boxes.shape==(6887,256,6)
    hits={'.25':0,'.50':0}
    for box,row in zip(boxes,rowset):
        box=box.astype(np.float64);box[:,3:]=np.maximum(box[:,3:],1e-6)
        gt=np.array(row['root_box'],dtype=np.float64)
        lo=np.maximum(box[:,:3]-box[:,3:]/2,gt[:3]-gt[3:]/2)
        hi=np.minimum(box[:,:3]+box[:,3:]/2,gt[:3]+gt[3:]/2)
        intersection=np.maximum(hi-lo,0).prod(-1)
        iou=intersection/(box[:,3:].prod(-1)+gt[3:].prod()-intersection)
        hits['.25']+=int(iou.max()>.25);hits['.50']+=int(iou.max()>.5)
    result['oracle'][root.name]=dict(full256_hits=hits,gt_diagnostic_only=True)
result.update(model_forwards=0,optimizer_steps=0,formal_rows=0,
    scope='6887 backbone-seen train-scene holdout; pre-update arithmetic, no formal gain claim; old det/text per-row tensors not exported')
destination=new/'initial_vsa_comparison.json'
with destination.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
