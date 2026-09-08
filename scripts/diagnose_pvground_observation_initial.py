"""Quantify actual B/C initial export differences without model execution."""
import datetime,hashlib,json
from pathlib import Path
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

roots=[Path('/root/autodl-tmp/mcln_pvground_scanrefer_finetune_'+s) for s in
       ['20260908_sourcequery_v1','20260909_observation_v1']]
receipts=[json.loads((r/'initial/receipt.json').read_bytes()) for r in roots]
configs=[json.loads((r/'spec.json').read_bytes()) for r in roots]
for key in ['seed','batch_size','input_manifest','checkpoint_sha256','runtime','env_spec_sha256','primary_mode','fit_passes','lr','lr_backbone']:
    assert configs[0][key]==configs[1][key],key
hashes=[]
for root,receipt in zip(roots,receipts):
    assert receipt['status']=='pass' and receipt['rows']==6887 and receipt['formal_rows']==0
    actual={n:sha(root/'initial'/n) for n in ['rows.jsonl','boxes.npy','scores.npy']}
    for n,k in [('rows.jsonl','rows'),('boxes.npy','boxes'),('scores.npy','scores')]:assert actual[n]==receipt[k+'_sha256']
    hashes.append(actual)
rows=[[json.loads(line) for line in (r/'initial/rows.jsonl').read_text().splitlines()] for r in roots]
same={k:sum(a[k]==b[k] for a,b in zip(*rows)) for k in ['row_id','scan_id','target_id','point_sha256','root_box']}
assert len(rows[0])==len(rows[1])==6887 and all(n==6887 for n in same.values())
result=dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    status='quantified_export_comparison',roots=list(map(str,roots)),identical_row_fields=same,
    export_sha256=hashes,byte_exact=hashes[0]==hashes[1],metrics=[r['metrics'] for r in receipts],
    arrays={},transitions={},model_forwards=0,optimizer_steps=0,formal_rows=0,
    claim='Same inputs and protocol checked; differences measured, cause not established; no baseline equivalence claim')
for name in ['boxes.npy','scores.npy']:
    a,b=[np.load(str(r/'initial'/name),mmap_mode='r') for r in roots]
    assert a.shape==b.shape and a.dtype==b.dtype and np.isfinite(a).all() and np.isfinite(b).all()
    d=np.abs(a.astype(np.float64)-b.astype(np.float64))
    result['arrays'][name]=dict(shape=list(a.shape),dtype=str(a.dtype),max_abs=float(d.max()),
        mean_abs=float(d.mean()),different_elements=int((d!=0).sum()),
        rows_with_difference=int(np.any(d.reshape(6887,-1)!=0,axis=1).sum()),
        elements_over_1e_5=int((d>1e-5).sum()),elements_over_1e_4=int((d>1e-4).sum()),
        row_max_quantiles=np.quantile(d.reshape(6887,-1).max(axis=1),[.5,.9,.99,1]).tolist())
for mode in ['bbs','bbf']:
    result['transitions'][mode]={}
    for t,k in [(.25,'rec_hits25'),(.5,'rec_hits50')]:
        a=np.asarray([r[mode]['iou']>t for r in rows[0]])
        b=np.asarray([r[mode]['iou']>t for r in rows[1]])
        assert int(a.sum())==receipts[0]['metrics'][mode][k] and int(b.sum())==receipts[1]['metrics'][mode][k]
        result['transitions'][mode][str(t)]=dict(fixes=int((~a&b).sum()),breaks=int((a&~b).sum()),net=int(b.sum()-a.sum()))
    result['transitions'][mode]['iou_max_abs']=max(abs(a[mode]['iou']-b[mode]['iou']) for a,b in zip(*rows))
    result['transitions'][mode]['changed_iou_rows']=sum(a[mode]['iou']!=b[mode]['iou'] for a,b in zip(*rows))
    changed=[(a,b) for a,b in zip(*rows) if a[mode]['query']!=b[mode]['query']]
    result['transitions'][mode]['changed_selected_query_rows']=len(changed)
    result['transitions'][mode]['changed_query_cases']=[dict(row_id=a['row_id'],scan_id=a['scan_id'],
        before=a[mode],after=b[mode]) for a,b in changed]
    result['transitions'][mode]['selected_box_max_abs']=max(float(np.max(np.abs(np.asarray(a[mode]['box'])-np.asarray(b[mode]['box'])))) for a,b in zip(*rows))
result['row_schema']={k:list(v) if isinstance(v,dict) else type(v).__name__ for k,v in rows[0][0].items()}
print(json.dumps(result))
