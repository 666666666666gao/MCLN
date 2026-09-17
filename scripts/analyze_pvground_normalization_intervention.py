"""Recompute fixed BN-intervention IoUs from exported raw boxes and dataset GT."""
import hashlib
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[1]/'refine-logs/pvground_normalization_intervention_20260917_v1'
result=json.loads((root/'diagnostic.json').read_bytes())
assert result['status']=='complete' and (root/'controller.exit').read_text().strip()=='0'
for filename,key in [('rows.json','rows_sha256'),('batches.json','batches_sha256'),('input_selection.json','input_selection_sha256'),('candidate_values.npz','arrays_sha256')]:
    assert hashlib.sha256((root/filename).read_bytes()).hexdigest()==result[key]
rows=json.loads((root/'rows.json').read_bytes())
batches=json.loads((root/'batches.json').read_bytes())
chosen=json.loads((root/'input_selection.json').read_bytes())['rows']
assert len(rows)==128 and len(batches)==16
assert [r['row_id'] for r in rows]==[r['row_id'] for r in chosen]
assert [r['row_id'] for r in rows]==[i for b in batches for i in b['rows']]
assert len({r['scan_id'].split('_')[0] for r in rows})==128
arrays=np.load(root/'candidate_values.npz')
assert len(arrays.files)==128*4
def corners(box):
    half=np.maximum(box[...,3:],np.float32(1e-6))*np.float32(.5)
    return box[...,:3]-half,box[...,:3]+half
def iou(box,gt):
    lo,hi=corners(box);glo,ghi=corners(gt)
    sizes=np.maximum(np.minimum(hi,ghi)-np.maximum(lo,glo),np.float32(0))
    inter=sizes[:,0]*sizes[:,1]*sizes[:,2]
    v=hi-lo;g=ghi-glo
    union=v[:,0]*v[:,1]*v[:,2]+g[0]*g[1]*g[2]-inter
    return inter/union
max_iou_error=0.
for row in rows:
    key=str(row['row_id']);gt=arrays[key+'_gt']
    assert gt.dtype==np.float32 and np.array_equal(gt,np.asarray(row['root_box'],dtype=np.float32))
    for arm in ['normal','repeat','parent_bn']:
        values=arrays[key+'_'+arm]
        assert values.shape==(256,8) and values.dtype==np.float32 and np.isfinite(values).all()
        measured=iou(values[:,:6],gt)
        error=float(np.max(np.abs(measured-values[:,7])))
        max_iou_error=max(max_iou_error,error)
        assert np.allclose(measured,values[:,7],rtol=1e-5,atol=1e-6)
        choice=row[arm]['selected']
        assert values[choice,6]==values[:,6].max()
        assert values[choice,7]==row[arm]['selected_iou']
        assert values[:,7].max()==row[arm]['raw_oracle_iou']
        for t in [.25,.5]:
            assert (measured[choice]>t)==(row[arm]['selected_iou']>t)
summary={}
for arm in ['normal','repeat','parent_bn']:
    summary[arm]={str(t):dict(hits=sum(r[arm]['selected_iou']>t for r in rows),
        fixes=sum(r['normal']['selected_iou']<=t<r[arm]['selected_iou'] for r in rows),
        breaks=sum(r[arm]['selected_iou']<=t<r['normal']['selected_iou'] for r in rows)) for t in [.25,.5]}
assert summary==result['summary']
diffs={arm:dict(selection_changes=sum(r[arm]['selected']!=r['normal']['selected'] for r in rows),
    box_max_abs=max(float(np.max(np.abs(arrays[str(r['row_id'])+'_'+arm][:,:6]-arrays[str(r['row_id'])+'_normal'][:,:6]))) for r in rows),
    score_max_abs=max(float(np.max(np.abs(arrays[str(r['row_id'])+'_'+arm][:,6]-arrays[str(r['row_id'])+'_normal'][:,6]))) for r in rows)) for arm in ['repeat','parent_bn']}
assert diffs==result['differences']
geometry={arm:dict(raw_nonpositive_size_candidates=sum(int(np.any(arrays[str(r['row_id'])+'_'+arm][:,3:6]<=0,axis=1).sum()) for r in rows),
    selected_iou_mean=float(np.mean([r[arm]['selected_iou'] for r in rows])),
    raw_oracle_hits={str(t):sum(r[arm]['raw_oracle_iou']>t for r in rows) for t in [.25,.5]}) for arm in ['normal','repeat','parent_bn']}
out=dict(status='pass_exported_box_gt_recount',rows=128,candidate_rows=128*256*3,max_iou_error=max_iou_error,
    summary=summary,differences=diffs,geometry=geometry,diagnostic_sha256=hashlib.sha256((root/'diagnostic.json').read_bytes()).hexdigest(),
    scope='Recomputed native-clamped IoU from exported raw boxes and dataset GT; no independent network rerun or annotation-source reparse')
(root/'recount.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,indent=2))
