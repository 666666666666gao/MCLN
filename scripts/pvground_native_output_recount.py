"""Native saved-output recount reused unchanged from the Scan F formal auditor."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def recount_native_rows(directory):
    receipt=json.loads((directory/'receipt.json').read_bytes())
    assert receipt['status']=='pass'
    for name,suffix in [('rows','.jsonl'),('boxes','.npy'),('scores','.npy')]:
        assert sha(directory/(name+suffix))==receipt[name+'_sha256'],name
    rows=[json.loads(line) for line in (directory/'rows.jsonl').read_text().splitlines()]
    count=len(rows)
    assert count==receipt['rows'] and count>0
    assert len({row['row_id'] for row in rows})==count
    boxes=np.load(str(directory/'boxes.npy'),mmap_mode='r')
    scores=np.load(str(directory/'scores.npy'),mmap_mode='r')
    assert boxes.shape==(count,256,6) and scores.shape==(count,2,256)
    assert boxes.dtype==scores.dtype==np.dtype('float32')
    assert np.isfinite(boxes).all() and np.isfinite(scores).all()
    maximum_error=0.
    for index,row in enumerate(rows):
        gt=np.asarray(row['root_box'],dtype=np.float64)
        assert gt.shape==(6,) and np.isfinite(gt).all() and (gt[3:]>0).all()
        candidates=np.asarray(boxes[index],dtype=np.float64).copy()
        candidates[:,3:]=np.maximum(candidates[:,3:],1e-6)
        lo=np.maximum(candidates[:,:3]-candidates[:,3:]/2,gt[:3]-gt[3:]/2)
        hi=np.minimum(candidates[:,:3]+candidates[:,3:]/2,gt[:3]+gt[3:]/2)
        intersection=np.maximum(hi-lo,0).prod(axis=-1)
        ious=intersection/(candidates[:,3:].prod(axis=-1)+gt[3:].prod()-intersection)
        assert np.isfinite(ious).all()
        for mode_index,mode in enumerate(['bbs','bbf']):
            selected=row[mode];query=selected['query']
            assert isinstance(query,int) and 0<=query<256
            assert scores[index,mode_index,query]==scores[index,mode_index].max()
            assert np.allclose(np.asarray(selected['box']),candidates[query],rtol=0,atol=1e-12)
            error=abs(float(ious[query])-selected['iou'])
            maximum_error=max(maximum_error,error)
            assert error<1e-5,(row['row_id'],mode,error)
            for key in ['iou','mask_iou']:
                assert math.isfinite(selected[key]) and 0<=selected[key]<=1
    metrics={}
    for mode in ['bbs','bbf']:
        mask_sum=sum(row[mode]['mask_iou'] for row in rows)
        metrics[mode]={'rec_hits25':sum(row[mode]['iou']>.25 for row in rows),
            'rec_hits50':sum(row[mode]['iou']>.5 for row in rows),
            'mask_hits25':sum(row[mode]['mask_iou']>.25 for row in rows),
            'mask_hits50':sum(row[mode]['mask_iou']>.5 for row in rows),
            'mask_iou_sum':mask_sum,'mask_miou':mask_sum/count*100.}
    assert metrics==receipt['metrics']
    return rows,metrics,{'rows':count,'candidate_boxes_checked':count*256,
                        'selection_checks':count*2,'max_selected_iou_error':maximum_error}
