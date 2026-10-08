"""Recount saved native parent predictions and distinguish protocol drift from gains."""
import hashlib
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
actual=root/'preflight_r3/formal_actual'
intake=json.loads((actual/'INTAKE.json').read_bytes())
assert intake['exitcode']==0
for entry in intake['files']:
    path=actual/entry['name']
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    assert path.stat().st_size==entry['bytes'] and h.hexdigest()==entry['sha256']
receipt=json.loads((actual/'formal_receipt.json').read_bytes())
assert receipt['status']=='COMPLETED' and receipt['formal_accuracy_result']
assert [r['batch'] for r in receipt['receipts']]==[8,24]
results={}
inputs={}
arrays={}
for batch in (8,24):
    folder=actual/'formal'/('batch'+str(batch))
    rows=[json.loads(line) for line in (folder/'rows.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==9508 and [r['row_id'] for r in rows]==list(range(9508))
    native=json.loads((folder/'receipt.json').read_bytes())
    assert native['rows']==9508 and native['weight_state_unchanged'] and native['model_state_tensors']==1234
    boxes=np.load(folder/'boxes.npy',mmap_mode='r')
    scores=np.load(folder/'scores.npy',mmap_mode='r')
    assert boxes.shape==(9508,256,6) and scores.shape==(9508,2,256)
    assert np.isfinite(boxes).all() and np.isfinite(scores).all()
    truth=np.asarray([r['root_box'] for r in rows],dtype=np.float32)
    prediction=np.asarray(boxes).copy()
    prediction[:,:,3:]=np.maximum(prediction[:,:,3:],np.float32(1e-6))
    lower=np.maximum(prediction[:,:,:3]-prediction[:,:,3:]/2,truth[:,None,:3]-truth[:,None,3:]/2)
    upper=np.minimum(prediction[:,:,:3]+prediction[:,:,3:]/2,truth[:,None,:3]+truth[:,None,3:]/2)
    intersection=np.prod(np.maximum(upper-lower,0),axis=-1)
    iou=intersection/(np.prod(prediction[:,:,3:],axis=-1)+np.prod(truth[:,3:],axis=-1)[:,None]-intersection)
    assert np.isfinite(iou).all()
    modes={}
    for k,mode in enumerate(('bbs','bbf')):
        chosen=np.asarray([r[mode]['query'] for r in rows])
        selected=iou[np.arange(9508),chosen]
        saved=np.asarray([r[mode]['iou'] for r in rows])
        assert np.max(np.abs(saved-selected))<1e-5
        assert np.array_equal(saved>.25,selected>.25) and np.array_equal(saved>.5,selected>.5)
        assert np.array_equal(scores[np.arange(9508),k,chosen],scores[:,k,:].max(axis=1))
        metrics=dict(rec_hits25=int((selected>.25).sum()),rec_hits50=int((selected>.5).sum()),
                     mask_hits25=sum(r[mode]['mask_iou']>.25 for r in rows),mask_hits50=sum(r[mode]['mask_iou']>.5 for r in rows))
        assert metrics==native['metrics'][mode]
        modes[mode]=dict(hits=[metrics['rec_hits25'],metrics['rec_hits50']],percent=[metrics['rec_hits25']/9508*100,metrics['rec_hits50']/9508*100],mask_hits_from_saved_iou=[metrics['mask_hits25'],metrics['mask_hits50']],max_saved_iou_difference=float(np.max(np.abs(saved-selected))))
    results[str(batch)]=dict(rows=9508,modes=modes,full256_oracle_hits=[int((iou>.25).any(axis=1).sum()),int((iou>.5).any(axis=1).sum())],seconds=native['seconds'],weight_state_unchanged=True,candidate_box_iou_recomputations=int(iou.size))
    inputs[batch]=rows
    arrays[batch]=(boxes,scores)

identity_keys=('row_id','scan_id','target_id','utterance','root_box','point_sha256','detector_sha256','native_maps_sha256')
identity={key:sum(a[key]!=b[key] for a,b in zip(inputs[8],inputs[24])) for key in identity_keys}
assert not any(identity.values())
contrasts={}
for mode in ('bbs','bbf'):
    pairs={}
    for threshold,name in ((.25,'025'),(.5,'050')):
        a=np.asarray([r[mode]['iou']>threshold for r in inputs[8]])
        b=np.asarray([r[mode]['iou']>threshold for r in inputs[24]])
        repair=int((~a&b).sum());damage=int((a&~b).sum())
        pairs[name]=dict(batch24_gain=repair,batch24_loss=damage,net=repair-damage)
    contrasts[mode]=pairs
historical=[json.loads(line) for line in (root/'historical_parent_rows.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(historical)==9508
historical_inputs={key:sum(a[key]!=b[key] for a,b in zip(historical,inputs[8])) for key in ('row_id','scan_id','target_id','root_box','point_sha256')}
history_compare={mode:dict(old_hits=[sum(r[mode]['iou']>.25 for r in historical),sum(r[mode]['iou']>.5 for r in historical)],new_batch8_hits=results['8']['modes'][mode]['hits']) for mode in ('bbs','bbf')}
record=dict(status='FULL_SAVED_PARENT_PREDICTION_CPU_RECOUNT_PASS',rows_each=9508,seed=2027,multiseed=False,batches=results,input_identity_differences=identity,batch24_vs_batch8=contrasts,historical_batch8_inputs=historical_inputs,historical_batch8_comparison=history_compare,candidate_box_iou_recomputations=sum(v['candidate_box_iou_recomputations'] for v in results.values()),all256_box_max_absolute_batch_difference=float(np.max(np.abs(arrays[8][0]-arrays[24][0]))),all_native_score_max_absolute_batch_difference=float(np.max(np.abs(arrays[8][1]-arrays[24][1]))),same_query_slot_counts={mode:sum(a[mode]['query']==b[mode]['query'] for a,b in zip(inputs[8],inputs[24])) for mode in ('bbs','bbf')},current_method_best_unchanged=True,current_method_best_hits=[5599,4859],current_method_target_hits=[5658,4850],new_accuracy_claim_scope='Official-parent evaluation protocol only; neither arm is a new method or a promoted checkpoint.',limits=['Saved Box predictions and saved root boxes recomputed; no author raw-data/cache rebuild.','Mask IoUs recounted from saved values, not raw point Masks.','Batch changes also change Gumbel draw grouping and text padding; individual causes are not isolated.','Query slot coincidence is not physical-instance identity.','Detector box hashes do not cover detector label-mask or class-ID tensors.'])
out=root/'formal_analysis';out.mkdir()
(out/'SUMMARY.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(record),flush=True)
