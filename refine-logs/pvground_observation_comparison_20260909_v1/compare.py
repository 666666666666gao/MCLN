"""Compare saved control and observation-query outputs; never run the model."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):
            h.update(block)
    return h.hexdigest()


def compare(spec, stage):
    roots=[Path(spec[name]) for name in ['native_root','candidate_root']]
    configs=[json.loads((p/'spec.json').read_bytes()) for p in roots]
    for key in ['checkpoint_sha256','seed','batch_size','fit_passes','lr','lr_backbone',
                'input_manifest','env_spec_sha256','primary_mode']:
        assert configs[0][key]==configs[1][key],key
    assert [sha(p/'spec.json') for p in roots]==spec['training_spec_sha256']
    assert configs[1]['source_query_read'] and configs[1]['observation_state']
    assert bool(configs[0].get('source_query_read',False))==spec['control_source_query']
    assert not configs[0].get('observation_state',False)
    assert sha(roots[1]/'pvground_observation_query.py')==configs[1]['observation_module_sha256']
    for root,config in zip(roots,configs):
        if config.get('source_query_read',False):
            assert sha(root/'pvground_source_query.py')==config['source_query_module_sha256']
    assert [sha(Path(config['source_port'])) for config in configs]==spec['source_port_sha256']
    if stage=='terminal':
        for root in roots:assert (root/'controller.exit').read_text().strip()=='0'
    receipts=[];sets=[]
    for root in roots:
        directory=root/stage
        receipt=json.loads((directory/'receipt.json').read_bytes())
        assert receipt['status']=='pass' and receipt['rows']==6887 and receipt['formal_rows']==0
        assert sha(directory/'rows.jsonl')==receipt['rows_sha256']
        assert sha(directory/'boxes.npy')==receipt['boxes_sha256']
        assert sha(directory/'scores.npy')==receipt['scores_sha256']
        rows=[json.loads(line) for line in (directory/'rows.jsonl').read_text().splitlines()]
        assert len(rows)==6887
        receipts.append(receipt);sets.append(rows)
    identical={k:sum(a[k]==b[k] for a,b in zip(*sets)) for k in
               ['row_id','scan_id','target_id','point_sha256','root_box']}
    assert all(v==6887 for v in identical.values()),identical
    output=dict(status='complete',stage=stage,rows=6887,formal_rows=0,model_forwards=0,optimizer_steps=0,
        time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        roots=[str(p) for p in roots],metrics=[r['metrics'] for r in receipts],identical_row_fields=identical,
        receipt_sha256=[sha(p/stage/'receipt.json') for p in roots],transitions={},scene_transitions={},
        full256_oracle={},script_sha256=sha(Path(__file__)))
    for mode in ['bbs','bbf']:
        output['transitions'][mode]={}
        bands=[[0]*3 for _ in range(3)]
        by_scene={}
        for old,new in zip(*sets):
            before=old[mode]['iou'];after=new[mode]['iou']
            bands[int(before>.25)+int(before>.5)][int(after>.25)+int(after>.5)]+=1
            scene=old['scan_id'].split('_')[0]
            entry=by_scene.setdefault(scene,dict(rows=0,net25=0,net50=0))
            entry['rows']+=1
            entry['net25']+=int(after>.25)-int(before>.25)
            entry['net50']+=int(after>.5)-int(before>.5)
        output['scene_transitions'][mode]=by_scene
        for threshold,key in [(.25,'rec_hits25'),(.5,'rec_hits50')]:
            before=np.asarray([r[mode]['iou']>threshold for r in sets[0]])
            after=np.asarray([r[mode]['iou']>threshold for r in sets[1]])
            assert int(before.sum())==receipts[0]['metrics'][mode][key]
            assert int(after.sum())==receipts[1]['metrics'][mode][key]
            output['transitions'][mode][str(threshold)]=dict(native_hits=int(before.sum()),candidate_hits=int(after.sum()),
                fixes=int((~before&after).sum()),breaks=int((before&~after).sum()),net=int(after.sum()-before.sum()))
        output['transitions'][mode]['iou_band_counts']=bands
    for name,root,rows in zip(['native','candidate'],roots,sets):
        boxes=np.load(str(root/stage/'boxes.npy'),mmap_mode='r')
        assert boxes.shape==(6887,256,6) and boxes.dtype==np.dtype('float32')
        hits25=hits50=0
        for raw,row in zip(boxes,rows):
            b=raw.astype(np.float64);b[:,3:]=np.maximum(b[:,3:],1e-6)
            gt=np.asarray(row['root_box'],dtype=np.float64)
            lo=np.maximum(b[:,:3]-b[:,3:]/2,gt[:3]-gt[3:]/2)
            hi=np.minimum(b[:,:3]+b[:,3:]/2,gt[:3]+gt[3:]/2)
            intersection=np.maximum(hi-lo,0).prod(-1)
            ious=intersection/(b[:,3:].prod(-1)+gt[3:].prod()-intersection)
            assert np.isfinite(ious).all()
            hits25+=int(ious.max()>.25);hits50+=int(ious.max()>.5)
        output['full256_oracle'][name]=dict(hits25=hits25,hits50=hits50,scope='all raw 256 boxes; GT-only upper bound, not legal-filter recall')
    if stage=='terminal':
        training=[[json.loads(line) for line in (p/'train.jsonl').read_text().splitlines()] for p in roots]
        assert len(training[0])==len(training[1])==3723
        assert all(a['step']==b['step'] and a['rows']==b['rows'] for a,b in zip(*training))
        initial=compare(spec,'initial')
        output['initial_cross_control_net']={m:{str(t):initial['transitions'][m][str(t)]['net'] for t in [.25,.5]} for m in ['bbs','bbf']}
        output['difference_in_training_changes']={m:{str(t):output['transitions'][m][str(t)]['net']-initial['transitions'][m][str(t)]['net'] for t in [.25,.5]} for m in ['bbs','bbf']}
        output['own_training_changes']={m:{key:receipts[1]['metrics'][m][key]-initial['metrics'][1][m][key] for key in ['rec_hits25','rec_hits50','mask_hits25','mask_hits50','mask_miou']} for m in ['bbs','bbf']}
        output['training_row_order_identical']=True
    output['export_byte_exact']={name:sha(roots[0]/stage/name)==sha(roots[1]/stage/name) for name in ['rows.jsonl','boxes.npy','scores.npy']}
    output['scope']='Backbone-seen module holdout; same input point hashes and GT. No new-scene or statistical seed-variance claim. Difference of changes is descriptive, not separate causal attribution. No gate or training change.'
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--stage',choices=['initial','terminal'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=compare(json.loads(args.spec.read_bytes()),args.stage)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:result[k] for k in ['status','stage','rows','transitions','full256_oracle','formal_rows']}),flush=True)
