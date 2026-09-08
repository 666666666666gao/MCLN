"""Independently recount native PV-Ground official rows and promotion criteria."""
import argparse
import datetime
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


def audit(root):
    assert (root/'evaluation.exit').read_text().strip()=='0'
    receipt=json.loads((root/'receipt.json').read_bytes())
    spec=json.loads((root/'spec.json').read_bytes())
    protocol=json.loads((root/'protocol.json').read_bytes())
    assert receipt['status']=='complete'
    assert receipt['formal_rows']==receipt['rows_per_arm']==protocol['formal_rows']==9508
    assert receipt['optimizer_steps']==receipt['new_checkpoint_files']==0
    assert receipt['all_model_states_unchanged'] and receipt['primary_mode']==protocol['primary_mode']=='bbs'
    assert sha(root/'spec.json')==receipt['spec_sha256']
    assert sha(root/'evaluate.py')==receipt['script_sha256']==spec['files']['evaluate.py']
    assert sha(root/'protocol.json')==receipt['protocol_sha256']
    for name,digest in spec['files'].items():
        assert sha(root/name)==digest,name
    training=Path(spec['training_root'])
    audited=Path(spec['training_audit_root'])
    assert sha(training/'receipt.json')==receipt['training_receipt_sha256']
    assert sha(training/'terminal.pth')==receipt['terminal_checkpoint_sha256']
    assert sha(audited/'audit.json')==receipt['training_audit_sha256']
    assert json.loads((audited/'audit.json').read_bytes())['primary_rec_nonregression']
    assert protocol['batch_size']==8 and protocol['workers']==2 and protocol['seed']==2027
    assert protocol['v99_rec_floor_hits']==[5572,4797]
    assert protocol['scan_mask_floor_percent']==[58.70,50.70,44.72]
    identities=protocol['parsed_identities']
    assert len(identities)==9508
    stages={};records={};metrics={}
    for arm in ['published_parent','fit_terminal']:
        directory=root/arm
        arm_receipt=json.loads((directory/'receipt.json').read_bytes())
        assert arm_receipt['arm']==arm and arm_receipt['formal_rows']==9508 and arm_receipt['state_unchanged']
        rows,actual,checks=recount_native_rows(directory)
        assert [row['row_id'] for row in rows]==list(range(9508))
        for row,identity in zip(rows,identities):
            assert row['scan_id']==identity[0] and row['target_id']==identity[1]
        assert actual==receipt['metrics'][arm]
        records[arm]=rows;metrics[arm]=actual;stages[arm]=checks
    before,after=records['published_parent'],records['fit_terminal']
    transitions={};bands={}
    for mode in ['bbs','bbf']:
        transitions[mode]={};bands[mode]=[[0]*3 for _ in range(3)]
        for old,new in zip(before,after):
            for key in ['row_id','scan_id','target_id','root_box','point_sha256']:
                assert old[key]==new[key],key
            i=int(old[mode]['iou']>.25)+int(old[mode]['iou']>.5)
            j=int(new[mode]['iou']>.25)+int(new[mode]['iou']>.5)
            bands[mode][i][j]+=1
        for threshold in [.25,.5]:
            fixes=sum(old[mode]['iou']<=threshold<new[mode]['iou'] for old,new in zip(before,after))
            breaks=sum(new[mode]['iou']<=threshold<old[mode]['iou'] for old,new in zip(before,after))
            transitions[mode][str(threshold)]={'fixes':fixes,'breaks':breaks,'net':fixes-breaks}
    assert transitions==receipt['transitions']
    candidate=metrics['fit_terminal']['bbs']
    checks={'rec25_historical_v99':candidate['rec_hits25']>=5572,
        'rec50_historical_v99':candidate['rec_hits50']>=4797,
        'mask25_paper':candidate['mask_hits25']*100./9508>=58.70,
        'mask50_paper':candidate['mask_hits50']*100./9508>=50.70,
        'mask_miou_paper':candidate['mask_miou']>=44.72}
    assert checks==receipt['promotion']['checks']
    assert all(checks.values())==receipt['promotion']['advance_to_nr3d_sr3d_rec']
    assert receipt['promotion']['requires_independent_formal_audit']
    assert not receipt['promotion']['nr3d_sr3d_mask_gate']
    return {'integrity_pass':True,'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'formal_rows':9508,'metrics':metrics,'transitions':transitions,'transition_counts':bands,
        'transition_bands':['[0,0.25]','(0.25,0.50]','(0.50,1]'],'checks':checks,
        'advance_to_nr3d_sr3d_rec':all(checks.values()),'stages':stages,
        'mask_audit_scope':'exported per-row IoU recount; binary predictions not reloaded',
        'v99_reference':'historical protected complete system; not a contemporaneous rerun',
        'receipt_sha256':sha(root/'receipt.json'),'auditor_sha256':sha(__file__)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.root)
    with args.out.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print('PVG_FORMAL_AUDIT_COMPLETE '+json.dumps(result),flush=True)


if __name__=='__main__':
    main()
