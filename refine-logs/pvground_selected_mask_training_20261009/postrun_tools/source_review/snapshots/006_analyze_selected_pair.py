"""Recount the closed selected-winner experiment; no model forward or promotion."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

ARMS=('content','selected_query')
BOX_NAMES=('parent',)+ARMS
N=9508


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def iou(boxes,truth,dtype):
    boxes=boxes.astype(dtype);truth=truth.astype(dtype)
    low=np.maximum(boxes[...,:3]-boxes[...,3:]/2,truth[:,None,:3]-truth[:,None,3:]/2)
    high=np.minimum(boxes[...,:3]+boxes[...,3:]/2,truth[:,None,:3]+truth[:,None,3:]/2)
    intersection=np.maximum(high-low,0).prod(-1)
    result=intersection/(boxes[...,3:].prod(-1)+truth[:,3:].prod(-1)[:,None]-intersection)
    assert np.isfinite(result).all()
    return result


def compare(before,after):
    return {str(t):dict(before_hits=int((before>t).sum()),after_hits=int((after>t).sum()),
        repairs=int(((before<=t)&(after>t)).sum()),damages=int(((before>t)&(after<=t)).sum()),
        net=int((after>t).sum()-(before>t).sum())) for t in (.25,.5)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--actual',type=Path,required=True)
    parser.add_argument('--split-protocol',type=Path,required=True)
    parser.add_argument('--historical-rows',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    intake=json.loads((args.actual/'INTAKE.json').read_bytes())
    assert intake['status']=='CLOSED_FIT_ARTIFACTS_COLLECTED' and intake['weights_copied']==2
    for item in intake['files']:
        path=args.actual/item['name'];assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
    status=json.loads((args.actual/'fit_status.json').read_bytes())
    assert status['status']=='complete' and status['protected_parents_exact'] and status['optimizer_steps_per_arm']==3723
    assert (args.actual/'fit_controller.exit').read_text().strip()=='0'
    spec=json.loads((args.actual/'pair_spec.json').read_bytes());assert spec['starting_hits']==[5599,4859]
    fit=json.loads((args.actual/'receipt.json').read_bytes())
    assert fit['status']=='complete' and fit['training_steps_per_arm']==3723 and fit['fit_rows_per_arm']==29778
    assert fit['fit_seen_exactly_once_per_arm'] and fit['parent_and_box_head_states_exact']
    assert fit['formal_rows']==0 and fit['no_new_initial_or_holdout_accuracy_claim']
    partition=json.loads(args.split_protocol.read_bytes())['row_ids']
    trained=load_rows(args.actual/'train.jsonl');assert len(trained)==3723
    assert [row['step'] for row in trained]==list(range(1,3724))
    seen=[index for row in trained for index in row['rows']]
    assert Counter(seen)==Counter(partition['fit']) and len(set(seen))==29778
    role_counts=Counter();extra_histogram=Counter();total_extra=0
    for row in trained:
        batch_size=len(row['rows']);assert batch_size==(2 if row['step']==3723 else 8)
        assert row['frozen_parent_forwards_per_batch']==row['final_semantic_head_calls']==1
        assert set(row['arms'])==set(ARMS)
        assignments=row['original_parent_assignment'];assert len(assignments)==batch_size
        for arm in ARMS:
            record=row['arms'][arm]
            assert record['native_coefficients']==[5,1,10,2] and record['independent_head_gradients']
            assert np.isfinite(record['loss']) and np.isfinite(record['gradient_norm'])
        control=row['arms']['content'];strategy=row['arms']['selected_query']
        extra=strategy['selected_query_objective'];assert len(extra['rows'])==batch_size
        assert control['expanded_positive_queries']==row['expanded_positive_queries']['content']==0
        assert strategy['expanded_positive_queries']==extra['extra_rows']==row['expanded_positive_queries']['selected_query']
        assert abs(strategy['loss']-strategy['native_matched_loss']-extra['loss'])<1e-5
        assert extra['native_coefficients']==[5,1,10,2]
        observed=Counter()
        for evidence,assignment in zip(extra['rows'],assignments):
            q=evidence['query'];assert 0<=q<256
            mapping=dict(zip(assignment['queries'],assignment['actual_gt_ids']))
            role=('matched_root' if mapping[q]==0 else 'matched_other') if q in mapping else 'unmatched'
            assert evidence['role']==role and evidence['supervised']==(role=='unmatched')
            if evidence['supervised']:assert evidence['target_gt_slot']==0 and np.isfinite(evidence['loss'])
            observed[role]+=1
        assert all(observed[key]==extra['selected_roles'][key] for key in ('matched_root','matched_other','unmatched'))
        assert observed['unmatched']==extra['extra_rows']
        role_counts.update(observed);extra_histogram[extra['extra_rows']]+=1;total_extra+=extra['extra_rows']
    directory=args.actual/'formal';records=load_rows(directory/'rows.jsonl')
    assert len(records)==N and [r['row_id'] for r in records]==list(range(N))
    receipt=json.loads((directory/'receipt.json').read_bytes())
    assert receipt['status']=='pass' and receipt['rows']==receipt['formal_rows']==N
    assert receipt['primary_mode']=='bbs' and receipt['all256_retained'] and receipt['same_selected_query_box_and_mask']
    assert sha(directory/'rows.jsonl')==receipt['rows_sha256']
    assert all(r['parent_forwards']==r['final_semantic_head_calls']==1 for r in records)
    historical=load_rows(args.historical_rows)
    assert len(historical)==N and [r['row_id'] for r in historical]==list(range(N))
    historical_iou=np.asarray([r['arms']['content']['iou'] for r in historical])
    assert [int((historical_iou>t).sum()) for t in (.25,.5)]==[5599,4859]
    identity_fields=('row_id','scan_id','target_id','root_box','point_sha256')
    identity_differences={key:sum(old[key]!=new[key] for old,new in zip(historical,records)) for key in identity_fields}
    assert not any(identity_differences.values())
    selected={name:[] for name in BOX_NAMES};coverage={name:[0,0] for name in BOX_NAMES}
    available_native={arm:[0,0] for arm in ARMS};available_cpu={arm:[0,0] for arm in ARMS}
    primary_flips={name:[0,0] for name in BOX_NAMES};float64_flips={name:[0,0] for name in BOX_NAMES}
    oracle_mismatches={arm:[0,0] for arm in ARMS};max_iou_error={name:0.0 for name in BOX_NAMES}
    offset=batches=0
    for path in sorted(directory.glob('batch_*.npz')):
        assert path.name=='batch_%05d.npz'%offset
        with np.load(path,allow_pickle=False) as batch:
            assert set(batch.files)=={'row_ids','root_gt','scores',*BOX_NAMES}
            count=len(batch['row_ids']);assert count==min(8,N-offset)
            assert np.array_equal(batch['row_ids'],np.arange(offset,offset+count))
            current=records[offset:offset+count];truth=batch['root_gt']
            assert truth.shape==(count,6) and truth.dtype==np.float32 and (truth[:,3:]>0).all()
            assert np.array_equal(truth,np.asarray([r['root_box'] for r in current],dtype=np.float32))
            scores=batch['scores'];assert scores.shape==(count,256) and np.isfinite(scores).all()
            q=np.asarray([r['query'] for r in current]);ids=np.arange(count)
            assert np.array_equal(scores[ids,q],scores.max(-1))
            for name in BOX_NAMES:
                boxes=batch[name];assert boxes.shape==(count,256,6) and boxes.dtype==np.float32
                assert np.isfinite(boxes).all() and (boxes[...,3:]>0).all()
                expected_boxes=[r['parent_box'] if name=='parent' else r['arms'][name]['box'] for r in current]
                saved=np.asarray([r['parent_iou'] if name=='parent' else r['arms'][name]['iou'] for r in current])
                assert np.array_equal(boxes[ids,q],np.asarray(expected_boxes,dtype=np.float32))
                primary=iou(boxes,truth,np.float32);secondary=iou(boxes,truth,np.float64)
                chosen=primary[ids,q];max_iou_error[name]=max(max_iou_error[name],float(np.abs(chosen-saved).max()))
                assert np.abs(chosen-saved).max()<1e-5
                selected[name].extend(saved.tolist())
                for at,t in enumerate((.25,.5)):
                    primary_flips[name][at]+=int(np.count_nonzero((chosen>t)!=(saved>t)))
                    float64_flips[name][at]+=int(np.count_nonzero((secondary[ids,q]>t)!=(saved>t)))
                    coverage[name][at]+=int((primary>t).any(-1).sum())
                    if name in ARMS:
                        field='oracle25' if at==0 else 'oracle50'
                        native_qualified=np.asarray([r['arms'][name][field][-1] for r in current],dtype=bool)
                        available_native[name][at]+=int((native_qualified&~(saved>t)).sum())
                        available_cpu[name][at]+=int(((primary>t).any(-1)&~(chosen>t)).sum())
                        oracle_mismatches[name][at]+=int(np.count_nonzero((primary>t).any(-1)!=[r['arms'][name][field][-1] for r in current]))
        offset+=count;batches+=1
    assert offset==N and batches==1189
    selected={k:np.asarray(v) for k,v in selected.items()}
    table={}
    for arm in ARMS:
        native=receipt['metrics'][arm];hits=[int((selected[arm]>t).sum()) for t in (.25,.5)]
        assert hits==[native['rec_hits25'],native['rec_hits50']]
        mask=np.asarray([r['arms'][arm]['mask_iou'] for r in records]);assert np.isfinite(mask).all() and ((mask>=0)&(mask<=1)).all()
        assert [int((mask>t).sum()) for t in (.25,.5)]==[native['mask_hits25'],native['mask_hits50']]
        assert abs(mask.mean()*100-native['mask_miou'])<1e-9
        difference=compare(selected['parent'],selected[arm])
        for t,suffix in ((.25,'25'),(.5,'50')):
            assert difference[str(t)]['repairs']==native['repairs'+suffix] and difference[str(t)]['damages']==native['damages'+suffix]
        table[arm]=dict(hits=hits,percent=[v/N*100 for v in hits],same_frame_uncorrected_parent=difference,
            historical_protected_comparison=compare(historical_iou,selected[arm]),
            final_mask_metrics_from_stored_iou=dict(hits25=native['mask_hits25'],hits50=native['mask_hits50'],miou=native['mask_miou']),
            qualified_but_unselected_native=available_native[arm],qualified_but_unselected_cpu=available_cpu[arm],
            no_qualified_candidate=[N-value for value in coverage[arm]],
            target_passed=hits[0]>=5658 and hits[1]>=4850,strict_gate=hits[1]>=4850)
    assert [int((selected['parent']>t).sum()) for t in (.25,.5)]==[receipt['parent_hits25'],receipt['parent_hits50']]
    restores={}
    for arm in ARMS:
        r=json.loads((args.actual/arm/'formal_restore.json').read_bytes())
        assert r['status']=='pass' and r['optimizer']['all_keys_moments_steps_and_groups_exact']
        assert r['terminal_sha256']==sha(args.actual/arm/'terminal.pth');restores[arm]=r
    report=dict(status='CLOSED_SELECTED_MASK_CPU_RECOUNT_COMPLETE',rows=N,steps=3723,fit_ids=29778,
        all_fit_ids_once=True,table=table,treatment_vs_same_budget_control=compare(selected['content'],selected['selected_query']),
        candidate_box_iou_recomputations=N*256*3,full256_coverage=coverage,primary_cpu_threshold_flips=primary_flips,
        float64_threshold_flips=float64_flips,full256_oracle_mismatches=oracle_mismatches,selected_iou_max_error=max_iou_error,
        current_best_hits=[5599,4859],identity_differences=identity_differences,selected_training_roles=dict(role_counts),
        total_extra_selected_rows=total_extra,extra_rows_per_batch_histogram=dict(extra_histogram),formal_restore_receipts=restores,
        cumulative_support_updates=11169,best_updated=False,three_effective_contributions_established=False,full_goal_complete=False,
        source_hashes=dict(intake=sha(args.actual/'INTAKE.json'),split_protocol=sha(args.split_protocol),historical_rows=sha(args.historical_rows)),
        limits=['GT is saved dataset root labels; this script does not reconstruct the raw ScanNet GT.',
                'Mask metrics use saved point-mask IoUs; no independent full point-mask replay.',
                'Same-frame parent is the uncorrected Mask-reference parent, not the retained learned corrector.',
                'Historical protected comparison is an independent forward; identity alignment does not prove bitwise outputs.',
                'CPU/Torch float threshold discrepancies remain explicit for actual audit; no automatic promotion.',
                'One seed and long-used development validation do not establish statistical stability or three effective modules.'])
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],table=table,treatment_vs_control=report['treatment_vs_same_budget_control'],best_updated=False)),flush=True)


if __name__=='__main__':main()
