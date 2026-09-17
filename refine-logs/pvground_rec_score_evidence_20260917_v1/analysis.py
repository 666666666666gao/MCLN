"""Independently recount frozen token evidence and observed margin eligibility."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root
    assert (root/'controller.exit').read_text().strip()=='0'
    diagnostic=json.loads((root/'diagnostic.json').read_bytes())
    assert diagnostic['status']=='complete' and diagnostic['state_unchanged']
    assert diagnostic['model_forwards']==32 and diagnostic['optimizer_steps']==diagnostic['formal_rows']==0
    for filename,key in [('rows.json','rows_sha256'),('batches.json','batches_sha256'),
                         ('candidate_values.npz','arrays_sha256'),('input_selection.json','input_selection_sha256')]:
        assert sha(root/filename)==diagnostic[key]
    records=json.loads((root/'rows.json').read_bytes())
    batches=json.loads((root/'batches.json').read_bytes())
    selected=json.loads((root/'input_selection.json').read_bytes())
    assert [r['row_id'] for r in records]==[r['row_id'] for r in selected['rows']]
    assert len(records)==len({r['scan_id'].split('_')[0] for r in records})==128
    assert [i for b in batches for i in b['rows']]==[r['row_id'] for r in records]
    archive=np.load(str(root/'candidate_values.npz'))
    summaries={};checks={};all_values={}
    term_names=['root','modifier','pronoun','relation','other_entity']
    for arm in ['initial','terminal']:
        summaries[arm]={};checks[arm]=dict(max_iou_error=0.,max_score_error=0.,max_competition_loss_error=0.)
        values=[];loss_by_row={}
        for row in records:
            a=archive[str(row['row_id'])+'_'+arm].astype(np.float64)
            assert a.shape==(256,14) and np.isfinite(a).all()
            box=a[:,:6].copy();box[:,3:]=np.maximum(box[:,3:],1e-6)
            gt=np.asarray(row['root_box'],dtype=np.float64)
            lo=np.maximum(box[:,:3]-box[:,3:]/2,gt[:3]-gt[3:]/2)
            hi=np.minimum(box[:,:3]+box[:,3:]/2,gt[:3]+gt[3:]/2)
            inter=np.maximum(hi-lo,0).prod(-1)
            iou=inter/(box[:,3:].prod(-1)+gt[3:].prod()-inter)
            error=float(np.abs(iou-a[:,7]).max());assert error<1e-5
            checks[arm]['max_iou_error']=max(checks[arm]['max_iou_error'],error)
            score=a[:,8:12].sum(-1)-a[:,12]
            error=float(np.abs(score-a[:,6]).max());assert error<3e-7
            checks[arm]['max_score_error']=max(checks[arm]['max_score_error'],error)
            roles=row[arm]['roles'];choice=roles['selected']['query'];matched=roles['matched_root']['query']
            assert a[choice,6]==a[:,6].max()
            role_probs=archive[str(row['row_id'])+'_'+arm+'_role_probs']
            assert role_probs.shape==(len(roles),256)
            assert np.all(role_probs>=0) and np.max(np.abs(role_probs.sum(-1)-1))<1e-6
            # The producer writes arrays in this insertion order; JSON keys
            # are sorted by write_json and therefore cannot supply the order.
            role_order=['selected','matched_root','best_iou']
            role_order += [name for name in ['best_good_0.25','best_good_0.5'] if name in roles]
            assert set(role_order)==set(roles)
            for index,name in enumerate(role_order):
                role=roles[name]
                q=role['query'];assert abs(role['score']-a[q,6])<1e-7
                assert abs(role['iou']-iou[q])<1e-5
                assert abs(role['no_object_probability']-a[q,13])<1e-7
                assert abs(role_probs[index,-1]-a[q,13])<1e-7
                assert all(abs(role['components'][n]-a[q,8+k])<1e-7 for k,n in enumerate(term_names))
                for token in role['top_probabilities']:
                    assert abs(token['probability']-role_probs[index,token['position']])<1e-7
            comparisons={};losses=[]
            for t in [.25,.5]:
                key=str(t);good=iou>t;eligible=bool(iou[matched]>t)
                stored=row[arm]['comparisons'][key]
                assert eligible==stored['eligible'] and bool(good.any())==stored['has_good']
                loss=0.
                if eligible and (~good).any():
                    n=int(np.flatnonzero(~good)[np.argmax(a[~good,6])])
                    loss=max(0.,iou[matched]-iou[n]+a[n,6]-a[matched,6])
                    assert abs(loss-stored['actual_margin_violation'])<1e-5
                losses.append(loss)
                entry=dict(hit=bool(iou[choice]>t),covered=bool(good.any()),eligible=eligible,violation=loss)
                if good.any():
                    q=roles['best_good_'+key]['query']
                    assert good[q] and a[q,6]==a[good,6].max()
                    component_gaps=a[q,8:13]-a[choice,8:13]
                    component_gaps[-1]*=-1
                    entry.update(rank=1+int((a[:,6]>a[q,6]).sum()),
                        score_gap=float(a[q,6]-a[choice,6]),
                        signed_component_gaps={n:float(v) for n,v in zip(term_names,component_gaps)},
                        no_object_gap=float(a[q,13]-a[choice,13]))
                    assert entry['rank']==stored['rank_min']
                comparisons[key]=entry
            loss_by_row[row['row_id']]=float(np.mean(losses))
            values.append(dict(row_id=row['row_id'],comparisons=comparisons,
                selected_no_object=float(a[choice,13]),matched_no_object=float(a[matched,13])))
        for batch in batches:
            loss=np.mean([loss_by_row[i] for i in batch['rows']])
            err=abs(float(loss)-batch['losses'][arm]['competition']);assert err<1e-5
            checks[arm]['max_competition_loss_error']=max(checks[arm]['max_competition_loss_error'],err)
        for t in [.25,.5]:
            key=str(t);rows=[v['comparisons'][key] for v in values]
            missed=[v for v in rows if v['covered'] and not v['hit']]
            summaries[arm][key]=dict(selected_hits=sum(v['hit'] for v in rows),
                raw256_hits=sum(v['covered'] for v in rows),matched_eligible=sum(v['eligible'] for v in rows),
                covered_errors=len(missed),covered_errors_matched_ineligible=sum(not v['eligible'] for v in missed),
                covered_errors_root_term_favors_good=sum(v['signed_component_gaps']['root']>0 for v in missed),
                covered_errors_good_no_object_higher=sum(v['no_object_gap']>0 for v in missed),
                component_gap_sums={n:sum(v['signed_component_gaps'][n] for v in missed) for n in term_names},
                competition_violation_sum=sum(v['violation'] for v in rows),
                missed_row_ids=[v['row_id'] for v in values if v['comparisons'][key]['covered'] and not v['comparisons'][key]['hit']])
            assert summaries[arm][key]['selected_hits']==diagnostic['summary'][arm][key]['selected_hits']
            assert summaries[arm][key]['raw256_hits']==diagnostic['summary'][arm][key]['raw256_hits']
        all_values[arm]=values
    pairs={}
    for t in [.25,.5]:
        key=str(t);old=all_values['initial'];new=all_values['terminal']
        pairs[key]=dict(fixes=sum(not a['comparisons'][key]['hit'] and b['comparisons'][key]['hit'] for a,b in zip(old,new)),
            breaks=sum(a['comparisons'][key]['hit'] and not b['comparisons'][key]['hit'] for a,b in zip(old,new)))
    result=dict(status='complete',integrity_pass=True,rows=128,formal_rows=0,model_forwards=0,optimizer_steps=0,
        diagnostic_sha256=sha(root/'diagnostic.json'),script_sha256=sha(Path(__file__)),checks=checks,
        summaries=summaries,pairs=pairs,scope='Same128 preselected fit scenes; frozen exported scores/matches. '
        'Component sums describe native scoring, not a modified deployment rule or causal module importance.')
    with (root/'analysis.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
