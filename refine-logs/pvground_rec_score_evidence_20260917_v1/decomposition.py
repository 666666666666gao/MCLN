"""Describe native score per non-null probability mass; never change a decision."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    root=parser.parse_args().root
    diagnostic=json.loads((root/'diagnostic.json').read_bytes())
    audit=json.loads((root/'analysis.json').read_bytes());assert audit['integrity_pass']
    assert audit['diagnostic_sha256']==sha(root/'diagnostic.json')
    assert sha(root/'rows.json')==diagnostic['rows_sha256']
    assert sha(root/'candidate_values.npz')==diagnostic['arrays_sha256']
    rows=json.loads((root/'rows.json').read_bytes());archive=np.load(str(root/'candidate_values.npz'))
    records=[];summary={}
    for arm in ['initial','terminal']:
        summary[arm]={}
        for threshold in [.25,.5]:
            key=str(threshold);records_for_threshold=[]
            for row in rows:
                roles=row[arm]['roles']
                if roles['selected']['iou']>threshold or not row[arm]['comparisons'][key]['has_good']:continue
                order=['selected','matched_root','best_iou']+[n for n in ['best_good_0.25','best_good_0.5'] if n in roles]
                probs=archive[str(row['row_id'])+'_'+arm+'_role_probs'].astype(np.float64)
                result=dict(row_id=row['row_id'],scan_id=row['scan_id'],arm=arm,threshold=threshold,roles={})
                for name in ['selected','matched_root','best_good_'+key]:
                    role=roles[name];p=probs[order.index(name)];mass=float(p[:-1].sum());assert mass>0
                    assert abs(float(p[-1])-role['no_object_probability'])<1e-7
                    result['roles'][name]=dict(query=role['query'],iou=role['iou'],native_score=role['score'],
                        nonnull_mass=mass,score_per_nonnull_mass=role['score']/mass,
                        root_per_nonnull_mass=role['components']['root']/mass)
                chosen=result['roles']['selected'];good=result['roles']['best_good_'+key];matched=result['roles']['matched_root']
                result.update(good_has_higher_score_per_nonnull=good['score_per_nonnull_mass']>chosen['score_per_nonnull_mass'],
                    good_has_higher_root_per_nonnull=good['root_per_nonnull_mass']>chosen['root_per_nonnull_mass'],
                    good_has_lower_nonnull_mass=good['nonnull_mass']<chosen['nonnull_mass'],
                    matched_has_higher_score_per_nonnull=matched['score_per_nonnull_mass']>chosen['score_per_nonnull_mass'],
                    best_good_is_matched=good['query']==matched['query'])
                records.append(result);records_for_threshold.append(result)
            summary[arm][key]=dict(covered_errors=len(records_for_threshold),**{k:sum(r[k] for r in records_for_threshold) for k in
                ['good_has_higher_score_per_nonnull','good_has_higher_root_per_nonnull','good_has_lower_nonnull_mass',
                 'matched_has_higher_score_per_nonnull','best_good_is_matched']})
    result=dict(status='complete',rows=128,model_forwards=0,optimizer_steps=0,formal_rows=0,
        diagnostic_sha256=sha(root/'diagnostic.json'),audit_sha256=sha(root/'analysis.json'),script_sha256=sha(Path(__file__)),
        summary=summary,records=records,
        scope='Native selected and highest-native-score GT-good candidates only. Dividing by non-null probability mass '
        'is an algebraic description, not a new ranking evaluation. Does not establish no-object causality or '
        'justify removing no-object supervision. No model/rule modification or new-scene result.')
    with (root/'nonnull_decomposition.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
