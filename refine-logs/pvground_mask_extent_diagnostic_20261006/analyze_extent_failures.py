"""Additional selected-Query breakdown, using actual closed9508 evidence only."""
import argparse
import json
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--directory',required=True,type=Path)
args=parser.parse_args()
root=args.directory
proof=json.loads((root/'CPU_SUMMARY.json').read_bytes())
receipt=json.loads((root/'receipt.json').read_bytes())
rows=[json.loads(line) for line in (root/'rows.jsonl').read_text().splitlines()]
assert proof['rows']==receipt['rows']==len(rows)==9508
assert proof['zero_optimization'] and receipt['model_states_unchanged']
assert not any(value for item in proof['CPU_stored_threshold_flips'].values() for value in item.values())
good=np.array([row['learned_iou']>.5 for row in rows])
fused=np.array([row['fused_mask_iou']>.5 for row in rows])
own=np.array([row['own_mask_iou']>.5 for row in rows])
for start in range(0,9508,8):
    with np.load(str(root/('batch_%04d.npz'%(start//8))),allow_pickle=False) as payload:
        for row in rows[start:start+8]:
            key='r'+str(row['row_id'])+'__'
            count=payload[key+'count']; target=payload[key+'target_count']; active=payload[key+'own_active']
            intersection=int(target[active].sum())
            value=intersection/(int(count[active].sum())+int(target.sum())-intersection)
            assert abs(value-row['own_mask_iou'])<1e-12

def effect(chosen,name):
    selected=np.flatnonzero(chosen)
    after=np.array([rows[index][name+'_iou']>.5 for index in selected],dtype=np.bool_)
    before=good[selected]
    return dict(rows=len(selected),before_hits=int(before.sum()),after_hits=int(after.sum()),
                repairs=int((after&~before).sum()),damages=int((before&~after).sum()),
                net=int(after.sum()-before.sum()))

groups={}
for label,chosen in [('own_and_fused_good',own&fused),('own_and_fused_good_box_bad',own&fused&~good),
                     ('fused_good_own_bad',fused&~own),('mask_both_bad',~own&~fused),
                     ('learned_correct',good),('learned_wrong',~good)]:
    groups[label]={name:effect(chosen,name) for name in ('exact','quantile')}
edge_stats={}
for name in ('learned','exact','quantile'):
    error=[]; normalized=[]
    for row in rows:
        box=row[name+'_box']
        if box is None:continue
        truth=np.array(row['root_box']);box=np.array(box)
        faces=np.concatenate((box[:3]-box[3:]/2,box[:3]+box[3:]/2))
        target=np.concatenate((truth[:3]-truth[3:]/2,truth[:3]+truth[3:]/2))
        deviation=np.abs(faces-target)
        error.append(float(deviation.max()))
        normalized.append(float((deviation/np.tile(truth[3:],2)).max()))
    edge_stats[name]=dict(valid_rows=len(error),max_face_error_median_metres=float(np.median(error)),
                          scale_normalized_max_face_error_median=float(np.median(normalized)))
result=dict(status='ACTUAL_9508_SELECTED_QUERY_FAILURE_BREAKDOWN',rows=9508,
            mask_combinations=dict(both_good=int((own&fused).sum()),fused_only=int((fused&~own).sum()),
                                   own_only=int((own&~fused).sum()),both_bad=int((~own&~fused).sum())),
            groups=groups,all_rows_edge_errors=edge_stats,own_mask_member_intersections_cpu_verified=True,
            scope='Same selectedQuery, GT diagnostic groups only; no deployable GT gating or additional formal model')
destination=root/'FAILURE_BREAKDOWN.json';assert not destination.exists()
destination.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
