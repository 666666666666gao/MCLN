"""Recorded recipe of the executed inline CPU analysis, not a new NN run.

The identical body was executed through stdin at 2026-10-05 05:38 CST.
PARENT_MASK_BOUNDARY_RELATION.json already exists; do not rerun this recipe.
"""
from pathlib import Path
import hashlib
import json
import math
import statistics
import datetime

q=Path('C:/Users/gb/.codex/tmp/pvground_final_quality_20261005')
b=q.parent/'pvground_boundary_distribution_20261004/complete/distribution/formal'
raw=(b/'rows.jsonl').read_bytes(); receipt=json.loads((b/'receipt.json').read_bytes())
assert hashlib.sha256(raw).hexdigest()==receipt['rows_sha256']
rows=[json.loads(x) for x in raw.decode('utf-8').splitlines()]
assert len(rows)==9508 and [r['row_id'] for r in rows]==list(range(9508))
assert sum(r['bbs']['iou']>.5 for r in rows)==4506
assert sum(r['bbs']['mask_iou']>.5 for r in rows)==receipt['metrics']['bbs']['mask_hits50']
quadrants={}
for box_ok,mask_ok in ((True,True),(True,False),(False,True),(False,False)):
    subset=[r for r in rows if (r['bbs']['iou']>.5)==box_ok and (r['bbs']['mask_iou']>.5)==mask_ok]
    errors=[]; moves=[]; correct_direction=nonzero=0
    for r in subset:
        box=r['bbs']['box']; coarse=r['bbs']['coarse_box']; gt=r['root_box']
        faces=lambda x:[x[a]-x[a+3]/2 for a in range(3)]+[x[a]+x[a+3]/2 for a in range(3)]
        ff,cf,tf=faces(box),faces(coarse),faces(gt)
        errors.append(max(abs(x-y) for x,y in zip(ff,tf)))
        moves.append(max(abs(x-y) for x,y in zip(ff,cf)))
        for current,old,truth in zip(ff,cf,tf):
            if current!=old and truth!=old:
                nonzero+=1; correct_direction+=(current-old)*(truth-old)>0
    quadrants[str((box_ok,mask_ok))]=dict(rows=len(subset),box_qualified=box_ok,mask_qualified=mask_ok,
        median_max_face_error_m=statistics.median(errors),median_max_face_move_m=statistics.median(moves),
        coarse_hits50=sum(r['bbs']['coarse_iou']>.5 for r in subset),final_hits50=sum(r['bbs']['iou']>.5 for r in subset),
        same_query_repairs50=sum(r['bbs']['coarse_iou']<=.5<r['bbs']['iou'] for r in subset),
        same_query_damages50=sum(r['bbs']['iou']<=.5<r['bbs']['coarse_iou'] for r in subset),
        moving_faces_direction_toward_gt=correct_direction,moving_faces_direction_denominator=nonzero)
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),source_rows=str(b/'rows.jsonl'),
    source_rows_sha256=hashlib.sha256(raw).hexdigest(),rows=9508,parent_hits=[5616,4506],
    parent_mask_hits50=receipt['metrics']['bbs']['mask_hits50'],quadrants=quadrants,
    scope='Protected distribution parent selected Query only. Same-Query coarse/final geometry; Mask unchanged by this head. Stored GT annotations used offline. Direction is geometric proxy, not causal attribution or complete semantic identity. Does not establish other/unmatched Query Mask eligibility. No NN/optimizer replay, no new quality-arm result.',
    inference_or_optimizer_replayed=False,new_accuracy_result=False)
p=q/'PARENT_MASK_BOUNDARY_RELATION.json'; assert not p.exists()
p.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record))
