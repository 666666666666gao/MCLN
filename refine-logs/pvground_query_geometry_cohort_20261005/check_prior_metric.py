import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from cohort_metrics import cpu_iou

root=Path(__file__).resolve().parent
data=root.parent/'pvground_mask_branch_responsibility_20261005'/'complete'
receipt=json.loads((data/'receipt.json').read_bytes())
chunks=[np.load(str(data/('batch_%02d.npz'%i)),allow_pickle=False) for i in range(8)]
arrays={key:np.concatenate([chunk[key] for chunk in chunks]) for key in
    ('boxes','root_box','box_iou','query_intersection','query_union','fused_intersection','fused_union','matched_slot')}
ious=cpu_iou(arrays['boxes'],arrays['root_box'])
maximum_difference=float(np.max(np.abs(ious-arrays['box_iou'])))
assert maximum_difference<1e-5
qualified=(2*arrays['query_intersection']>arrays['query_union'])&(
    2*arrays['fused_intersection']>arrays['fused_union'])&(arrays['box_iou']<=.5)&(arrays['matched_slot']<0)
assert int(qualified.sum())==receipt['totals']['own_query_confirmed']
result=dict(status='PASS_PRIOR_CLOSED_DATA_METRIC_PRIMITIVE_ONLY',time_cst=datetime.datetime.now().astimezone().isoformat(),
    rows=len(ious),candidates=int(ious.size),qualified_candidates=int(qualified.sum()),
    maximum_CPU_GPU_iou_difference=maximum_difference,
    threshold_changes={str(t):int(((ious>t)!=(arrays['box_iou']>t)).sum()) for t in (.25,.5)},
    receipt_sha256=hashlib.sha256((data/'receipt.json').read_bytes()).hexdigest(),
    metric_source_sha256=hashlib.sha256((root/'cohort_metrics.py').read_bytes()).hexdigest(),
    new_cohort_executed=False,accuracy_result=False,
    limitation='Existing closed branch diagnostic only; does not verify future paired-head GPU execution.')
for chunk in chunks:
    chunk.close()
(root/'METRIC_PRIMITIVE_CHECK.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(result))
