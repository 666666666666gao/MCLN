import hashlib
import json
from pathlib import Path

import numpy as np

from cohort_metrics import summarize

root=Path(__file__).resolve().parent
data=root/'complete'
receipt=json.loads((data/'receipt.json').read_bytes())
spec=json.loads((data/'spec.json').read_bytes())
assert receipt['status']=='pass' and receipt['rows']==64 and receipt['batches']==8
assert receipt['optimizer_steps']==0 and receipt['weight_files_created']==0
assert receipt['frozen_upstream_masks_and_score_exact'] and receipt['parent_cached_replay_exact']
assert receipt['model_state_restored'] and receipt['model_gradients_absent']
assert hashlib.sha256((data/'spec.json').read_bytes()).hexdigest()==receipt['spec_sha256']
assert hashlib.sha256((data/'run_cohort_probe.py').read_bytes()).hexdigest()==receipt['runner_sha256']
assert hashlib.sha256((data/'rows.jsonl').read_bytes()).hexdigest()==receipt['rows_sha256']
assert (data/'child.exit').read_text().strip()=='0'
assert (data/'controller.exit').read_text().strip()=='0'
rows=[json.loads(line) for line in (data/'rows.jsonl').read_text().splitlines()]
assert len(rows)==64
reference=root.parent/'pvground_mask_geometry_responsibility_20261005'/'complete'/'rows.jsonl'
prior=[json.loads(line) for line in reference.read_text().splitlines()]
chunks=[np.load(str(data/('batch_%02d.npz'%i)),allow_pickle=False) for i in range(8)]
names=set(chunks[0].files)
assert all(set(chunk.files)==names for chunk in chunks)
arrays={name:np.concatenate([chunk[name] for chunk in chunks]) for name in names}
assert arrays['row_id'].tolist()==[row['row_id'] for row in rows]
assert len(set(arrays['row_id'].tolist()))==64
for index,(row,old) in enumerate(zip(rows,prior)):
    for key in ('row_id','scan_id','point_sha256','root_box'):
        assert row[key]==old[key],(index,key)
    assert row['root_box']==arrays['root_box'][index].tolist()
    assert row['selected_query']==int(arrays['bbs'][index].argmax())
    assert row['native_head_calls']==1 and row['head_replays']==3 and row['frozen_upstream_and_score']
    assert row['valid_native_GT_slots']==[0]
for chunk in chunks:
    chunk.close()
summary=summarize(arrays)
assert summary['parent_qualified_candidates']==sum(row['parent_qualified_count'] for row in rows)
assert summary['qualified_outside_candidates']==sum(row['parent_qualified_outside_count'] for row in rows)
summary.update(status='CPU_RECOUNT_COMPLETE_AUDIT_PENDING',terminal_review_pending=True,
    source_spec_sha256=receipt['spec_sha256'],source_runner_sha256=receipt['runner_sha256'],
    probe_finished_cst=receipt['time_cst'],receipt_sha256=hashlib.sha256((data/'receipt.json').read_bytes()).hexdigest())
output=root/'analysis'
output.mkdir(exist_ok=True)
(output/'SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({key:summary[key] for key in ('status','rows','parent_qualified_candidates',
    'qualified_outside_candidates','CPU_parent_qualification_changes','accuracy_result')}))
