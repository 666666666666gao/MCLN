"""Select the39 actually observed empty supports; do not fabricate inputs."""
import hashlib
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
formal=root.parent/'pvground_mask_extent_diagnostic_20261006/complete/formal'
rows=[json.loads(line) for line in (formal/'rows.jsonl').read_text().splitlines()]
parent=root.parent/'pvground_auxiliary_target_20261005/complete/control/formal/rows.jsonl'
prior=[json.loads(line) for line in parent.read_text().splitlines()]
assert len(rows)==len(prior)==9508
values={};records=[]
for index,row in enumerate(rows):
    if row['foreground_count']!=0:
        continue
    historical=prior[index]
    assert row['row_id']==historical['row_id'] and row['point_sha256']==historical['point_sha256']
    assert row['query']==historical['bbs']['query']
    path=formal/('batch_%04d.npz'%(index//8))
    with np.load(path,allow_pickle=False) as batch:
        key='r%d__'%index
        assert not batch[key+'fused_active'].any()
        prefix='r%d__'%len(records)
        for name in ('lower','upper','own_logits','text_logits','alpha'):
            values[prefix+name]=batch[key+name].copy()
        values[prefix+'prior_box']=np.asarray(historical['bbs']['coarse_box'],dtype=np.float32)
        assert values[prefix+'prior_box'].shape==(6,)
    records.append(dict(row_id=index,query=row['query'],point_sha256=row['point_sha256'],
        source_batch=path.name,source_batch_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
assert len(records)==39
np.savez_compressed(root/'invalid_reference_fixture.npz',**values)
(root/'invalid_reference_fixture.json').write_text(json.dumps(dict(status='ACTUAL_EMPTY_SUPPORT_FIXTURE',
    rows=records,no_gt_fields=True,parent_rows_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
    source_rows_sha256=hashlib.sha256((formal/'rows.jsonl').read_bytes()).hexdigest()),indent=2)+'\n')
print(json.dumps(dict(actual_empty_support_rows=len(records),bytes=(root/'invalid_reference_fixture.npz').stat().st_size)))
