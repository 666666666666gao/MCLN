import hashlib
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
complete=root/'complete'
analysis=root/'analysis'
analysis.mkdir()
receipt=json.loads((complete/'receipt.json').read_bytes())
assert receipt['status']=='pass' and receipt['rows']==64 and receipt['optimizer_steps']==0
intake=json.loads((complete/'INTAKE.json').read_bytes())
for entry in intake['files']:
    raw=(complete/entry['name']).read_bytes()
    assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256']
rows=[json.loads(line) for line in (complete/'rows.jsonl').read_text().splitlines()]
prior_root=root.parent/'pvground_mask_geometry_responsibility_20261005'
old_rows=[json.loads(line) for line in (prior_root/'complete/rows.jsonl').read_text().splitlines()]
parts=[dict(np.load(complete/('batch_%02d.npz'%index))) for index in range(8)]
arrays={name:np.concatenate([part[name] for part in parts],axis=0) for name in parts[0]}
assert arrays['box_iou'].shape==(64,256)
assert arrays['row_id'].tolist()==[row['row_id'] for row in rows]==[row['row_id'] for row in old_rows]
assert all(new['point_sha256']==old['point_sha256'] and new['root_box']==old['root_box'] for new,old in zip(rows,old_rows))
for branch in ('text','query','fused'):
    computed=arrays[branch+'_intersection']/arrays[branch+'_union']
    assert np.max(np.abs(computed-arrays[branch+'_iou']))<1e-7
    assert np.array_equal(computed>.5,arrays[branch+'_iou']>.5)
assert np.array_equal(arrays['text_iou'],np.repeat(arrays['text_iou'][:,:1],256,axis=1))
unmatched=arrays['matched_slot']<0
boxpoor=arrays['box_iou']<=.5
fused=arrays['fused_iou']>.5
query=arrays['query_iou']>.5
text=arrays['text_iou']>.5
pool=unmatched&boxpoor&fused
own=pool&query
common=pool&~query&text
neither=pool&~query&~text
counts=dict(fused_mask_only_unmatched=int(pool.sum()),own_query_confirmed=int(own.sum()),
    query_not_qualified_text_qualified=int(common.sum()),neither_branch_qualified=int(neither.sum()),
    own_query_mask_only_unmatched=int((unmatched&boxpoor&query).sum()),
    rows_with_own_query_confirmed=int(own.any(1).sum()))
assert counts==receipt['totals']
old_parts=[dict(np.load(prior_root/'complete'/('batch_%02d.npz'%index))) for index in range(8)]
old_box=np.concatenate([part['box_iou'] for part in old_parts])
old_mask=np.concatenate([part['mask_iou'] for part in old_parts])
old_matched=np.concatenate([part['matched_slot'] for part in old_parts])
matched_diff=int((old_matched!=arrays['matched_slot']).sum())
box_threshold_diff=int(((old_box>.5)!=(arrays['box_iou']>.5)).sum())
mask_threshold_diff=int(((old_mask>.5)!=fused).sum())
summary=dict(status='CPU_RECOUNT_PASS',rows=64,physical_scenes=len(set(row['scan_id'] for row in rows)),
    candidates=16384,counts=counts,own_query_fraction_of_fused_mask_only_unmatched=float(own.sum()/pool.sum()),
    rows_text_mask_qualified=int(text[:,0].sum()),all_native_target_slot_lists_root_only=all(row['valid_native_GT_slots']==[0] for row in rows),
    prior_matched_slot_differences=matched_diff,prior_box_threshold_differences=box_threshold_diff,
    prior_fused_threshold_differences=mask_threshold_diff,
    prior_max_box_iou_difference=float(np.max(np.abs(old_box-arrays['box_iou']))),
    prior_max_fused_iou_difference=float(np.max(np.abs(old_mask-arrays['fused_iou']))),
    alpha_min=float(arrays['alpha'].min()),alpha_max=float(arrays['alpha'].max()),
    optimizer_steps=0,weight_files_created=0,accuracy_result=False,
    interpretation='Training-GT support qualification proxy on fixed augmentedfit64. Text is shared, candidate Query qualification narrows fused support but is not physical identity proof. Independent CUDA forward differences recorded rather than rewriting prior arrays. No new loss/training implemented.')
(analysis/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary))
