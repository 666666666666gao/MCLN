"""Describe the retained model's geometry errors from immutable saved artifacts."""
import datetime
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np

root = Path(__file__).resolve().parent
workspace = Path('C:/Users/gb')
prior = workspace/'.codex/tmp/pvground_compressed_geometry_support_20261008'
member_root = workspace/'.codex/tmp/pvground_support_boundary_cases_20261007'
source = prior/'complete_fit/formal/rows.jsonl'
intake = json.loads((prior/'complete_fit/INTAKE.json').read_bytes())
entry = next(v for v in intake['files'] if v['name'] == 'formal/rows.jsonl')
assert source.stat().st_size == entry['bytes']
assert hashlib.sha256(source.read_bytes()).hexdigest() == entry['sha256']
best_path = workspace/'.codex/archives/pvg_compressed_support_best_20261008/terminal.pth'
assert hashlib.sha256(best_path.read_bytes()).hexdigest() == '6f9109cb690399880b8dc3bcf6e4b98d1053145732ee2a6b504bfaecbd1eaac2'
assert not (root/'SUMMARY.json').exists()
rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
assert len(rows) == 9508 and [r['row_id'] for r in rows] == list(range(9508))
assert all(r['parent_forwards'] == 1 and r['final_semantic_head_calls'] == 1 for r in rows)
pred = np.asarray([r['arms']['content']['box'] for r in rows],dtype=np.float64)
truth = np.asarray([r['root_box'] for r in rows],dtype=np.float64)
ps = np.maximum(pred[:,3:],1e-6); ts = truth[:,3:]
inter_dims = np.maximum(0,np.minimum(pred[:,:3]+ps/2,truth[:,:3]+ts/2)-np.maximum(pred[:,:3]-ps/2,truth[:,:3]-ts/2))
inter = inter_dims.prod(axis=1)
pv = ps.prod(axis=1); tv = ts.prod(axis=1)
iou = inter/(pv+tv-inter)
assert [int((iou>.25).sum()),int((iou>.5).sum())] == [5599,4859]
assert np.max(np.abs(iou-np.asarray([r['arms']['content']['iou'] for r in rows]))) < 1e-5
coverage = inter/tv; precision = inter/pv
mask = np.asarray([r['arms']['content']['mask_iou'] for r in rows])
valid = np.asarray([r['arms']['content']['reference_valid'] for r in rows])
pool = {}
for row in rows:
    key = (row['scan_id'],row['target_id'])
    value = np.asarray(row['root_box'],dtype=np.float64)
    if key in pool:
        assert np.array_equal(pool[key],value), key
    else:
        pool[key] = value
scenes = {}
for (scene,target),box in pool.items():
    scenes.setdefault(scene,[]).append((target,box))
other_better = np.zeros(len(rows),dtype=bool)
other_hit = np.zeros(len(rows),dtype=bool)
other_ids = []
for index,row in enumerate(rows):
    candidates = [(target,box) for target,box in scenes[row['scan_id']] if target != row['target_id']]
    if not candidates:
        other_ids.append(None)
        continue
    ids,boxes = zip(*candidates); boxes = np.asarray(boxes)
    overlap = np.maximum(0,np.minimum(pred[index,:3]+ps[index]/2,boxes[:,:3]+boxes[:,3:]/2)-np.maximum(pred[index,:3]-ps[index]/2,boxes[:,:3]-boxes[:,3:]/2)).prod(axis=1)
    values = overlap/(pv[index]+boxes[:,3:].prod(axis=1)-overlap)
    which = int(np.argmax(values))
    other_ids.append(int(ids[which]))
    other_better[index] = values[which] > iou[index]
    other_hit[index] = values[which] > .25

partitions = {}
for threshold in (.25,.5):
    wrong = iou <= threshold
    # Geometry descriptions, not physical-instance or language-identity labels.
    zero = wrong & (inter == 0)
    over = wrong & ~zero & (coverage >= .5) & (precision < .5)
    under = wrong & ~zero & ~over & (precision >= .5) & (coverage < .5)
    mixed = wrong & ~(zero|over|under)
    assert np.array_equal(wrong,zero|over|under|mixed)
    groups = {'zero_root_overlap':zero,'root_half_covered_low_box_precision':over,'high_box_precision_low_root_coverage':under,'other_partial_overlap':mixed}
    partitions[str(threshold)] = {'errors':int(wrong.sum()),'mask_good_box_bad':int((wrong & (mask>.5)).sum()),'invalid_reference_errors':int((wrong & ~valid).sum()),'another_referenced_gt_higher_overlap':int((wrong & other_better).sum()),'another_referenced_gt_iou_above025':int((wrong & other_hit).sum()),'groups':{name:{'rows':int(flag.sum()),'mask_iou_above05':int((flag & (mask>.5)).sum()),'median_root_coverage':float(np.median(coverage[flag])) if flag.any() else None,'median_box_precision':float(np.median(precision[flag])) if flag.any() else None} for name,flag in groups.items()}}

# Reuse actual member slices only where current point identity and target agree.
# No old Query, old foreground mask, or new model forward is reconstructed.
sys.path.insert(0,str(member_root))
from historical_face_provenance import historical_faces, TOLERANCE_M
manifest = json.loads((member_root/'case_manifest.json').read_bytes())
member_intake = json.loads((member_root/'complete/INTAKE.json').read_bytes())
archived = {v['name']:v for v in member_intake['files']}
case_counts = Counter(); face_counts = Counter(); case_details = []
for case in manifest['diagnostic_rows']:
    row_id = case['cached']['row_id']; row = rows[row_id]
    assert row['point_sha256'] == case['point_sha256'], row_id
    path_name = 'arrays/row_%05d.npz' % row_id
    path = member_root/'complete'/path_name; record = archived[path_name]
    assert path.stat().st_size == record['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
    with np.load(path,allow_pickle=False) as data:
        arrays = {name:data[name] for name in ('xyz','target','superpoint','root_gt')}
    assert arrays['root_gt'].tolist() == row['root_box']
    case_counts['same_input_cases'] += 1
    case_counts['still_wrong025'] += int(iou[row_id] <= .25)
    case_counts['still_wrong050'] += int(iou[row_id] <= .5)
    case_counts['current_mask_iou_above05'] += int(mask[row_id] > .5)
    if not valid[row_id]:
        case_counts['invalid_reference'] += 1
        continue
    adapted = {'row_id':row_id,'query':row['query'],'reference_box':row['arms']['content']['box'],'root_gt':row['root_box'],'reference_valid':True}
    details = historical_faces(arrays,adapted)
    faces = details['faces']
    case_counts['valid_reference_cases'] += 1
    bad_faces = [f for f in faces if f['signed_gt_outward_error_m'] > TOLERANCE_M]
    case_counts['any_overextending_face'] += bool(bad_faces)
    case_counts['overextending_background_certain'] += any(f['member_certainty']=='only_background_member_possible' for f in bad_faces)
    case_counts['overextending_pure_background_superpoint_certain'] += any(f['all_possible_superpoints_pure_background'] for f in bad_faces)
    for face in faces:
        if face['signed_gt_outward_error_m'] > TOLERANCE_M:
            face_counts['overextending_faces'] += 1
            face_counts['overextending_background_certain'] += face['member_certainty']=='only_background_member_possible'
            face_counts['overextending_pure_background_superpoint_certain'] += face['all_possible_superpoints_pure_background']
        if face['signed_gt_outward_error_m'] < -TOLERANCE_M:
            face_counts['missing_faces'] += 1
            face_counts['missing_faces_with_observed_target_outside'] += face['observed_target_members_beyond_face']>0
    case_details.append({'row_id':row_id,'historical_queue_group':case['group'],'current_query':row['query'],'current_box':row['arms']['content']['box'],'current_iou':float(iou[row_id]),'current_mask_iou_saved':float(mask[row_id]),'faces':[{('current_reference_coordinate' if key=='historical_reference_coordinate' else key):value for key,value in face.items()} for face in faces]})

summary = {'status':'EXECUTED_CURRENT_RETAINED_MODEL_DESCRIPTIVE_ERROR_ANALYSIS','generated_cst':datetime.datetime.now().astimezone().isoformat(),'source_rows_sha256':entry['sha256'],'checkpoint_sha256':'6f9109cb690399880b8dc3bcf6e4b98d1053145732ee2a6b504bfaecbd1eaac2','retained_hits':[5599,4859],'target_hits':[5658,4850],'rows':9508,'scenes':len(scenes),'referenced_gt_instances':len(pool),'partitions':partitions,'member_case_panel':{'case_counts':dict(case_counts),'face_counts':dict(face_counts),'coordinate_tolerance_m':TOLERANCE_M},'neural_forwards':0,'optimizer_updates':0,'new_weights':0,'new_metric_promotion':False,'review_independence':'executor_descriptive_analysis_not_independent_audit','limits':['Geometry overlap groups and nearest referenced GT are proxies, not language or physical-instance identity.','GT pool contains only expressed targets represented in these9508 rows, not every scene object.','Saved Mask IoU is not raw Mask reconstruction.','191 member cases were selected using old-model errors; this is not a random or full-validation subgroup.','Current point hash and target equal the archived member slice; current foreground sets and Query Mask logits are not available.','Face source is conservative possible-coordinate provenance, not proof of the selected superpoint membership.','No existing inference, score, foreground threshold, or trained checkpoint was changed.']}
(root/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
with (root/'MEMBER_CASES.jsonl').open('x',encoding='utf-8') as stream:
    for row in case_details:stream.write(json.dumps(row)+'\n')
with (root/'CURRENT_ERROR_ROWS.jsonl').open('x',encoding='utf-8') as stream:
    for index,row in enumerate(rows):
        if iou[index] <= .5:stream.write(json.dumps({'row_id':index,'query':row['query'],'iou':float(iou[index]),'mask_iou_saved':float(mask[index]),'root_coverage':float(coverage[index]),'box_precision':float(precision[index]),'volume_ratio':float(pv[index]/tv[index]),'other_referenced_target_id':other_ids[index],'other_referenced_gt_higher_overlap':bool(other_better[index]),'other_referenced_gt_iou_above025':bool(other_hit[index])})+'\n')
print(json.dumps(summary),flush=True)
