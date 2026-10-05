"""Recount all saved candidate geometry/role evidence on CPU."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

local = Path(__file__).resolve().parent
complete = local/'complete'
output = local/'analysis'
output.mkdir()
receipt = json.loads((complete/'receipt.json').read_bytes())
intake = json.loads((complete/'INTAKE.json').read_bytes())
for item in intake['files']:
    assert hashlib.sha256((complete/item['name']).read_bytes()).hexdigest() == item['sha256']
rows = [json.loads(line) for line in (complete/'rows.jsonl').read_text().splitlines()]
assert len(rows) == receipt['rows'] == 64
counts = Counter()
errors = []; changes = []; batch_rows = []; checks = 0
outside_candidates = 0; outside_faces = 0; coarse_hits = 0
for index in range(8):
    arrays = np.load(complete/('batch_%02d.npz' % index), allow_pickle=False)
    assert arrays['boxes'].shape == (8,256,6)
    batch_rows.extend(arrays['row_id'].tolist())
    boxes = arrays['boxes'].astype(np.float64)
    truth = arrays['root_box'].astype(np.float64)[:,None]
    low = np.maximum(boxes[...,:3]-boxes[...,3:]/2,truth[...,:3]-truth[...,3:]/2)
    high = np.minimum(boxes[...,:3]+boxes[...,3:]/2,truth[...,:3]+truth[...,3:]/2)
    intersection = np.maximum(0,high-low).prod(-1)
    iou = intersection/(boxes[...,3:].prod(-1)+truth[...,3:].prod(-1)-intersection)
    assert np.isfinite(iou).all()
    assert not np.any((iou > .5)!=(arrays['box_iou'] > .5))
    mask_iou = arrays['mask_intersection']/arrays['mask_union']
    assert np.isfinite(mask_iou).all()
    assert not np.any((mask_iou > .5)!=(arrays['mask_iou'] > .5))
    checks += 4
    coarse = arrays['coarse_boxes'].astype(np.float64)
    reference_size = np.maximum(coarse[...,3:], 1e-6)
    face_targets = np.concatenate([
        (coarse[...,:3]-(truth[...,:3]-truth[...,3:]/2))/reference_size*4-2,
        ((truth[...,:3]+truth[...,3:]/2)-coarse[...,:3])/reference_size*4-2], -1)
    outside = (face_targets < -4) | (face_targets > 4)
    clow = np.maximum(coarse[...,:3]-coarse[...,3:]/2,truth[...,:3]-truth[...,3:]/2)
    chigh = np.minimum(coarse[...,:3]+coarse[...,3:]/2,truth[...,:3]+truth[...,3:]/2)
    ci = np.maximum(0,chigh-clow).prod(-1)
    coarse_iou = ci/(coarse[...,3:].prod(-1)+truth[...,3:].prod(-1)-ci)
    for bid in range(8):
        row = rows[index*8+bid]
        assert int(arrays['row_id'][bid]) == row['row_id']
        selected = int(np.argsort(-arrays['bbs'][bid],kind='stable')[0])
        assert selected == row['selected_query']
        matched = arrays['matched_slot'][bid]
        positive = (mask_iou[bid] > .5) & (iou[bid] <= .5)
        good = (mask_iou[bid] > .5) & (iou[bid] > .5)
        assert np.all(arrays['box_gradient_max'][bid][matched<0] == 0)
        assert np.all(arrays['boundary_gradient_max'][bid][matched<0] == 0)
        actual = dict(mask_only=int(positive.sum()),both_good=int(good.sum()),
            mask_only_unmatched=int((positive & (matched<0)).sum()),
            mask_only_matched_root=int((positive & (matched==0)).sum()),
            mask_only_matched_other=int((positive & (matched>0)).sum()),
            mask_only_box_gradient_nonzero=int((positive & (arrays['box_gradient_max'][bid]>0)).sum()),
            mask_only_boundary_gradient_nonzero=int((positive & (arrays['boundary_gradient_max'][bid]>0)).sum()),
            unmatched_good_box=int(((iou[bid]>.5) & (matched<0)).sum()))
        assert actual == row['counts']
        counts.update(actual)
        counts['rows_with_mask_only_unmatched'] += actual['mask_only_unmatched'] > 0
        counts['rows_with_both_good'] += actual['both_good'] > 0
        counts['selected_mask_only'] += bool(positive[selected])
        counts['selected_mask_only_unmatched'] += bool(positive[selected] and matched[selected] < 0)
        counts['valid_native_GT_slots_total'] += len(row['valid_native_GT_slots'])
        errors.extend(arrays['max_face_error'][bid][positive & (matched<0)].tolist())
        changes.extend(arrays['max_face_change'][bid][positive & (matched<0)].tolist())
        pool = positive & (matched<0)
        outside_candidates += int(outside[bid][pool].any(-1).sum())
        outside_faces += int(outside[bid][pool].sum())
        coarse_hits += int(((coarse_iou[bid]>.5) & pool).sum())
        assert set(matched[matched>=0].tolist()) == set(row['native_matched_slots'])
        checks += 8
    arrays.close()
assert batch_rows == [row['row_id'] for row in rows]
assert dict(counts) == receipt['totals']
assert errors and len(errors) == counts['mask_only_unmatched']
summary = dict(status='CPU_RECOUNT_PASS',rows=64,candidates=64*256,counts=dict(counts),
    cpu_checks=checks+4,box_threshold_differences=0,mask_count_threshold_differences=0,
    mask_only_unmatched_median_max_face_error_m=float(np.median(errors)),
    mask_only_unmatched_median_max_face_change_m=float(np.median(changes)),
    mask_only_unmatched_boundary_target_outside_candidates=outside_candidates,
    mask_only_unmatched_boundary_target_outside_faces=outside_faces,
    mask_only_unmatched_coarse_box_hits50=coarse_hits,
    boundary_target_check='CPU float64 face_targets with executed REG_SCALE4/reference_size_floor1e-6 and knot range[-4,4]; range diagnostic, not new training target.',
    all_native_target_slot_lists_root_only=all(row['valid_native_GT_slots']==[0] for row in rows),
    receipt_sha256=hashlib.sha256((complete/'receipt.json').read_bytes()).hexdigest(),
    optimizer_steps=0,accuracy_result=False,independent_raw_mask_replay=False,
    interpretation='Current fixed64 augmented fit inputs, not historical assignments or formal prevalence. Direct output-level geometry gradients only; shared parameter and other loss effects remain possible.')
(output/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary))
