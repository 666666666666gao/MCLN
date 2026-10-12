"""Read one retained real augmented batch; no inference or dataset-wide claims."""
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
saved = Path('C:/Users/gb/.codex/tmp/pvground_extremal_span_evidence_20261009/runner_v1/complete_fit/preflight_support_step_1')
output = root/'superpoint_boundary_pilot_20261012'
assert not output.exists()
receipt_path = saved/'receipt.json'
receipt = json.loads(receipt_path.read_bytes())
assert len(receipt['rows']) == 8
bindings = {str(receipt_path.resolve()):hashlib.sha256(receipt_path.read_bytes()).hexdigest()}
rows = []
for witness in receipt['rows']:
    path = saved/Path(witness['path']).name
    raw = path.read_bytes()
    assert len(raw) == witness['bytes'] and hashlib.sha256(raw).hexdigest() == witness['sha256']
    bindings[str(path.resolve())] = witness['sha256']
    with np.load(str(path),allow_pickle=False) as z:
        query = witness['selected_query']
        obs = int(np.flatnonzero(z['observed_queries']==query)[0])
        xyz = z['raw_points'][:,:3]
        mapping = z['superpoints']
        truth = z['actual_gt_masks'][0]
        predicted_sp = z['foreground_sp'][query]
        reconstructed_sp = z['alpha']*z['observed_text_logits'][obs]+(1-z['alpha'])*z['observed_query_logits'][obs]>0
        assert np.array_equal(predicted_sp,reconstructed_sp)
        foreground = predicted_sp[mapping]
        assert xyz.shape == (50000,3) and truth.shape == foreground.shape == (50000,)
        assert truth.any() and foreground.any() and z['reference_valid'][query]
        support_lo = xyz[foreground].min(0);support_hi = xyz[foreground].max(0)
        saved_box = np.r_[z['mask_center'][query],z['mask_size'][query]]
        reconstructed_box = np.r_[(support_lo+support_hi)/2,support_hi-support_lo]
        assert np.array_equal(saved_box,reconstructed_box)
        truth_lo = xyz[truth].min(0);truth_hi = xyz[truth].max(0)
        count = np.bincount(mapping,minlength=len(predicted_sp))
        root_count = np.bincount(mapping,weights=truth.astype(np.int64),minlength=len(predicted_sp))
        mixed_sp = (root_count>0)&(root_count<count)
        pure_background_sp = (count>0)&(root_count==0)
        false_positive = foreground&~truth
        false_positive_from_mixed = false_positive&mixed_sp[mapping]
        false_positive_from_pure_background = false_positive&pure_background_sp[mapping]
        assert false_positive.sum() == false_positive_from_mixed.sum()+false_positive_from_pure_background.sum()
        faces=[]
        for axis in range(3):
            for side in ('lower','upper'):
                value = support_lo[axis] if side=='lower' else support_hi[axis]
                boundary = truth_lo[axis] if side=='lower' else truth_hi[axis]
                outward = max(0.,float(boundary-value if side=='lower' else value-boundary))
                extrema = foreground&(xyz[:,axis]==value)
                faces.append(dict(axis=axis,side=side,
                    outward_vs_GT_member_extent_m=outward, extrema_points=int(extrema.sum()),
                    root_extrema_points=int((extrema&truth).sum()),
                    mixed_superpoint_false_extrema=int((extrema&false_positive_from_mixed).sum()),
                    pure_background_superpoint_false_extrema=int((extrema&false_positive_from_pure_background).sum())))
        intersection = int((foreground&truth).sum());union=int((foreground|truth).sum())
        training_box = z['actual_gt_boxes'][0]
        training_lo = training_box[:3]-training_box[3:]/2
        training_hi = training_box[:3]+training_box[3:]/2
        rows.append(dict(training_row_id=witness['row_id'],scene=witness['scene'],query=query,
            fusion_mask_iou=intersection/float(union),fusion_mask_good=2*intersection>union,
            selected_original_matched=bool((z['matched_queries']==query).any()),
            GT_points=int(truth.sum()),predicted_foreground_points=int(foreground.sum()),
            false_positive_points=int(false_positive.sum()),
            false_positive_points_from_mixed_superpoints=int(false_positive_from_mixed.sum()),
            false_positive_points_from_pure_background_superpoints=int(false_positive_from_pure_background.sum()),
            input_GT_member_extent_vs_training_box_max_face_gap_m=float(max(
                np.max(np.abs(truth_lo-training_lo)),np.max(np.abs(truth_hi-training_hi)))),
            max_outward_vs_GT_member_extent_m=max(face['outward_vs_GT_member_extent_m'] for face in faces),
            faces=faces))

good = [row for row in rows if row['fusion_mask_good']]
result=dict(scope='REAL_ONE_AUGMENTED_TRAINING_BATCH_POINT_MEMBERSHIP_PILOT',
    observations=8,unique_training_rows=8,steps_used=[1],input_sha256=bindings,
    model_forwards=0,optimizer_updates=0,GPU_calls=0,remote_queries=0,
    final_B_boxes_available=False,geometry='saved A-corrected Mask member range before B, not final deployed box',
    reference='actual root point membership extent, kept separate from actual training box',
    masks_good_rows=len(good),
    good_mask_rows_with_mixed_superpoint_false_points=sum(row['false_positive_points_from_mixed_superpoints']>0 for row in good),
    good_mask_rows_with_pure_background_superpoint_false_points=sum(row['false_positive_points_from_pure_background_superpoints']>0 for row in good),
    good_mask_rows_with_outward_extent=sum(row['max_outward_vs_GT_member_extent_m']>0 for row in good),
    good_mask_outward_faces_with_mixed_superpoint_false_extrema=sum(
        face['outward_vs_GT_member_extent_m']>0 and face['mixed_superpoint_false_extrema']>0
        for row in good for face in row['faces']),
    good_mask_outward_faces_with_pure_background_false_extrema=sum(
        face['outward_vs_GT_member_extent_m']>0 and face['pure_background_superpoint_false_extrema']>0
        for row in good for face in row['faces']),
    rows=rows,accuracy=None,
    interpretation='Small real training pilot, not a validation failure rate or causal proof; extreme points may tie across both categories, so face counts are not disjoint. No claim about final B outputs or noise as the sole cause.')
output.mkdir()
(output/'RESULT.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps({key:result[key] for key in ('scope','observations','masks_good_rows',
    'good_mask_rows_with_mixed_superpoint_false_points','good_mask_rows_with_pure_background_superpoint_false_points',
    'good_mask_rows_with_outward_extent','good_mask_outward_faces_with_mixed_superpoint_false_extrema',
    'good_mask_outward_faces_with_pure_background_false_extrema','model_forwards','accuracy')}))
