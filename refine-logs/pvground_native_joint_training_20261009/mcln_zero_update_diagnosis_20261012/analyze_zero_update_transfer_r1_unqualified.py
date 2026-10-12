"""Analyze retained real validation rows; never run a model or query a server."""
import hashlib
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
mcln_path = root / 'full_paired_rows.jsonl'
pv_path = Path('C:/Users/gb/.codex/tmp/pvground_extremal_span_evidence_20261009/runner_v1/complete_fit/formal/rows.jsonl')
output = root / 'zero_update_diagnosis_20261012'
assert not output.exists()
mcln = [json.loads(line) for line in mcln_path.read_text(encoding='utf-8').splitlines()]
pv = [json.loads(line) for line in pv_path.read_text(encoding='utf-8').splitlines()]
assert len(mcln) == len(pv) == 9508
arms = ('native', 'mask_reference', 'full_transfer')
for i, (left, right) in enumerate(zip(mcln, pv)):
    assert left['index'] == right['row_id'] == i
    assert (left['scan_id'], left['target_id']) == (right['scan_id'], right['target_id'])
    assert left['point_sha256'] == right['point_sha256']
    assert left['native']['root_gt_box'] == right['root_box']
    assert all(left[arm]['root_gt_box'] == left['native']['root_gt_box'] for arm in arms)
    assert all(left[arm]['selected_query'] == left['native']['selected_query'] for arm in arms)

volumes = [row['native']['root_gt_box'][3] * row['native']['root_gt_box'][4]
           * row['native']['root_gt_box'][5] for row in mcln]
assert min(volumes) > 0
cutoffs = statistics.quantiles(volumes, n=4, method='inclusive')
volume_groups = [[i for i, value in enumerate(volumes)
                  if sum(value > cutoff for cutoff in cutoffs) == group] for group in range(4)]

def comparison(before, after, indices, threshold):
    before_hits = sum(before[i] > threshold for i in indices)
    after_hits = sum(after[i] > threshold for i in indices)
    repairs = sum(before[i] <= threshold < after[i] for i in indices)
    breaks = sum(after[i] <= threshold < before[i] for i in indices)
    assert after_hits - before_hits == repairs - breaks
    return dict(rows=len(indices), before_hits=before_hits, after_hits=after_hits,
                repairs=repairs, breaks=breaks, net=repairs-breaks)

def faces(box):
    return [box[axis] + sign * box[axis+3] / 2
            for axis in range(3) for sign in (-1, 1)]

ious = {arm: [row[arm]['selected_iou'] for row in mcln] for arm in arms}
pv_ious = [row['arms']['extremal_support']['iou'] for row in pv]
assert [sum(value > threshold for value in pv_ious) for threshold in (0.25, 0.5)] == [5677, 4920]
assert {arm: [sum(value > threshold for value in values) for threshold in (0.25, 0.5)]
        for arm, values in ious.items()} == dict(native=[5515,4409], mask_reference=[5459,4767], full_transfer=[5473,4777])
all_indices = list(range(9508))
transitions = [(arms[0],arms[1]), (arms[0],arms[2]), (arms[1],arms[2])]
result = dict(scope='REAL_9508_RETAINED_ZERO_UPDATE_GEOMETRY_DIAGNOSIS',
    input_sha256={str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in (mcln_path,pv_path)}, rows=9508,
    same_expression_order_scene_target_GT_box_and_pointcloud_sha=True,
    same_MCLN_selected_Query_all_three_arms=True, new_model_forwards=0,
    new_optimizer_updates=0, new_remote_queries=0, thresholds=[0.25,0.5],
    gt_volume_cutoffs_m3=cutoffs, volume_group_sizes=[len(group) for group in volume_groups],
    whole={}, by_GT_volume={}, cross_model={}, formal_epoch1_explained=False,
    interpretation='MCLN within-checkpoint zero-update geometry changes; PV versus MCLN uses different trained models and cannot isolate a module cause; GT groups are offline diagnostics only.')

for suffix, threshold in (('025',0.25),('050',0.5)):
    result['whole'][suffix] = {before+'__to__'+after: comparison(ious[before],ious[after],all_indices,threshold)
                              for before,after in transitions}
    result['by_GT_volume'][suffix] = [
        dict(group=group+1, rows=len(indices), min_volume_m3=min(volumes[i] for i in indices),
             max_volume_m3=max(volumes[i] for i in indices),
             transitions={before+'__to__'+after: comparison(ious[before],ious[after],indices,threshold)
                          for before,after in transitions})
        for group, indices in enumerate(volume_groups)]
    counts = dict(both_correct=0, MCLN_only_correct=0, PV_only_correct=0, both_wrong=0)
    for mi, pi in zip(ious['full_transfer'],pv_ious):
        key = 'both_correct' if mi>threshold and pi>threshold else (
              'MCLN_only_correct' if mi>threshold else 'PV_only_correct' if pi>threshold else 'both_wrong')
        counts[key] += 1
    counts['PV_minus_MCLN_hits'] = counts['PV_only_correct'] - counts['MCLN_only_correct']
    result['cross_model'][suffix] = counts

support_groups = dict(both_masks_good=[], own_only_good=[], fused_only_good=[], neither_good=[])
for i,row in enumerate(mcln):
    own = row['full_transfer']['selected_own_mask_iou'] > 0.5
    fused = row['full_transfer']['selected_fused_mask_iou'] > 0.5
    key = 'both_masks_good' if own and fused else (
          'own_only_good' if own else 'fused_only_good' if fused else 'neither_good')
    support_groups[key].append(i)
result['by_selected_MCLN_mask_support'] = {}
for name,indices in support_groups.items():
    result['by_selected_MCLN_mask_support'][name] = dict(rows=len(indices),
        strict_hits={arm:sum(ious[arm][i]>0.5 for i in indices) for arm in arms},
        strict_changes={before+'__to__'+after: comparison(ious[before],ious[after],indices,0.5)
                        for before,after in transitions},
        qualified_candidate_available_but_not_selected=sum(
            ious['full_transfer'][i]<=0.5<mcln[i]['full_transfer']['full256_max_iou'] for i in indices))

damage = [i for i in all_indices if ious['mask_reference'][i]<=0.5<ious['native'][i]]
repair = [i for i in all_indices if ious['native'][i]<=0.5<ious['mask_reference'][i]]
result['mask_reference_strict_transition_geometry'] = {}
for name,indices in (('repairs',repair),('breaks',damage)):
    ratios = []
    gt_face_errors = []
    changes = []
    for i in indices:
        row = mcln[i]
        native = row['native']['selected_box']; mask = row['mask_reference']['selected_box']
        ratios.append(mask[3]*mask[4]*mask[5] / volumes[i])
        gt_face_errors.append(max(abs(a-b) for a,b in zip(faces(mask),faces(row['native']['root_gt_box']))))
        changes.append(max(abs(a-b) for a,b in zip(faces(mask),faces(native))))
    result['mask_reference_strict_transition_geometry'][name] = dict(rows=len(indices),
        mask_reference_to_GT_volume_ratio_median=statistics.median(ratios),
        mask_reference_max_face_error_median_m=statistics.median(gt_face_errors),
        native_to_mask_max_face_change_median_m=statistics.median(changes),
        selected_own_mask_good=sum(mcln[i]['mask_reference']['selected_own_mask_iou']>0.5 for i in indices),
        selected_fused_mask_good=sum(mcln[i]['mask_reference']['selected_fused_mask_iou']>0.5 for i in indices))

misses = [i for i in all_indices if ious['full_transfer'][i] <= 0.5]
result['MCLN_full_transfer_strict_errors'] = dict(rows=len(misses),
    with_other_qualified_candidate=sum(mcln[i]['full_transfer']['full256_max_iou']>0.5 for i in misses),
    no_qualified_candidate=sum(mcln[i]['full_transfer']['full256_max_iou']<=0.5 for i in misses),
    own_and_fused_masks_good=sum(i in support_groups['both_masks_good'] for i in misses))
pv_only = [i for i in all_indices if ious['full_transfer'][i]<=0.5<pv_ious[i]]
mcln_only = [i for i in all_indices if pv_ious[i]<=0.5<ious['full_transfer'][i]]
result['cross_model_strict_support'] = {
    name:dict(rows=len(indices),
        MCLN_own_and_fused_masks_good=sum(i in support_groups['both_masks_good'] for i in indices),
        PV_selected_mask_good=sum(pv[i]['mask_iou']>0.5 for i in indices),
        MCLN_has_qualified_candidate=sum(mcln[i]['full_transfer']['full256_max_iou']>0.5 for i in indices))
    for name,indices in (('PV_only_correct',pv_only),('MCLN_only_correct',mcln_only))}
output.mkdir()
(output/'RESULT.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps({name:result[name] for name in ('scope','rows','whole','cross_model',
    'MCLN_full_transfer_strict_errors','cross_model_strict_support',
    'mask_reference_strict_transition_geometry')},ensure_ascii=False))
