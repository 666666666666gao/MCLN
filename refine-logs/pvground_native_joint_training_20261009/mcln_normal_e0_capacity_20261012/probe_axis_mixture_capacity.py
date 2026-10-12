"""GT-only CPU probe of the saved native/reference three-axis mixing family."""
import hashlib
import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
raw = (root / 'full_paired_rows.jsonl').read_bytes()
binding = json.loads((root / 'full_ROWS_TRANSFER.json').read_bytes())
assert len(raw) == binding['bytes'] and hashlib.sha256(raw).hexdigest() == binding['sha256']
rows = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
assert len(rows) == 9508
native = np.asarray([r['native']['selected_box'] for r in rows], dtype=np.float32)
reference = np.asarray([r['mask_reference']['selected_box'] for r in rows], dtype=np.float32)
truth = np.asarray([r['native']['root_gt_box'] for r in rows], dtype=np.float32)
native_saved = np.asarray([r['native']['selected_iou'] for r in rows], dtype=np.float32)
reference_saved = np.asarray([r['mask_reference']['selected_iou'] for r in rows], dtype=np.float32)
full_saved = np.asarray([r['full_transfer']['selected_iou'] for r in rows], dtype=np.float32)
assert all(r['native']['selected_query'] == r['mask_reference']['selected_query'] == r['full_transfer']['selected_query'] for r in rows)


def iou(boxes, target):
    size = np.maximum(boxes[..., 3:], np.float32(1e-6))
    target_size = np.maximum(target[..., 3:], np.float32(1e-6))
    low, high = boxes[..., :3] - size * np.float32(.5), boxes[..., :3] + size * np.float32(.5)
    gt_low, gt_high = target[..., :3] - target_size * np.float32(.5), target[..., :3] + target_size * np.float32(.5)
    overlap = np.maximum(np.minimum(high, gt_high) - np.maximum(low, gt_low), np.float32(0))
    extent, gt_extent = high - low, gt_high - gt_low
    intersection = overlap[..., 0] * overlap[..., 1] * overlap[..., 2]
    volume = extent[..., 0] * extent[..., 1] * extent[..., 2]
    gt_volume = gt_extent[..., 0] * gt_extent[..., 1] * gt_extent[..., 2]
    return intersection / (volume + gt_volume - intersection)


assert np.array_equal(iou(native, truth), native_saved)
assert np.array_equal(iou(reference, truth), reference_saved)
nodes = np.linspace(0, 1, 9, dtype=np.float32)
gates = np.stack(np.meshgrid(nodes, nodes, nodes, indexing='ij'), -1).reshape(-1, 3)
assert gates.shape == (729, 3)
best_iou, best_gate = [], []
for start in range(0, len(rows), 128):
    n, r, gt = native[start:start+128, None], reference[start:start+128, None], truth[start:start+128, None]
    centers = (1 - gates[None]) * r[..., :3] + gates[None] * n[..., :3]
    sizes = (1 - gates[None]) * r[..., 3:] + gates[None] * n[..., 3:]
    values = iou(np.concatenate([centers, sizes], -1), gt)
    indices = values.argmax(-1)
    best_iou.extend(values[np.arange(len(n)), indices].tolist())
    best_gate.extend(gates[indices].tolist())
best_iou = np.asarray(best_iou, dtype=np.float32)
best_gate = np.asarray(best_gate, dtype=np.float32)
assert np.all(best_iou >= np.maximum(native_saved, reference_saved))
table = []
for threshold, suffix in ((.25, '025'), (.5, '050')):
    native_good, reference_good = native_saved > threshold, reference_saved > threshold
    union = native_good | reference_good
    grid_good = best_iou > threshold
    table.append(dict(threshold=threshold, native_hits=int(native_good.sum()), reference_hits=int(reference_good.sum()),
                      deployed_full_transfer_hits=int((full_saved > threshold).sum()),
                      offline_GT_whole_box_choice_hits=int(union.sum()),
                      offline_GT_axis_grid_reachable_hits=int(grid_good.sum()),
                      extra_reachable_beyond_whole_box_choice=int((grid_good & ~union).sum()),
                      native_only_correct=int((native_good & ~reference_good).sum()),
                      reference_only_correct=int((reference_good & ~native_good).sum()),
                      both_incorrect=int((~union).sum()),
                      GT_grid_gap_to_deployed_full=int(grid_good.sum()) - int((full_saved > threshold).sum())))
examples = []
eligible = (reference_saved <= .5) & (best_iou > .5)
indices = np.flatnonzero(eligible)
indices = indices[np.argsort(-(best_iou[indices] - reference_saved[indices]))][:12]
for index in indices:
    row = rows[int(index)]
    examples.append(dict(index=int(index), scan_id=row['scan_id'],
                         selected_query=row['native']['selected_query'],
                         native_iou=float(native_saved[index]), reference_iou=float(reference_saved[index]),
                         deployed_full_iou=float(full_saved[index]), GT_axis_grid_iou=float(best_iou[index]),
                         GT_axis_gate=best_gate[index].tolist(),
                         root_gt_box=truth[index].tolist(), native_box=native[index].tolist(), reference_box=reference[index].tolist()))
result = dict(status='CPU_SAVED_SELECTED_QUERY_GT_AXIS_MIXING_REPRESENTATION_PROBE',
              rows=9508, grid_nodes=nodes.tolist(), gate_combinations=729, table=table, examples=examples,
              paired_rows_sha256=binding['sha256'], selected_query_unchanged=True,
              new_neural_forwards=0, new_optimizer_updates=0, inference_changed=False,
              full_goal_complete=False, method_accuracy_claim=False,
              limitations=[
                  'GT chooses the gate offline; these are not deployable scores, model results or promised gains.',
                  'This finite grid shows reachable members of the family, not an exact continuous-IoU upper bound.',
                  'Native and uncorrected fused-Mask reference are the saved endpoints; A-corrected references before B were not saved.',
                  'Geometry overlap does not establish physical instance identity or learned gate predictability.',
                  'New formal training has its own E0; this saved replay does not replace its validation.'])
(root / 'AXIS_MIXTURE_CAPACITY_PROBE.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(status=result['status'], table=table, method_accuracy_claim=False)))
