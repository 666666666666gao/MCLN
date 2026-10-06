"""CPU verification of actual selected-Query extent evidence."""
import argparse
import json
from pathlib import Path
import numpy as np


def iou(box, target):
    if box is None:
        return 0.
    a = np.asarray(box, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    intersection = np.maximum(0, np.minimum(a[:3] + a[3:]/2, b[:3] + b[3:]/2)
                              - np.maximum(a[:3] - a[3:]/2, b[:3] - b[3:]/2)).prod()
    return float(intersection / (a[3:].prod() + b[3:].prod() - intersection))


def as_box(lower, upper):
    return None if not np.all(upper > lower) else np.concatenate(((lower+upper)/2, upper-lower))


parser = argparse.ArgumentParser()
parser.add_argument('--directory', required=True, type=Path)
args = parser.parse_args()
root = args.directory
receipt = json.loads((root/'receipt.json').read_bytes())
assert receipt['status'] == 'pass' and receipt['model_states_unchanged'] and receipt['optimizer_updates'] == 0
rows = [json.loads(line) for line in (root/'rows.jsonl').read_text().splitlines()]
assert len(rows) == receipt['rows'] and [row['row_id'] for row in rows] == list(range(len(rows)))
flips = {name: {str(t): 0 for t in (.25, .5)} for name in ('learned', 'exact', 'quantile')}
raw_replayed = 0
cpu_records = []
for start in range(0, len(rows), 8):
    with np.load(str(root / ('batch_%04d.npz' % (start//8))), allow_pickle=False) as payload:
        for row in rows[start:start+8]:
            prefix = 'r' + str(row['row_id']) + '__'
            arr = {name: payload[prefix+name] for name in ('ids','count','lower','upper','target_count',
                'own_active','fused_active','quantile_order_values','quantile_order_ranks')}
            active = arr['fused_active']
            arr['lower'] = arr['lower'].astype(np.float64)
            arr['upper'] = arr['upper'].astype(np.float64)
            assert arr['count'].sum() == 50000 and (arr['target_count'] <= arr['count']).all()
            foreground = int(arr['count'][active].sum())
            assert foreground == row['foreground_count']
            intersection = int(arr['target_count'][active].sum())
            mask_iou = intersection / (foreground + int(arr['target_count'].sum()) - intersection)
            assert abs(mask_iou - row['fused_mask_iou']) < 1e-12
            exact = None if not active.any() else as_box(arr['lower'][active].min(0), arr['upper'][active].max(0))
            assert (exact is None) == (row['exact_box'] is None)
            if exact is not None:
                assert np.array_equal(exact, row['exact_box'])
            if foreground == 0:
                trimmed = None
            else:
                positions = (foreground-1) * np.array([.005, .995])
                ranks = arr['quantile_order_ranks']
                assert np.array_equal(ranks, np.stack((np.floor(positions), np.ceil(positions)), 1))
                fraction = positions % 1
                values = arr['quantile_order_values']
                limits = values[:, 0] + (values[:, 1] - values[:, 0]) * fraction[:, None]
                trimmed = as_box(limits[0], limits[1])
            assert (trimmed is None) == (row['quantile_box'] is None)
            if trimmed is not None:
                assert np.allclose(trimmed, row['quantile_box'], rtol=0, atol=1e-12)
            if receipt['mode'] == 'preflight':
                points = payload[prefix+'point_clouds']
                sp = payload[prefix+'superpoint']
                truth = payload[prefix+'gt_point_mask']
                ids, inverse, counts = np.unique(sp, return_inverse=True, return_counts=True)
                assert np.array_equal(ids, arr['ids']) and np.array_equal(counts, arr['count'])
                point_active = active[inverse]
                assert abs(float((point_active & truth).sum() / (point_active | truth).sum()) - mask_iou) < 1e-12
                for index, slot in enumerate(ids):
                    members = points[sp == slot, :3].astype(np.float64)
                    assert np.array_equal(members.min(0), arr['lower'][index])
                    assert np.array_equal(members.max(0), arr['upper'][index])
                    assert int(truth[sp == slot].sum()) == int(arr['target_count'][index])
                if foreground:
                    xyz = points[point_active, :3].astype(np.float64)
                    raw_exact = as_box(xyz.min(0), xyz.max(0))
                    assert (raw_exact is None) == (exact is None)
                    if exact is not None:
                        assert np.array_equal(raw_exact, exact)
                    quantiles = np.quantile(xyz, [.005, .995], axis=0)
                    raw_trimmed = as_box(quantiles[0], quantiles[1])
                    assert (raw_trimmed is None) == (trimmed is None)
                    if trimmed is not None:
                        assert np.allclose(raw_trimmed, trimmed, rtol=0, atol=1e-12)
                raw_replayed += 1
            cpu = {name: iou(row[name+'_box'], row['root_box']) for name in ('learned', 'exact', 'quantile')}
            for name, value in cpu.items():
                assert abs(value - row[name+'_iou']) < 1e-5
                for t in (.25, .5):
                    flips[name][str(t)] += int((value > t) != (row[name+'_iou'] > t))
            cpu_records.append(cpu)

comparisons = {}
for name in ('exact', 'quantile'):
    comparisons[name] = {}
    for t in (.25, .5):
        before = np.array([row['learned'] > t for row in cpu_records])
        after = np.array([row[name] > t for row in cpu_records])
        repairs = int((after & ~before).sum())
        damages = int((before & ~after).sum())
        comparisons[name][str(t)] = dict(before_hits=int(before.sum()), after_hits=int(after.sum()),
                                        repairs=repairs, damages=damages, net=repairs-damages)
mask_good = np.array([row['fused_mask_iou'] > .5 for row in rows])
learned_good = np.array([row['learned'] > .5 for row in cpu_records])
volumes = np.asarray([np.prod(row['root_box'][3:]) for row in rows])
cuts = np.quantile(volumes, [.25, .5, .75])
volume_group = np.searchsorted(cuts, volumes, side='right')
groups = {}
for label, chosen in [('all_mask_good', np.flatnonzero(mask_good))] + [
        ('volume_Q'+str(index+1), np.flatnonzero(volume_group == index)) for index in range(4)]:
    metrics = {}
    before = learned_good[chosen]
    for name in ('exact','quantile'):
        after = np.asarray([cpu_records[index][name] > .5 for index in chosen])
        repairs = int((after & ~before).sum())
        damages = int((before & ~after).sum())
        metrics[name] = dict(repairs=repairs, damages=damages, net=repairs-damages,
                             learned_hits=int(before.sum()), extent_hits=int(after.sum()))
    groups[label] = dict(rows=len(chosen), strict=metrics)
result = dict(status='ACTUAL_CPU_EXTENT_ROWS_VERIFIED', rows=len(rows), comparisons=comparisons,
    CPU_stored_threshold_flips=flips, preflight_actual_raw_member_rows_replayed=raw_replayed,
    mask_good_box_bad=int((mask_good & ~learned_good).sum()),
    history_query_changes=receipt['history_query_changes'], history_hit50_changes=receipt['history_hit50_changes'],
    invalid_exact=receipt['invalid_exact'], invalid_quantile=receipt['invalid_quantile'],
    strict_groups=groups, expression_weighted_GT_volume_quartile_cuts=cuts.tolist(),
    quartile_tie_assignment='equal to boundary assigned to higher group',
    zero_optimization=True, full_formal_raw_point_sorting_replayed=False,
    quantile_formal_scope='Stored actual neighboring order-statistic values and rank decoding; raw XYZ independently replayed only in M0.')
destination = root/'CPU_SUMMARY.json'
assert not destination.exists()
destination.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
