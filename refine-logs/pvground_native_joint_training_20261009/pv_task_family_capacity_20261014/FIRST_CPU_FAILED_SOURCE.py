"""GT-only saved-data bound for B's three-axis box family; never deployed."""
import csv
import datetime
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def iou(boxes, targets):
    lower = np.maximum(boxes[:, :3] - boxes[:, 3:] / 2,
                       targets[:, :3] - targets[:, 3:] / 2)
    upper = np.minimum(boxes[:, :3] + boxes[:, 3:] / 2,
                       targets[:, :3] + targets[:, 3:] / 2)
    intersection = np.maximum(upper - lower, 0).prod(-1)
    union = boxes[:, 3:].prod(-1) + targets[:, 3:].prod(-1) - intersection
    assert (union > 0).all()
    return intersection / union


def family_max(native, mask, target):
    """Evaluate Cartesian products of all axis-wise overlap breakpoints.

    On each interval, with other axes fixed, IoU is affine/affine with positive
    denominator. Its derivative has constant sign. A maximum therefore occurs
    at a breakpoint on every axis. Four edge/GT crossings plus 0 and 1 suffice.
    This assumes positive endpoint sizes and ordinary unclamped geometric IoU.
    """
    assert native.shape == mask.shape == target.shape and native.shape[1] == 6
    assert all(np.isfinite(box).all() and (box[:, 3:] > 0).all()
               for box in (native, mask, target))
    count = len(target)
    candidates = np.zeros((count, 3, 6), dtype=np.float64)
    candidates[:, :, 1] = 1
    nlo, nhi = native[:, :3] - native[:, 3:] / 2, native[:, :3] + native[:, 3:] / 2
    mlo, mhi = mask[:, :3] - mask[:, 3:] / 2, mask[:, :3] + mask[:, 3:] / 2
    glo, ghi = target[:, :3] - target[:, 3:] / 2, target[:, :3] + target[:, 3:] / 2
    for index, (start, stop, boundary) in enumerate((
            (mlo, nlo, glo), (mlo, nlo, ghi), (mhi, nhi, glo), (mhi, nhi, ghi)), 2):
        slope = stop - start
        # An unchanged edge has no crossing event; duplicate endpoint0 suffices.
        np.divide(boundary - start, slope, out=candidates[:, :, index], where=slope != 0)
    candidates = np.clip(candidates, 0, 1)
    best = np.full(count, -1., dtype=np.float64)
    best_alpha = np.zeros((count, 3), dtype=np.float64)
    for x, y, z in itertools.product(range(6), repeat=3):
        alpha = np.stack((candidates[:, 0, x], candidates[:, 1, y], candidates[:, 2, z]), -1)
        weights = np.concatenate((alpha, alpha), -1)
        value = iou((1 - weights) * mask + weights * native, target)
        improved = value > best
        best[improved] = value[improved]
        best_alpha[improved] = alpha[improved]
    assert ((best >= 0) & (best <= 1 + 1e-12)).all()
    return best, best_alpha


def independent_checks():
    gt = np.array([[0., 0., 0., 1., 1., 1.]] * 5)
    mask = gt.copy()
    native = gt.copy()
    mask[1, 0] = 2
    native[2, 0] = 2
    native[3, 3] = 2
    mask[3, 4] = 2
    mask[4, 0], native[4, 0] = -.5, .5
    value, alpha = family_max(native, mask, gt)
    assert np.allclose(value, 1, atol=1e-12)
    assert np.allclose(alpha[3, :2], [0, 1])
    assert abs(alpha[4, 0] - .5) < 1e-12
    miss = gt[:1].copy()
    miss[:, 0] = 3
    assert family_max(miss, miss, gt[:1])[0][0] == 0
    random = np.random.RandomState(2027)
    target = np.concatenate((random.normal(size=(24, 3)),
                             np.exp(random.uniform(-1, 1, size=(24, 3)))), -1)
    n = target.copy()
    m = target.copy()
    n[:, :3] += random.normal(scale=.7, size=(24, 3))
    m[:, :3] += random.normal(scale=.7, size=(24, 3))
    n[:, 3:] *= np.exp(random.uniform(-.8, .8, size=(24, 3)))
    m[:, 3:] *= np.exp(random.uniform(-.8, .8, size=(24, 3)))
    exact, _ = family_max(n, m, target)
    dense = np.zeros(24)
    for a in itertools.product(np.linspace(0, 1, 13), repeat=3):
        weights = np.array(a + a)[None]
        dense = np.maximum(dense, iou((1 - weights) * m + weights * n, target))
    assert (exact + 1e-12 >= dense).all()
    return {'closed_form_cases': 6, 'random_cases': 24,
            'independent_dense_grid_triples_per_random_case': 13 ** 3,
            'minimum_exact_minus_dense': float((exact - dense).min())}


def main():
    root = Path('\\\\?\\' + str(Path(__file__).resolve().parent))
    output = root / 'selected_family_capacity_20261014'
    assert not output.exists()
    output.mkdir()
    checks = independent_checks()
    captured = root / 'OBS_selected_geometry_rows_20261014_files'
    reports = []
    bindings = {}
    for epoch in range(3):
        path, = captured.rglob('native_rows_epoch%03d.jsonl' % epoch)
        diagnostic_path, = captured.rglob('native_diagnostic_epoch%03d.json' % epoch)
        raw = path.read_bytes()
        diagnostic = json.loads(diagnostic_path.read_bytes())
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == diagnostic['row_file_sha256'] and len(raw) == diagnostic['row_file_bytes']
        bindings[str(path)] = {'bytes': len(raw), 'sha256': digest}
        rows = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
        assert len(rows) == 9508
        native, mask, target, final = (np.array([row[key] for row in rows], dtype=np.float64)
            for key in ('native_box', 'mask_endpoint_box', 'root_gt_box', 'selected_box'))
        assert min(target[:, 3:].prod(-1)) > 1e-6
        assert min(native[:, 3:].flat) >= 1e-6 and min(mask[:, 3:].flat) >= 1e-6
        delta = np.stack((native[:, :3] - mask[:, :3], native[:, 3:] - mask[:, 3:]), -1)
        moved = np.stack((final[:, :3] - mask[:, :3], final[:, 3:] - mask[:, 3:]), -1)
        denominator = (delta * delta).sum(-1)
        inferred = np.zeros_like(denominator)
        np.divide((delta * moved).sum(-1), denominator, out=inferred, where=denominator > 0)
        residual = moved - np.clip(inferred, 0, 1)[..., None] * delta
        assert abs(residual).max() < 1e-5
        selected_iou, native_iou, mask_iou = (iou(box, target) for box in (final, native, mask))
        best, alpha = family_max(native, mask, target)
        assert (best + 1e-5 >= selected_iou).all()
        qualified = np.array([2 * row['own_mask_intersection'] > row['own_mask_union'] and
            2 * row['fused_mask_intersection'] > row['fused_mask_union'] for row in rows])
        report = {'epoch': epoch, 'rows': 9508,
            'minimum_GT_volume': float(target[:, 3:].prod(-1).min()),
            'maximum_box_family_reconstruction_residual': float(abs(residual).max()),
            'maximum_saved_final_IoU_minus_ideal_family_max': float((selected_iou - best).max())}
        # Saved support qualification does not identify Hungarian eligibility.
        for suffix, threshold in (('025', .25), ('050', .5)):
            wrong = selected_iou <= threshold
            reachable = best > threshold
            endpoint_pass = (native_iou > threshold) | (mask_iou > threshold)
            support_bad_box = qualified & wrong
            assert int((~wrong).sum()) == diagnostic[suffix]['hits']
            report[suffix] = {'actual_selected_hits': int((~wrong).sum()),
                'GT_only_independent_axis_family_upper_bound_hits': int(reachable.sum()),
                'saved_errors_with_passing_same_selected_endpoint': int((wrong & endpoint_pass).sum()),
                'saved_errors_reachable_in_family': int((wrong & reachable).sum()),
                'saved_errors_not_reachable_in_family': int((wrong & ~reachable).sum()),
                'selected_own_and_fused_support_good_box_bad': int(support_bad_box.sum()),
                'support_good_box_bad_reachable': int((support_bad_box & reachable).sum()),
                'support_good_box_bad_not_reachable': int((support_bad_box & ~reachable).sum()),
                'family_max_within_1e_5_of_threshold': int((abs(best - threshold) < 1e-5).sum()),
                'float64_bound_threshold_losses_vs_saved_final': int((~wrong & ~reachable).sum())}
        with (output / ('epoch%03d.csv' % epoch)).open('x', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['row_index', 'selected_query', 'saved_final_IoU', 'native_IoU',
                'mask_endpoint_IoU', 'GT_only_family_max_IoU', 'GT_only_alpha_x', 'GT_only_alpha_y',
                'GT_only_alpha_z', 'own_and_fused_support_qualified'])
            for index, row in enumerate(rows):
                writer.writerow([index, row['selected_query'], selected_iou[index], native_iou[index],
                    mask_iou[index], best[index]] + alpha[index].tolist() + [bool(qualified[index])])
        reports.append(report)
    result = {'scope': 'CLOSED_C_FIXED_SELECTED_QUERY_GT_ONLY_THREE_AXIS_BOX_FAMILY_CAPACITY',
        'completed_cst': datetime.datetime.now().astimezone().isoformat(), 'epochs': reports,
        'source_rows': bindings, 'meaningful_numeric_checks': checks,
        'candidates_per_axis_max': 6, 'Cartesian_candidates_max': 216,
        'GT_used_to_choose_alpha': True, 'deployable': False, 'new_model_performance': False,
        'features_may_not_identify_the_GT_optimal_alpha': True,
        'not_full256_candidate_capacity': True, 'not_current_low_LR_new_result': True,
        'neural_forwards': 0, 'optimizer_updates': 0, 'remote_queries': 0,
        'production_source_changed': False, 'best_promoted': False, 'full_goal_complete': False}
    (output / 'RESULT.json').write_bytes((json.dumps(result, indent=2) + '\n').encode())
    print(json.dumps({'scope': result['scope'], 'checks': checks,
        'epochs': [{'epoch': report['epoch'], '025': report['025'], '050': report['050']}
                   for report in reports]}))


if __name__ == '__main__':
    main()
