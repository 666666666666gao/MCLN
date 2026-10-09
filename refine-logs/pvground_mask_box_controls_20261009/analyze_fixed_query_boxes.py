"""Closed, same-forward geometry controls from the archived October 6 snapshot."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = Path(r'C:\Users\gb\.codex\tmp\pvground_mask_reference_20261006\complete_fit\fused_mask_reference\initial_formal')
N = 9508
NAMES = ('native_regression', 'mask_reference', 'fixed_half_blend')
THRESHOLDS = (.25, .5)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def iou(boxes, truth, dtype):
    boxes = np.asarray(boxes, dtype=dtype)
    truth = np.asarray(truth, dtype=dtype)
    low = np.maximum(boxes[:, :3] - boxes[:, 3:] / 2, truth[:, :3] - truth[:, 3:] / 2)
    high = np.minimum(boxes[:, :3] + boxes[:, 3:] / 2, truth[:, :3] + truth[:, 3:] / 2)
    intersection = np.maximum(high - low, 0).prod(axis=1)
    union = boxes[:, 3:].prod(axis=1) + truth[:, 3:].prod(axis=1) - intersection
    return intersection / union


def compare(before, after):
    return {str(t): dict(before_hits=int((before > t).sum()),
                        after_hits=int((after > t).sum()),
                        repairs=int(((before <= t) & (after > t)).sum()),
                        damages=int(((before > t) & (after <= t)).sum()),
                        net=int((after > t).sum() - (before > t).sum()))
            for t in THRESHOLDS}


def main():
    receipt = json.loads((SOURCE / 'receipt.json').read_bytes())
    rows_path = SOURCE / 'rows.jsonl'
    rows = [json.loads(line) for line in rows_path.read_text(encoding='utf-8').splitlines()]
    assert receipt['status'] == 'pass' and receipt['stage'] == 'initial_formal'
    assert receipt['primary_mode'] == 'bbs' and receipt['rows'] == receipt['formal_rows'] == N
    assert digest(rows_path) == receipt['rows_sha256']
    assert [row['row_id'] for row in rows] == list(range(N))
    assert all(row['same_forward_geometry_exact'] and row['native_head_calls'] == 1
               and row['diagnostic_native_head_replay_calls'] == 0 for row in rows)
    assert [sum(row['bbs']['iou'] > t for row in rows) for t in THRESHOLDS] == [5598, 4848]
    assert [receipt['metrics']['bbs']['rec_hits25'], receipt['metrics']['bbs']['rec_hits50']] == [5598, 4848]

    values = {name: [] for name in NAMES}
    values64 = {name: [] for name in NAMES}
    selected_records, manifest = [], []
    offset = 0
    for path in sorted(SOURCE.glob('batch_*.npz')):
        assert path.name == 'batch_%05d.npz' % offset
        with np.load(path, allow_pickle=False) as batch:
            assert set(batch.files) == {'row_ids', 'original_prior', 'reference', 'final',
                                        'scores', 'reference_valid', 'root_gt'}
            size = min(8, N - offset)
            np.testing.assert_array_equal(batch['row_ids'], np.arange(offset, offset + size))
            current = rows[offset:offset + size]
            truth = batch['root_gt']
            np.testing.assert_array_equal(truth, np.asarray([row['root_box'] for row in current], dtype=np.float32))
            queries = np.asarray([row['bbs']['query'] for row in current])
            indices = np.arange(size)
            scores = batch['scores']
            assert scores.shape == (size, 256) and np.isfinite(scores).all()
            np.testing.assert_array_equal(scores[indices, queries], scores.max(axis=1))
            for field in ('original_prior', 'reference', 'final'):
                assert batch[field].shape == (size, 256, 6) and batch[field].dtype == np.float32
                assert np.isfinite(batch[field]).all() and (batch[field][..., 3:] > 0).all()
            # In this specific snapshot the six-face output contributes no offset.
            np.testing.assert_array_equal(batch['final'], batch['reference'])
            native = batch['original_prior'][indices, queries]
            mask = batch['reference'][indices, queries]
            np.testing.assert_array_equal(native, np.asarray([row['bbs']['coarse_box'] for row in current], dtype=np.float32))
            np.testing.assert_array_equal(mask, np.asarray([row['bbs']['reference_box'] for row in current], dtype=np.float32))
            np.testing.assert_array_equal(batch['reference_valid'][indices, queries],
                                          [row['bbs']['reference_valid'] for row in current])
            # The coefficient is fixed by the named prior's equation, not searched.
            boxes = dict(native_regression=native, mask_reference=mask,
                         fixed_half_blend=(native + mask) * np.float32(.5))
            local32 = {name: iou(value, truth, np.float32) for name, value in boxes.items()}
            local64 = {name: iou(value, truth, np.float64) for name, value in boxes.items()}
            for name in NAMES:
                values[name].extend(local32[name].tolist())
                values64[name].extend(local64[name].tolist())
            for bid, row in enumerate(current):
                selected_records.append(dict(row_id=row['row_id'], scan_id=row['scan_id'],
                    target_id=row['target_id'], query=int(queries[bid]), point_sha256=row['point_sha256'],
                    root_box=truth[bid].tolist(), reference_valid=bool(row['bbs']['reference_valid']),
                    boxes={name: value[bid].tolist() for name, value in boxes.items()},
                    cpu_float32_iou={name: float(local32[name][bid]) for name in NAMES},
                    cpu_float64_iou={name: float(local64[name][bid]) for name in NAMES},
                    saved_gpu_native_iou=row['bbs']['coarse_iou'], saved_gpu_mask_iou=row['bbs']['reference_iou']))
        manifest.append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path)))
        offset += size
    assert offset == N and len(manifest) == 1189
    values = {name: np.asarray(value) for name, value in values.items()}
    values64 = {name: np.asarray(value) for name, value in values64.items()}
    table = []
    for name in NAMES:
        hits = [int((values[name] > t).sum()) for t in THRESHOLDS]
        table.append(dict(condition=name, rows=N, hits25=hits[0], hits50=hits[1],
            accuracy25=hits[0] / N * 100, accuracy50=hits[1] / N * 100,
            float64_hits=[int((values64[name] > t).sum()) for t in THRESHOLDS],
            float32_float64_threshold_flips={str(t): int(((values[name] > t) != (values64[name] > t)).sum()) for t in THRESHOLDS},
            meets_numerical_dual_gate=hits[0] >= 5658 and hits[1] >= 4850))
    saved = dict(native_regression=np.asarray([row['bbs']['coarse_iou'] for row in rows]),
                 mask_reference=np.asarray([row['bbs']['reference_iou'] for row in rows]))
    saved_flips = {name: {str(t): int(((values[name] > t) != (saved[name] > t)).sum())
                          for t in THRESHOLDS} for name in saved}
    rows_output = ROOT / 'SELECTED_BOX_ROWS.jsonl'
    rows_output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in selected_records), encoding='utf-8')
    manifest_path = ROOT / 'ARRAY_MANIFEST.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    summary = dict(status='EXECUTED_CLOSED_SAME_FORWARD_FIXED_QUERY_GEOMETRY_CONTROLS',
        generated_cst=datetime.datetime.now().astimezone().isoformat(), source_stage='20261006_initial_formal',
        source_directory=str(SOURCE), source_rows_sha256=digest(rows_path),
        source_receipt_sha256=digest(SOURCE / 'receipt.json'), source_script_sha256=digest(Path(__file__)),
        array_manifest_sha256=digest(manifest_path), output_rows_sha256=digest(rows_output),
        table=table, comparisons={
            'native_to_mask': compare(values['native_regression'], values['mask_reference']),
            'native_to_half_blend': compare(values['native_regression'], values['fixed_half_blend']),
            'mask_to_half_blend': compare(values['mask_reference'], values['fixed_half_blend'])},
        saved_gpu_cpu_float32_threshold_flips=saved_flips,
        invalid_selected_references=sum(not row['bbs']['reference_valid'] for row in rows),
        independent_selected_iou_values_per_precision=N * 3, coefficient_search=False,
        neural_forwards=0, optimizer_updates=0, ssh_queries=0, new_weights=0,
        retained_best_changed=False, live_training_changed=False, three_effective_contributions=False,
        full_goal_complete=False, review_status='PENDING_FRESH_ACTUAL_AUDIT',
        limits=[
            'This is the archived October 6 zero-offset reference snapshot, not the October 8 retained Mask-corrector checkpoint.',
            'The 50/50 condition applies only the named prior geometry formula to PV outputs; it does not reproduce the full EG-3DVG model.',
            'All three conditions reuse the identical saved native-bbs winner, input and dataset GT from one original forward.',
            'The new blended output is a CPU diagnostic, not a trained model result or a checkpoint eligible for promotion.',
            'Reference-invalid rows retain the existing archived original-prior behavior; no new fallback is added.',
            'All256 candidates were preserved by the original pipeline; this analysis evaluates selected geometry only, not a new candidate-coverage audit.',
            'No GT, threshold result or per-row oracle selects among the three fixed conditions at inference.'
        ])
    (ROOT / 'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=summary['status'], table=table, comparisons=summary['comparisons'],
                         saved_gpu_cpu_float32_threshold_flips=saved_flips), ensure_ascii=False))


if __name__ == '__main__':
    main()
