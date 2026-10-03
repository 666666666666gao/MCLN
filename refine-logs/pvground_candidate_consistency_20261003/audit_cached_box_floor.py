"""Measure size-floor incidence in the existing, completed P3 fused cache."""
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    source = Path(r'C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\candidate_audit\full_gt_scope_v2')
    output = Path(__file__).parent/'cached_p3_box_floor'
    assert not output.exists()
    receipt = json.loads((source/'receipt.json').read_bytes())
    intake = json.loads((source/'INTAKE.json').read_bytes())
    raw = (source/'rows.jsonl').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == receipt['rows_sha256']
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == receipt['rows'] == 9508
    assert receipt['parent_arm'] == 'tail_fused'
    floor = np.float32(1e-6)
    counts = {name: 0 for name in (
        'candidates', 'coarse_floor_candidates', 'final_floor_candidates',
        'introduced_floor_candidates', 'removed_floor_candidates',
        'rows_with_final_floor_candidate', 'selected_coarse_floor_rows',
        'selected_final_floor_rows', 'selected_introduced_floor_rows',
        'matched_root_final_floor_rows', 'qualified50_final_floor_candidates',
        'selected_floor_with_qualified50_alternative', 'selected_floor_strict_errors',
    )}
    selected_minima = []
    selected_floor_examples = []
    chunks = {}
    index = 0
    hits25 = hits50 = 0
    for entry in intake['chunks']:
        path = source/entry['name']
        content = path.read_bytes()
        assert len(content) == entry['bytes']
        chunks[entry['name']] = hashlib.sha256(content).hexdigest()
        with np.load(path, allow_pickle=False) as data:
            boxes = data['boxes']
            coarse = data['coarse_boxes']
            assert boxes.shape[1:] == (256, 6) and coarse.shape == boxes.shape
            assert boxes.dtype == coarse.dtype == np.float32
            assert np.isfinite(boxes).all() and np.isfinite(coarse).all()
            assert (boxes[:, :, 3:] >= floor).all() and (coarse[:, :, 3:] >= floor).all()
            for offset, identity in enumerate(data['row_id']):
                row = rows[index]
                assert int(identity) == row['row_id'] == index
                query = row['selected_query']
                before = (coarse[offset, :, 3:] == floor).any(axis=1)
                after = (boxes[offset, :, 3:] == floor).any(axis=1)
                good50 = data['root_iou'][offset] > .5
                assert float(data['root_iou'][offset, query]) == row['selected_iou']
                counts['candidates'] += 256
                counts['coarse_floor_candidates'] += int(before.sum())
                counts['final_floor_candidates'] += int(after.sum())
                counts['introduced_floor_candidates'] += int((after & ~before).sum())
                counts['removed_floor_candidates'] += int((before & ~after).sum())
                counts['rows_with_final_floor_candidate'] += int(after.any())
                counts['selected_coarse_floor_rows'] += int(before[query])
                counts['selected_final_floor_rows'] += int(after[query])
                counts['selected_introduced_floor_rows'] += int(after[query] and not before[query])
                counts['matched_root_final_floor_rows'] += int((after & (data['matched_GT_slot'][offset] == 0)).sum())
                counts['qualified50_final_floor_candidates'] += int((after & good50).sum())
                if after[query]:
                    counts['selected_floor_strict_errors'] += int(not good50[query])
                    counts['selected_floor_with_qualified50_alternative'] += int(good50.any())
                    selected_floor_examples.append({
                        'row_id': index, 'scan_id': row['scan_id'], 'target_id': row['target_id'],
                        'query': query, 'selected_iou': row['selected_iou'],
                        'coarse_size': coarse[offset, query, 3:].tolist(),
                        'final_size': boxes[offset, query, 3:].tolist(),
                        'first_qualified_rank50': row['first_qualified_rank']['50'],
                    })
                selected_minima.append(float(boxes[offset, query, 3:].min()))
                hits25 += int(row['selected_iou'] > .25)
                hits50 += int(row['selected_iou'] > .5)
                index += 1
    assert index == 9508 and counts['candidates'] == 2434048
    assert len(chunks) == receipt['chunks'] == 1189
    assert {'25': hits25, '50': hits50} == receipt['rec_hits']
    result = {
        'status': 'complete_cached_CPU_analysis', 'source': str(source),
        'model': 'completed_tail_fused_negative_result', 'rows': index,
        'selected_rec_hits25': hits25, 'selected_rec_hits50': hits50,
        'counts': counts, 'selected_minimum_size_quantiles': {
            str(q): float(np.quantile(selected_minima, q)) for q in (0, .01, .1, .5, .9, .99, 1)},
        'selected_floor_examples': selected_floor_examples,
        'raw_preclamp_sizes_available': False,
        'interpretation_limits': [
            'Only evaluator-clamped sizes are cached; a floor hit does not recover the original sign.',
            'This is the old fused P3 checkpoint, not the active G candidate-consistency pair.',
            'Removing a floor hit is not evidence that the same expression would become correct.',
            'All 256 slots are counted and retained; no model or inference filtering is changed.',
        ],
        'optimizer_updates': 0, 'GPU_forward_executed': False,
        'rows_sha256': receipt['rows_sha256'], 'chunk_sha256': chunks,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output.mkdir()
    (output/'SUMMARY.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('status','rows','selected_rec_hits25','selected_rec_hits50','counts','selected_minimum_size_quantiles')}, indent=2))


if __name__ == '__main__':
    main()
