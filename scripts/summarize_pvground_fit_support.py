"""Summarize completed fit-only support evidence without producing accuracy claims."""
import hashlib
import json
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
root = repo / 'refine-logs/pvground_fit_support_20260908_v2'
assert (root / 'controller.exit').read_text().strip() == '0'
receipt = json.loads((root / 'receipt.json').read_bytes())
assert receipt['status'] == 'complete' and receipt['forwards'] == 4 and receipt['checkpoint_unchanged']
assert hashlib.sha256((root / 'rows.jsonl').read_bytes()).hexdigest() == receipt['rows_sha256']
rows = [json.loads(line) for line in (root / 'rows.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(rows) == 32 and len({r['row_id'] for r in rows}) == 16
assert len({r['scan_id'].split('_')[0] for r in rows}) == 16
assert all(r['query_indices'] == list(range(0, 256, 8)) for r in rows)
keys = [(s['source'], s['radius_m']) for s in rows[0]['sources']]
assert len(keys) == 10 and len(set(keys)) == 10
assert all([(s['source'], s['radius_m']) for s in r['sources']] == keys for r in rows)
results = []
for augmented in [False, True]:
    cases = [r for r in rows if r['augmented'] == augmented]
    assert len(cases) == 16
    record = dict(augmented=augmented, scenes=16, input_points=16*50000,
                  points_outside_voxel_range=sum(r['points_outside_voxel_range'] for r in cases),
                  root_points=sum(r['root_points'] for r in cases),
                  root_points_outside_voxel_range=sum(r['root_points_outside_voxel_range'] for r in cases),
                  raw_nonpositive_size_queries=sum(r['raw_nonpositive_size_queries'] for r in cases),
                  source_groups=[])
    for index, (source, radius) in enumerate(keys):
        observations = [r['sources'][index] for r in cases]
        sampled_iou = [iou for r in cases for iou in r['sampled_root_iou']]
        counts = [n for s in observations for n in s['hypothetical_center_support_counts']]
        assert len(counts) == len(sampled_iou) == 512 and all(n >= 0 for n in counts)
        record['source_groups'].append(dict(source=source, radius_m=radius,
            actual_vsa_queries=sum(s['actual_vsa_queries'] for s in observations),
            actual_vsa_empty=sum(s['actual_vsa_empty'] for s in observations),
            sampled_candidate_centers=512, hypothetical_center_empty=sum(n == 0 for n in counts),
            sampled_good25=sum(i > .25 for i in sampled_iou),
            hypothetical_center_empty_among_good25=sum(n == 0 and i > .25 for n, i in zip(counts, sampled_iou)),
            gt_centers=16, hypothetical_gt_center_empty=sum(s['hypothetical_gt_center_support'] == 0 for s in observations)))
    fine_index = keys.index(('x_conv1', .2))
    coarse_index = keys.index(('x_conv3', .8))
    pairs = [(fine, coarse, iou) for r in cases for fine, coarse, iou in zip(
        r['sources'][fine_index]['hypothetical_center_support_counts'],
        r['sources'][coarse_index]['hypothetical_center_support_counts'], r['sampled_root_iou'])]
    record['fine_empty_coarse_nonempty'] = sum(f == 0 and c > 0 for f, c, _ in pairs)
    record['fine_empty_coarse_nonempty_good25'] = sum(f == 0 and c > 0 and i > .25 for f, c, i in pairs)
    results.append(record)
result = dict(scope='16 deterministic fit scenes, 32 source observations; not formal accuracy, not dataset-wide prevalence',
              receipt_sha256=hashlib.sha256((root / 'receipt.json').read_bytes()).hexdigest(),
              conditions=results, model_forwards=0, optimizer_steps=0, formal_rows=0)
(root / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result), flush=True)
