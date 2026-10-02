"""Summarize completed all256 evidence; no model, training or pruning."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=Path, required=True)
args = parser.parse_args()
root = args.directory
receipt = json.loads((root / 'receipt.json').read_bytes())
cpu = json.loads((root / 'CPU_RECOUNT.json').read_bytes())
assert receipt['status'] == 'pass' and cpu['status'] in ('pass', 'warn')
assert receipt['rows'] == cpu['rows'] == 9508
assert receipt['all_candidates_retained'] == cpu['all_candidates_retained'] == 256
raw = (root / 'rows.jsonl').read_bytes()
assert hashlib.sha256(raw).hexdigest() == receipt['rows_sha256']
rows = [json.loads(line) for line in raw.splitlines()]
assert len(rows) == 9508
summary = {}
for threshold, suffix in ((.25, '25'), (.5, '50')):
    errors = [r for r in rows if r['selected_iou'] <= threshold]
    bins = {'2-16': 0, '17-32': 0, '33-64': 0, '65-256': 0, 'none': 0}
    for row in errors:
        rank = row['first_qualified_rank'][suffix]
        if rank is None:
            bins['none'] += 1
        else:
            assert 2 <= rank <= 256
            key = ('2-16' if rank <= 16 else '17-32' if rank <= 32
                   else '33-64' if rank <= 64 else '65-256')
            bins[key] += 1
    group = cpu['error_partition'][suffix]
    assert len(errors) == group['errors'] == sum(bins.values())
    assert bins['none'] == group['no_qualified']
    summary[suffix] = dict(
        selected_hits=cpu['rec_hits'][suffix], errors=len(errors),
        **{k: v for k, v in group.items() if k != 'errors'},
        errors_with_alternative_percent=100 * group['has_alternative'] / len(errors),
        first_qualified_rank_bins=bins,
        first_qualified_beyond32=bins['33-64'] + bins['65-256'],
        qualified_candidate_counts=cpu['qualified_candidate_counts'][suffix])

examples = []
bands = {name: {'candidates': 0, 'expression_rows': 0} for name in
         ('weak_overlap_0_to_025', 'loose_only_025_to_050', 'strict_gt050')}
strict_errors_with_joint_box_mask_alternative = 0
for chunk in sorted(root.glob('candidates_*.npz')):
    data = np.load(chunk, allow_pickle=False)
    for offset, identity in enumerate(data['row_id']):
        row = rows[int(identity)]
        iou = data['root_iou'][offset]
        matched = data['matched_GT_slot'][offset]
        proxy = data['root_joint_best_scene_overlap'][offset]
        unmatched_proxy = (matched < 0) & proxy
        for name, condition in (
                ('weak_overlap_0_to_025', (iou > 0) & (iou <= .25)),
                ('loose_only_025_to_050', (iou > .25) & (iou <= .5)),
                ('strict_gt050', iou > .5)):
            count = int((unmatched_proxy & condition).sum())
            bands[name]['candidates'] += count
            bands[name]['expression_rows'] += int(count > 0)
        if row['selected_iou'] <= .5:
            strict_errors_with_joint_box_mask_alternative += int((
                unmatched_proxy & (iou > .5) & (data['root_mask_iou'][offset] > .5)).any())
            usable = np.flatnonzero((iou > .5) & (matched < 0) & proxy)
            if len(usable) and len(examples) < 8:
                query = int(usable[iou[usable].argmax()])
                scores = data['bbs_scores'][offset]
                examples.append(dict(
                    row_id=row['row_id'], scan_id=row['scan_id'], target_id=row['target_id'],
                    utterance=row['utterance'], selected_query=row['selected_query'],
                    selected_iou=row['selected_iou'], alternative_query=query,
                    alternative_iou=float(iou[query]),
                    alternative_mask_iou=float(data['root_mask_iou'][offset, query]),
                    alternative_matched_slot=int(matched[query]),
                    score_rank_tie_lower=1 + int((scores > scores[query]).sum()),
                    score_rank_tie_upper=int((scores >= scores[query]).sum()),
                    root_in_scene_detection_GT=row['root_in_scene_detection_GT'],
                    identity_rule='Root IoU > .5 and root ties/exceeds filtered-scene GT maximum; geometric proxy only.'))
assert len(examples) == 8
result = dict(
    status=cpu['status'], rows=9508, candidates_per_row=256, retained_candidates=9508 * 256,
    optimizer_updates=receipt['optimizer_updates'], model_state_unchanged=receipt['model_state_unchanged'],
    thresholds=summary, root_excluded_from_scene_detection_GT_rows=cpu['root_excluded_from_scene_detection_GT_rows'],
    unmatched_root_overlap_proxy_bands=bands,
    strict_errors_with_unmatched_joint_box_mask_alternative=strict_errors_with_joint_box_mask_alternative,
    versus_archived_formal=cpu['versus_archived_formal'],
    CPU_all_box_thresholds_exact=cpu['CPU_all_box_thresholds_exact'],
    CPU_all_box_thresholds_exact_by_threshold=cpu['CPU_all_box_thresholds_exact_by_threshold'],
    CPU_threshold_differences=cpu['CPU_threshold_differences'],
    examples=examples, evidence_limits=receipt['evidence_limits'],
    source_rows_sha256=receipt['rows_sha256'],
    CPU_recount_sha256=hashlib.sha256((root / 'CPU_RECOUNT.json').read_bytes()).hexdigest(),
    policy='Retain all256, including unmatched and low-ranked candidates. GT qualification is offline evidence, never an inference gate or validation-training label. No blanket positive conversion or new supervision.')
(root / 'SUMMARY.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k != 'examples'}, ensure_ascii=False))
