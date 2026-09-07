import gzip
import hashlib
import json
from pathlib import Path

directory = Path('C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/native_score_fit_20260907_v2')
analysis = json.loads((directory / 'independent_analysis.json').read_text())
assert analysis == json.loads((directory / 'local_independent_analysis.json').read_text())
rows = [row for path in sorted(directory.glob('rows_*.json.gz')) for row in json.loads(gzip.decompress(path.read_bytes()))]
changed = []
for row, summary in zip(rows, analysis['per_row']):
    if summary['membership_changed']:
        changed.append({'row_id': row['row_id'], 'scan_id': row['scan_id'], 'main_map_values': row['main_map_values'],
                        'native_iou': summary['native_iou'], 'old_best_iou': summary['original_oracle'],
                        'new_best_iou': summary['counterfactual_oracle'],
                        'removed': sorted(set(row['original_candidates']) - set(row['native_score_counterfactual_candidates'])),
                        'added': sorted(set(row['native_score_counterfactual_candidates']) - set(row['original_candidates']))})
proof = {'local_and_remote_analysis_json_equal': True,
         'fractional_map_rows': sum(any(v != 1 for v in row['main_map_values']) for row in rows),
         'score_difference_above_2e_6_rows': sum(max(abs(a - b) for a, b in zip(row['native_scores'], row['adapter_scores'])) > 2e-6 for row in rows),
         'max_score_difference': max(abs(a - b) for row in rows for a, b in zip(row['native_scores'], row['adapter_scores'])),
         'selected_scan_count': len(json.loads((directory / 'protocol.json').read_text())['selected_scans']),
         'changed_candidate_rows': changed, 'row_shards_bytes': sum(p.stat().st_size for p in directory.glob('rows_*.json.gz')),
         'analysis_sha256': hashlib.sha256((directory / 'independent_analysis.json').read_bytes()).hexdigest()}
(directory / 'local_verification.json').write_bytes(json.dumps(proof, indent=2).encode() + b'\n')
(directory / 'local_verify_from_source.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(proof), flush=True)
