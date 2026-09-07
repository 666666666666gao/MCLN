"""Read-only alignment census on the already verified 512 ScanRefer fit rows."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = (args.directory / 'rows.json').read_bytes()
    receipt = json.loads((args.directory / 'receipt.json').read_text())
    assert hashlib.sha256(raw).hexdigest() == receipt['rows_sha256']
    rows = json.loads(raw)
    assert len(rows) == 512
    eligible = [row for row in rows if row['ious']['teacher'] > .25
                and row['ious']['teacher'] > row['ious']['hungarian_root']]
    assert len(eligible) == 342
    groups = {}
    for name, selected in [('all_fit', rows), ('eligible_teacher_geometry', eligible)]:
        counts = {'rows': len(selected),
            'native_selected_equals_current_gt_root_query': sum(row['native_query_index'] == row['hungarian_root_query_index'] for row in selected),
            'teacher_source_equals_current_gt_root_query': sum(row['teacher_source_query_index'] == row['hungarian_root_query_index'] for row in selected),
            'teacher_geometry_nearest_equals_current_gt_root_query': sum(row['corresponding_query_index'] == row['hungarian_root_query_index'] for row in selected)}
        for threshold, suffix in [(.25, '025'), (.5, '050')]:
            counts['native_hits' + suffix] = sum(row['ious']['native'] > threshold for row in selected)
            counts['gt_root_matched_hits' + suffix] = sum(row['ious']['hungarian_root'] > threshold for row in selected)
            counts['teacher_hits' + suffix] = sum(row['ious']['teacher'] > threshold for row in selected)
            counts['teacher_passes_native_fails' + suffix] = sum(row['ious']['teacher'] > threshold >= row['ious']['native'] for row in selected)
            counts['teacher_passes_matched_fails' + suffix] = sum(row['ious']['teacher'] > threshold >= row['ious']['hungarian_root'] for row in selected)
            counts['native_fails_but_matched_passes' + suffix] = sum(row['ious']['hungarian_root'] > threshold >= row['ious']['native'] for row in selected)
        groups[name] = counts
    result = {'schema': 'mcln-teacher-box-target-alignment-v1', 'status': 'complete',
        'input_rows_sha256': receipt['rows_sha256'], 'groups': groups,
        'new_gpu_forwards': 0, 'new_optimizer_steps': 0, 'formal_rows': 0,
        'interpretation': 'Fit-row supervision alignment only; Query equality is not an instance-identity label; teacher boxes remain training targets, not deployed GT choices.',
        'training_configuration_changed': False}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
