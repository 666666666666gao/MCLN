"""Summarize a completed fixed Nr3D pair; never select intermediate weights."""

import argparse
import hashlib
import json
from pathlib import Path


def analyze(run):
    files = {name: (run / name).read_bytes() for name in
             ('receipt.json', 'baseline_rows.json', 'terminal_rows.json')}
    receipt = json.loads(files['receipt.json'])
    baseline = json.loads(files['baseline_rows.json'])
    terminal = json.loads(files['terminal_rows.json'])
    assert receipt['status'] == 'complete'
    assert receipt['optimizer_steps_per_arm'] == 8230
    assert receipt['training_rows_per_arm'] == 32919
    assert receipt['formal_rows'] == len(baseline) == len(terminal) == 7899
    assert receipt['terminal_weights_reloaded_before_evaluation']
    for rows in (baseline, terminal):
        assert [row['row_id'] for row in rows] == list(range(7899))
    assert all((a['scan_id'], a['point_sha256']) ==
               (b['scan_id'], b['point_sha256']) for a, b in zip(baseline, terminal))
    stages = {'parent': [row['native'] for row in baseline],
              'native': [row['native'] for row in terminal],
              'replacement': [row['replacement'] for row in terminal]}
    report = {'rows': 7899, 'input_sha256': {
        name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()},
        'metrics': {}, 'paired': {}, 'goal': {}}
    for threshold, suffix, target in ((.25, '025', 4726), (.5, '050', 4059)):
        hits = {name: [row['iou'] > threshold for row in rows]
                for name, rows in stages.items()}
        for name, rows in stages.items():
            count = sum(hits[name])
            source = receipt['baseline']['native'] if name == 'parent' else receipt['terminal'][name]
            assert count == source['hits' + suffix]
            profiles = [row['candidate_profile'] for row in rows]
            legal = [p['after_filter']['top_256']['hit' + suffix] for p in profiles]
            unfiltered = [p['before_filter']['top_256']['hit' + suffix] for p in profiles]
            assert all(not hit or valid for hit, valid in zip(hits[name], legal))
            no_candidate = sum(not valid for valid in legal)
            not_selected = sum(valid and not hit for valid, hit in zip(legal, hits[name]))
            assert no_candidate + not_selected + count == 7899
            report['metrics'].setdefault(name, {})[suffix] = {
                'hits': count, 'accuracy_percent': count * 100 / 7899,
                'legal_oracle_hits': sum(legal), 'unfiltered_oracle_hits': sum(unfiltered),
                'no_legal_qualified_candidate': no_candidate,
                'legal_qualified_but_not_selected': not_selected,
                'failed_with_legal_top16_qualified': sum(
                    not hit and p['after_filter']['top_16']['hit' + suffix]
                    for hit, p in zip(hits[name], profiles))}
        for before, after in (('parent', 'native'), ('parent', 'replacement'), ('native', 'replacement')):
            repairs = sum(b and not a for a, b in zip(hits[before], hits[after]))
            breaks = sum(a and not b for a, b in zip(hits[before], hits[after]))
            assert repairs - breaks == sum(hits[after]) - sum(hits[before])
            report['paired'].setdefault(before + '_to_' + after, {})[suffix] = {
                'repairs': repairs, 'breaks': breaks, 'net': repairs - breaks}
            if before == 'native':
                assert receipt['replacement_vs_native'][str(threshold)] == {
                    'repairs': repairs, 'breaks': breaks}
        report['goal'][suffix] = {'required_hits': target,
            'replacement_remaining_hits': max(0, target - sum(hits['replacement']))}
    report['goal']['both_rec_targets_met'] = all(
        report['goal'][key]['replacement_remaining_hits'] == 0 for key in ('025', '050'))
    report['interpretation'] = ('Candidate decomposition is descriptive, not causal. '
        'Only replacement versus same-budget native isolates the label intervention.')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result, allow_nan=False))
