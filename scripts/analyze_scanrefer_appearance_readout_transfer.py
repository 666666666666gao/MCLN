"""Compare saved native and full-system endpoint decisions without new inference."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def changes(before, after, threshold):
    repair = sum(b <= threshold < a for b, a in zip(before, after))
    damage = sum(a <= threshold < b for b, a in zip(before, after))
    return dict(repair=repair, damage=damage, net=repair-damage)


def join_rows(native, full):
    """Join by recorded row and point identity, never by dynamic Query number."""
    assert set(native) == set(full) == {'initial', 'control', 'appearance'}
    reference = native['initial']
    ids = [r['row_id'] for r in reference]
    assert len(set(ids)) == len(ids)
    for arm in native:
        assert len(native[arm]) == len(full[arm]) == len(ids)
        for rows in [native[arm], full[arm]]:
            for expected, actual in zip(reference, rows):
                for key in ['row_id', 'scan_id', 'physical_space', 'point_sha256']:
                    assert expected[key] == actual[key], (arm, key, expected['row_id'])
    joined = []
    for i, row in enumerate(reference):
        value = {k: row[k] for k in ['row_id', 'scan_id', 'physical_space', 'point_sha256']}
        for arm in native:
            value[arm+'_native_iou'] = native[arm][i]['rec_iou']
            value[arm+'_system_iou'] = full[arm][i]['rec_iou']
            value[arm+'_system_mask_iou'] = full[arm][i]['mask_iou']
        joined.append(value)
    return joined


def summarize(rows):
    result = {}
    for threshold in [.25, .5]:
        arms = {}
        for arm in ['initial', 'control', 'appearance']:
            native = [r[arm+'_native_iou'] for r in rows]
            system = [r[arm+'_system_iou'] for r in rows]
            arms[arm] = dict(native_hits=sum(x > threshold for x in native),
                             system_hits=sum(x > threshold for x in system),
                             system_relative_to_native=changes(native, system, threshold))
        pairs = {}
        for reference in ['initial', 'control']:
            native_effect = changes([r[reference+'_native_iou'] for r in rows],
                                    [r['appearance_native_iou'] for r in rows], threshold)
            system_effect = changes([r[reference+'_system_iou'] for r in rows],
                                    [r['appearance_system_iou'] for r in rows], threshold)
            cross = Counter(''.join(str(int(r[k] > threshold)) for k in [
                reference+'_native_iou', reference+'_system_iou',
                'appearance_native_iou', 'appearance_system_iou']) for r in rows)
            lift_change = (arms['appearance']['system_relative_to_native']['net']
                           - arms[reference]['system_relative_to_native']['net'])
            assert system_effect['net'] == native_effect['net'] + lift_change
            damaged = [r for r in rows if r['appearance_system_iou'] <= threshold
                       < r[reference+'_system_iou']]
            pairs[reference] = dict(native_effect=native_effect, system_effect=system_effect,
                system_lift_change=lift_change,
                cross_table_bit_order=[reference+'_native', reference+'_system',
                                       'appearance_native', 'appearance_system'],
                cross_table=dict(sorted(cross.items())),
                system_damage_count=len(damaged),
                system_damage_appearance_native_pass=sum(r['appearance_native_iou'] > threshold for r in damaged),
                system_damage_still_pass025=sum(r['appearance_system_iou'] > .25 for r in damaged))
        result[str(threshold)] = dict(arms=arms, comparisons=pairs)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    receipt = json.loads((root/'receipt.json').read_text())
    audit = json.loads((root/'audit.json').read_text())
    native_receipt = json.loads((root/'native_evaluation/receipt.json').read_text())
    assert receipt['status'] == native_receipt['status'] == 'complete'
    assert audit['integrity_pass'] and native_receipt['state_and_files_unchanged']
    assert receipt['formal_rows'] == native_receipt['formal_rows'] == 0
    assert receipt['previous_pretraining_has_seen_development_holdout']
    assert native_receipt['previous_pretraining_has_seen_development_holdout']
    assert native_receipt['training_receipt_sha256'] == sha(root/'receipt.json')
    assert native_receipt['audit_sha256'] == sha(root/'audit.json')
    assert native_receipt['training_manifest_sha256'] == receipt['manifest_sha256']
    names = ['baseline_rows.json', 'terminal_rows.json', 'native_evaluation/rows.json']
    expected = [receipt['baseline_rows_sha256'], receipt['terminal_rows_sha256'],
                native_receipt['rows_sha256']]
    for name, digest in zip(names, expected):
        assert sha(root/name) == digest, name
    baseline, terminal, native = [json.loads((root/name).read_text()) for name in names]
    assert baseline['appearance'] == baseline['control']
    full = dict(initial=baseline['control'], control=terminal['control'], appearance=terminal['appearance'])
    rows = join_rows(native, full)
    assert len(rows) == receipt['holdout_rows'] == native_receipt['rows_per_arm'] == 6887
    summary = summarize(rows)
    for threshold, suffix in [(.25, '025'), (.5, '050')]:
        for arm in native:
            metrics = summary[str(threshold)]['arms'][arm]
            assert metrics['native_hits'] == native_receipt['metrics'][arm]['hits'+suffix]
            stage_metrics = (receipt['baseline_metrics']['control'] if arm == 'initial'
                             else receipt['terminal_metrics'][arm])
            assert metrics['system_hits'] == stage_metrics['rec_hits'+suffix]
    out = args.output.resolve()
    out.mkdir()
    with (out/'paired_rows.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(status='complete', rows=len(rows), formal_rows=0,
        previous_backbone_saw_module_holdout=True, input_sha256=dict(zip(names, expected)),
        native_receipt_sha256=sha(root/'native_evaluation/receipt.json'),
        paired_rows_sha256=sha(out/'paired_rows.csv'), thresholds=summary,
        scope='Same input-point identities; arithmetic decomposition of native/full-system effects. '
              'No attribution to a particular Parent, Geometry or V99 stage. '
              'No candidate-instance identity claim; no new training or selection rule.')
    with (out/'receipt.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
