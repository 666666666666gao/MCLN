"""Analyze fixed G exports; no model run, fitting, score changes, or new evaluation."""
import argparse
import datetime
import json
from pathlib import Path

from analyze_pvground_fixed_memory_selection import load, paired, sha
from compare_pvground_semantic_assignment_control import compare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_bytes())
    control, candidate = [Path(spec[k]) for k in ['native_root', 'candidate_root']]
    formal = Path(spec['formal_root'])
    assert (formal/'controller.exit').read_text().strip() == '0'
    receipt = json.loads((formal/'receipt.json').read_bytes())
    audit = json.loads((formal/'audit.json').read_bytes())
    assert audit['integrity_pass'] and audit['formal_rows'] == 9508
    assert audit['receipt_sha256'] == sha(formal/'receipt.json')
    assert receipt['terminal_checkpoint_sha256'] == sha(candidate/'terminal.pth')
    comparison = compare(spec, 'terminal')
    result = dict(status='complete', time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        script_sha256=sha(Path(__file__)), spec_sha256=sha(args.spec),
        analyzer_sha256=sha(Path(__file__).with_name('analyze_pvground_fixed_memory_selection.py')),
        comparison=comparison, stages={}, pairs={}, new_formal_rows=0, model_forwards=0, optimizer_steps=0,
        formal_receipt_sha256=sha(formal/'receipt.json'), formal_audit_sha256=sha(formal/'audit.json'),
        formal_protocol_sha256=sha(formal/'protocol.json'),
        scope='Existing fixed endpoint exports only. Module holdout and formal9508 remain separate. '
              'Formal parent comparison includes both added reader and training, not isolated G-label effect. '
              'Raw256 oracle is a GT diagnostic, not legal recall or deployable accuracy. '
              'No validation-derived thresholds, configuration or model changes.')
    data = {}; identities = {}
    for label, root, stage, count, split in [
        ('G_initial', candidate, 'initial', 6887, 'module'),
        ('D_terminal', control, 'terminal', 6887, 'module'),
        ('G_terminal', candidate, 'terminal', 6887, 'module'),
        ('formal_parent', formal, 'published_parent', 9508, 'formal'),
        ('formal_G', formal, 'fit_terminal', 9508, 'formal')]:
        rows, summary, arrays = load(root, stage, expected_rows=count)
        identity = [[r[k] for k in ['row_id', 'scan_id', 'target_id', 'point_sha256', 'root_box']] for r in rows]
        if split in identities: assert identities[split] == identity
        else: identities[split] = identity
        result['stages'][label] = summary
        for threshold in ['0.25', '0.5']:
            a = arrays[threshold]
            missed = a['available'] & ~a['hit']
            summary['thresholds'][threshold]['missed_good_rank_buckets'] = {
                '%d_%d' % (lo, hi): int((missed & (a['rank'] >= lo) & (a['rank'] <= hi)).sum())
                for lo, hi in [(1, 2), (3, 16), (17, 64), (65, 256)]}
            summary['thresholds'][threshold]['selected_iou_025_to_050'] = int(((a['picked'] > .25) & (a['picked'] <= .5)).sum())
        data[label] = arrays
    for old, new in [('G_initial', 'G_terminal'), ('D_terminal', 'G_terminal'), ('formal_parent', 'formal_G')]:
        result['pairs'][old+'->'+new] = paired(data[old], data[new])
    for threshold in ['0.25', '0.5']:
        assert result['pairs']['D_terminal->G_terminal'][threshold]['net'] == comparison['transitions']['bbs'][threshold]['net']
        assert result['pairs']['formal_parent->formal_G'][threshold]['net'] == receipt['transitions']['bbs'][threshold]['net']
    result['formal_primary'] = {
        arm: {key: receipt['metrics'][arm]['bbs'][key] for key in ['rec_hits25', 'rec_hits50']}
        for arm in ['published_parent', 'fit_terminal']}
    result['formal_primary_percent'] = {
        arm: {key: hits*100/9508 for key, hits in values.items()} for arm, values in result['formal_primary'].items()}
    result['protected_v99_gap_hits'] = {key: result['formal_primary']['fit_terminal'][key]-floor
        for key, floor in [('rec_hits25', 5572), ('rec_hits50', 4797)]}
    result['promotion'] = audit['advance_to_nr3d_sr3d_rec']
    with args.out.open('x') as f: json.dump(result, f, indent=2, allow_nan=False);f.write('\n')
    print(json.dumps({k:result[k] for k in ['status','formal_primary_percent','protected_v99_gap_hits','pairs','promotion']}),flush=True)


if __name__ == '__main__':
    main()
