"""Recount native candidate availability and selection in the completed D/F pair."""
import argparse
import datetime
import json
from pathlib import Path

from analyze_pvground_fixed_memory_selection import load, paired, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_bytes())
    comparison = json.loads(args.comparison.read_bytes())
    roots = [Path(spec[k]) for k in ['native_root', 'candidate_root']]
    assert comparison['status'] == 'complete' and comparison['stage'] == 'terminal'
    assert comparison['roots'] == [str(p) for p in roots]
    assert comparison['training_row_order_identical']
    assert [sha(p / 'spec.json') for p in roots] == spec['training_spec_sha256']
    assert [sha(p / 'terminal/receipt.json') for p in roots] == comparison['receipt_sha256']
    result = dict(status='complete', rows=6887, formal_rows=0, model_forwards=0, optimizer_steps=0,
        time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        comparison_sha256=sha(args.comparison), spec_sha256=sha(args.spec),
        script_sha256=sha(Path(__file__)),
        reused_analysis_sha256=sha(Path(__file__).with_name('analyze_pvground_fixed_memory_selection.py')),
        stages={}, pairs={},
        scope='Backbone-seen module holdout. All256 raw boxes, GT-only capacity. '
              'Rank ties reported as bounds. No cross-query identity, causal regression, '
              'new-scene generalization or deployed oracle claim. No scoring/training changes.')
    data = {}
    identity = None
    for label, root, stage in [('F_initial', roots[1], 'initial'),
                               ('D_terminal', roots[0], 'terminal'),
                               ('F_terminal', roots[1], 'terminal')]:
        assert (root / 'controller.exit').read_text().strip() == '0'
        rows, summary, arrays = load(root, stage)
        current = [[r[k] for k in ['row_id', 'scan_id', 'target_id', 'point_sha256', 'root_box']] for r in rows]
        if identity is None:
            identity = current
        else:
            assert current == identity
        result['stages'][label] = summary
        data[label] = arrays
    result['pairs']['F_initial_to_terminal'] = paired(data['F_initial'], data['F_terminal'])
    result['pairs']['D_to_F_terminal'] = paired(data['D_terminal'], data['F_terminal'])
    for threshold in ['0.25', '0.5']:
        assert result['pairs']['D_to_F_terminal'][threshold]['net'] == comparison['transitions']['bbs'][threshold]['net']
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
