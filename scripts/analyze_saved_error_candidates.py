"""Partition saved evaluation errors without discarding any candidate."""
import datetime
import hashlib
import json
from pathlib import Path

root = Path(r'C:\Users\gb\.codex\tmp\pvground_fused_support_20261002')
paths = {
    'same_tail_raw': root / 'complete_tail_raw/arm/formal',
    'G_continued': Path(r'C:\Users\gb\.codex\tmp\pvground_g_p2_20261002\complete\g_control\formal'),
}
result = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
              GPU_forward=False, optimizer_updates=0, candidates_retained=256, systems={})
for system, directory in paths.items():
    raw = (directory / 'rows.jsonl').read_bytes()
    receipt = json.loads((directory / 'receipt.json').read_bytes())
    assert hashlib.sha256(raw).hexdigest() == receipt['rows_sha256']
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == 9508 and len({r['row_id'] for r in rows}) == 9508
    modes = {}
    for mode in ('bbs', 'bbf'):
        values = [r[mode] for r in rows]
        counts = {}
        for threshold, suffix in ((.25, '25'), (.5, '50')):
            key = 'oracle' + suffix
            for v in values:
                assert len(v[key]) == 4 and v[key] == sorted(v[key])
                assert not (v['iou'] > threshold) or v[key][0] == 1
            errors = [v for v in values if v['iou'] <= threshold]
            bins = {name: 0 for name in ('rank_2_to_16', 'rank_17_to_32', 'rank_33_to_64', 'rank_65_to_256', 'no_qualified_box')}
            for v in errors:
                first = next((i for i, present in enumerate(v[key]) if present), 4)
                bins[list(bins)[first]] += 1
            assert sum(bins.values()) == len(errors)
            hits = len(values) - len(errors)
            assert hits == receipt['metrics'][mode]['rec_hits' + suffix]
            recoverable = len(errors) - bins['no_qualified_box']
            counts['rec' + suffix] = dict(hits=hits, errors=len(errors),
                error_rows_with_another_qualified_box=recoverable,
                fraction_of_errors_with_another_qualified_box=recoverable / len(errors),
                first_qualified_rank_interval=bins,
                total_qualified_Full256=sum(v[key][-1] for v in values))
        loose_only = [v for v in values if .25 < v['iou'] <= .5]
        counts['selected_loose_only'] = dict(rows=len(loose_only),
            has_alternative_strict_box=sum(v['oracle50'][-1] for v in loose_only),
            no_alternative_strict_box=sum(1-v['oracle50'][-1] for v in loose_only))
        modes[mode] = counts
    result['systems'][system] = dict(rows=9508, path=str(directory), modes=modes)
result['evidence_limits'] = (
    'Saved GPU top-K coverage flags only; exact first rank, all candidate coordinates, native matching and physical instance identity are not saved. '
    'A box passing root IoU is an offline geometric qualification, not proof it satisfies every word. '
    'Selected loose-only boxes may have range errors, but this alone does not prove correct physical identity. '
    'Unmatched/low-ranked candidates are retained; no new pruning, labels, loss or active-training changes.')
destination = root / 'SAVED_ERROR_CANDIDATES.json'
assert not destination.exists()
destination.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(output=str(destination), primary={k:v['modes']['bbs'] for k,v in result['systems'].items()})))
