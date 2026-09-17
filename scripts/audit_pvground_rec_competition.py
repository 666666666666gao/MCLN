"""Recount endpoint and additionally verify every recorded competition term."""
import argparse
import importlib.util
import json
import math
from pathlib import Path

module_spec = importlib.util.spec_from_file_location('base_audit', Path(__file__).parent / 'base_audit.py')
base = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(base)
sha = base.sha
audit_stage = base.audit_stage


def audit(root):
    result = base.audit(root)
    spec = json.loads((root / 'spec.json').read_bytes())
    receipt = json.loads((root / 'receipt.json').read_bytes())
    assert spec['rec_competition'] and spec['competition_weight'] == 1.0
    assert receipt['rec_competition']
    assert sha(root / 'pvground_rec_competition.py') == spec['competition_module_sha256'] == receipt['competition_module_sha256']
    rows = [json.loads(line) for line in (root / 'train.jsonl').read_text().splitlines()]
    counts = {key: 0 for key in ['eligible25', 'eligible50', 'active25', 'active50']}
    for row in rows:
        assert math.isfinite(row['loss_native']) and math.isfinite(row['loss_rec_competition'])
        assert row['loss_rec_competition'] >= 0
        assert math.isclose(row['loss'], row['loss_native'] + row['loss_rec_competition'], rel_tol=1e-6, abs_tol=1e-6)
        for suffix in ['25', '50']:
            assert 0 <= row['active' + suffix] <= row['eligible' + suffix] <= len(row['rows'])
        for key in counts:
            counts[key] += row[key]
    assert counts['active25'] + counts['active50'] > 0
    result.update(rec_competition_verified=True, competition_pair_counts=counts,
                  auditor_sha256=sha(Path(__file__)), base_auditor_sha256=sha(Path(__file__).parent / 'base_audit.py'))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('REC_COMPETITION_ENDPOINT_AUDIT_COMPLETE ' + json.dumps(result), flush=True)
