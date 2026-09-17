"""Independent endpoint recount and exact signed label-replacement accounting."""
import argparse
import importlib.util
import json
import math
from pathlib import Path

spec = importlib.util.spec_from_file_location('base_audit', Path(__file__).parent / 'base_audit.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
sha = base.sha
audit_stage = base.audit_stage


def audit(root):
    result = base.audit(root)
    spec = json.loads((root / 'spec.json').read_bytes())
    receipt = json.loads((root / 'receipt.json').read_bytes())
    assert spec['semantic_assignment'] and spec['assignment_threshold'] == .5
    assert receipt['semantic_assignment']
    assert sha(root / 'pvground_semantic_assignment.py') == spec['assignment_module_sha256'] == receipt['assignment_module_sha256']
    rows = [json.loads(line) for line in (root / 'train.jsonl').read_text().splitlines()]
    totals = dict(reassigned_queries=0, reassigned_samples=0)
    for row in rows:
        for key in ['loss_native', 'loss_assignment_correction', 'replaced_ce_old', 'replaced_ce_new']:
            assert math.isfinite(row[key])
        assert math.isclose(row['loss_assignment_correction'],
            (row['replaced_ce_new'] - row['replaced_ce_old']) * (.5/7), rel_tol=1e-5, abs_tol=1e-6)
        assert math.isclose(row['loss'], row['loss_native'] + row['loss_assignment_correction'], rel_tol=1e-6, abs_tol=1e-6)
        batch = len(row['rows'])
        assert batch <= row['matched_queries'] <= batch * 256
        assert 0 <= row['reassigned_queries'] <= batch * 256 - row['matched_queries']
        assert 0 <= row['reassigned_samples'] <= min(batch, row['reassigned_queries'])
        if row['reassigned_queries'] == 0:
            assert row['replaced_ce_old'] == row['replaced_ce_new'] == row['loss_assignment_correction'] == 0
        for key in totals:
            totals[key] += row[key]
    assert totals['reassigned_queries'] > 0
    result.update(semantic_assignment_verified=True, assignment_counts=totals,
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
    print('SEMANTIC_ASSIGNMENT_ENDPOINT_AUDIT_COMPLETE ' + json.dumps(result), flush=True)
