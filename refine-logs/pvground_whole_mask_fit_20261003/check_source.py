"""Syntax and actual prior initial-row checks; no model, GPU or remote invocation."""
import ast
import hashlib
import json
from pathlib import Path
import datetime

from initial_range_comparison import compare_initial_rows

root = Path(__file__).parent
engineering = root.parent/'pvground_whole_mask_integration_20261003'
prior = root.parent/'pvground_fused_support_20261002'
files = sorted(root.glob('*.py'))
for path in files:
    ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 7))
modules = json.loads((root/'source_preparation.json').read_bytes())['model_modules_copied_without_changes']
assert all((root/name).read_bytes() == (engineering/name).read_bytes() for name in modules)
actual = [json.loads(line) for line in (prior/'complete_tail_raw/arm/initial/rows.jsonl').read_text().splitlines()]
reference = [json.loads(line) for line in (prior/'complete_tail_fused_retry/arm/initial/rows.jsonl').read_text().splitlines()]
comparison = compare_initial_rows(actual, reference)
assert comparison['input_identities_exact'] and comparison['zero_head_within_forward_exact']
assert not comparison['cross_process_output_exact']
assert comparison['rec_threshold_decisions_exact']
assert comparison['differences']['bbs']['mask_rows']
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), status='PASS_LOCAL_SOURCE_ONLY',
    python37_syntax_files=len(files), actual_historical_initial_rows_per_group=6887,
    actual_float_discrepancies_preserved=True, unchanged_model_modules=modules,
    model_forwards=0, GPU_invocations=0, optimizer_updates=0, new_weights=0,
    files=[dict(path=str(path), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in files])
(root/'local_source_checks.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps(record))
