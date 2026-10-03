"""Record local preparation without observing or changing the running experiment."""
import datetime
import hashlib
import json
from pathlib import Path


root = Path(__file__).parent
pair = root.parent / 'pvground_candidate_consistency_20261003'
receipt_path = root / 'postrun_preparation.json'
assert not receipt_path.exists()
assert not (root / 'complete').exists() and not (root / 'analysis').exists()
sources = {}
for name in ('collect_complete.py', 'analyze_complete.py'):
    path = root / name
    raw = path.read_bytes()
    compile(raw, str(path), 'exec')
    sources[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
helper = pair / 'analyze_complete_initial_qualified.py'
assert hashlib.sha256(helper.read_bytes()).hexdigest() == 'a5d01e0cf5babeda9f440f8a3ab815ffb27b6a9606a01f89c9b6cd402589ce07'
old = json.loads((pair / 'complete/g_consistent/spec.json').read_bytes())
new = json.loads((root / 'normalized_spec.json').read_bytes())
allowed = {'root', 'comparison', 'consistency_module_sha256', 'contrastive_normalization', 'completed_comparators'}
assert {key: value for key, value in old.items() if key not in allowed} == {
    key: value for key, value in new.items() if key not in allowed}
assert new['semantic_consistency'] and not new['p2']
assert new['contrastive_normalization'] == 'expanded_correspondence_count'
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
    status='PREPARED_NOT_EXECUTED', sources=sources, scripts_compile=True,
    intended_spec_differences_only=True, metrics_helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
    collector_requires_original_controller_exit_zero=True,
    collector_copies_checkpoints=False, collector_deletes_weights=False,
    analyzer_compares_completed_control_and_unscaled_expansion=True,
    analyzer_records_starting_threshold_differences_without_assuming_equality=True,
    comparator_directory=str(pair / 'complete'),
    initial_terminal_rows=6887, formal_rows=9508, primary_mode='bbs',
    training_source_changed=False, gpu_forwards=0, optimizer_updates=0,
    new_accuracy_available=False, remote_observations=0,
    next_remote_check='2026-10-03T19:32:02.226257+08:00')
receipt_path.write_bytes((json.dumps(record, indent=2) + '\n').encode())
print(json.dumps(record))
