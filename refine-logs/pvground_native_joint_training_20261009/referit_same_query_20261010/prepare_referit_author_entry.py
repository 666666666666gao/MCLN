"""Prepare the existing native ReferIt entry for the verified author data protocol."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
previous = root / 'referit_author_core_20261010'
destination = root / 'referit_author_entry_20261010'
assert not destination.exists()
source = previous / 'source'
old = (source / 'train_dist_mod.py').read_text(encoding='utf-8')
before = '    assert opt.butd and not opt.butd_cls and not opt.butd_gt and not opt.joint_det'
after = ('    assert opt.butd_cls and opt.joint_det and opt.detect_intermediate\n'
         '    assert not opt.butd and not opt.butd_gt and not opt.augment_det')
assert old.count(before) == 1
new = old.replace(before, after)
ast.parse(new, feature_version=(3, 7))
copied = []
for path in source.rglob('*.py'):
    relative = path.relative_to(source)
    target = destination / 'source' / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(path.read_bytes())
    copied.append(relative.as_posix())
(destination / 'source/train_dist_mod.py').write_text(new, encoding='utf-8')
for dataset in ('nr3d', 'sr3d'):
    target = destination / dataset / 'init.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((previous / dataset / 'init.json').read_bytes())
    initial = json.loads(target.read_bytes())
    assert initial['dataset'] == dataset and initial['seed'] == 2027
    assert initial['new_module_initialization'] == 'fresh'
    assert all(initial[name] is None for name in ('g_checkpoint', 'support_checkpoint', 'span_checkpoint'))
diff = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
    fromfile='prior/source/train_dist_mod.py', tofile='prepared/source/train_dist_mod.py'))
(destination / 'ENTRY_CHANGE.diff').write_text(diff, encoding='utf-8')
data_result = root / 'referit_data_interface_cpu_20261010/cpu_execution/REFERIT_DATA_CPU_RESULT.json'
result = json.loads(data_result.read_bytes())
assert result['status'] == 'ACTUAL_NATIVE_REFERIT_LIMITED_DATA_CPU_PASS'
port = json.loads((root / 'referit_data_interface_cpu_20261010/WARM_SOURCE_HASHES.json').read_bytes())
changed = []
for name in copied:
    original = source / name
    prepared = destination / 'source' / name
    if original.read_bytes() != prepared.read_bytes():
        changed.append(name)
    port[name] = hashlib.sha256(prepared.read_bytes()).hexdigest()
assert changed == ['train_dist_mod.py']
(destination / 'WARM_SOURCE_HASHES.json').write_text(json.dumps(port, indent=2) + '\n')
receipt = dict(status='AUTHOR_DATA_PROTOCOL_NATIVE_ENTRY_SOURCE_PREPARED',
    previous_source=str(source), changed_files=changed, source_files=len(copied),
    unchanged_other_source_files=len(copied)-1, init_specs_unchanged=True,
    author_flags={dataset: result['cases'][dataset]['author_flags'] for dataset in ('nr3d', 'sr3d')},
    data_check_result=str(data_result), warm_source_files=len(port),
    full_training_recipe_matches_author=False, final_method_selection_pending=True,
    source_review_pending=True, source_deployed=False, actual_entry_executed=False,
    actual_GPU_preflight_completed=False, Nr3D_or_Sr3D_training_launched=False,
    active_ScanRefer_source_changed=False, current_training_queries=0,
    neural_calls=0, saved_weights=0, formal_accuracy=None, full_goal_complete=False)
(destination / 'SOURCE_PREPARATION.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps({key: receipt[key] for key in ('status', 'changed_files', 'source_files',
    'source_deployed', 'actual_entry_executed', 'full_goal_complete')}))
