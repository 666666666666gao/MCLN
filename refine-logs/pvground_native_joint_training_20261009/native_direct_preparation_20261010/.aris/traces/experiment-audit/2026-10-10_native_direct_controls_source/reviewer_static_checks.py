"""Review-only parsing, byte comparisons and hashes; never imports project code."""
import ast
import datetime
import hashlib
import json
from pathlib import Path

trace = Path(__file__).resolve().parent
root = trace.parents[3]
controls = root / 'native_direct_controls_20261010'
out = controls / 'actual_source_review'
request = (trace / 'request.txt').read_text(encoding='utf-8')
inputs = [Path(line[2:]) for line in request.splitlines() if line.startswith('- D:')]
supplemental = [controls / 'CPU_BUNDLE.json', root / 'prepare_direct_control_cpu_bundle.py',
                root / 'run_direct_control_modules_cpu_authorized.py',
                Path('C:/Users/gb/.codex/tmp/pvground_extremal_span_evidence_20261009/runner_v1/source_review/snapshots/034_whole_mask_range.py')]
inputs += supplemental
assert len(inputs) == len(set(inputs))
records = {}
trees = {}
for path in inputs:
    raw = path.read_bytes()
    content = raw.decode('utf-8')
    records[str(path)] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                             lines=len(content.splitlines()))
    if path.suffix == '.py':
        trees[str(path)] = ast.parse(content, filename=str(path))
    if path.suffix == '.json':
        json.loads(content)

prep = json.loads((controls / 'DIRECT_CONTROL_PREPARATION.json').read_bytes())
for name, expected in prep['original_source_sha256'].items():
    assert records[str(root / 'source' / name)]['sha256'] == expected
for name, expected in prep['prepared_sha256'].items():
    assert records[str(controls / name)]['sha256'] == expected

changed, identical = [], []
for name in prep['original_source_sha256']:
    original = (root / 'source' / name).read_bytes()
    prepared = (controls / 'source' / name).read_bytes()
    (identical if original == prepared else changed).append(name)
    ast.parse(prepared.decode('utf-8'), feature_version=(3, 7))
assert sorted(changed) == sorted(prep['changed_sources'])

arms = ['common_behavior_check', 'fixed_half_geometry', 'support_bypass', 'support_content_only']
specs = {arm: json.loads((controls / arm / 'init.json').read_bytes()) for arm in arms}
protocols = {arm: json.loads((controls / arm / 'NORMAL_NATIVE_RUN_PROTOCOL.json').read_bytes()) for arm in arms}
spec_diffs = {arm: [key for key in specs[arms[0]] if specs[arm][key] != specs[arms[0]][key]] for arm in arms}
protocol_diffs = {arm: [key for key in protocols[arms[0]] if protocols[arm][key] != protocols[arms[0]][key]] for arm in arms}
assert spec_diffs == prep['config_differences']
assert protocol_diffs == {arm: ([] if arm == arms[0] else ['native_init_spec']) for arm in arms}
assert len({(controls / arm / 'normal_joint_controller.py').read_bytes() for arm in arms}) == 1

bundle = json.loads((controls / 'CPU_BUNDLE.json').read_bytes())
assert bundle['original_source'] == '/root/autodl-tmp/pvground_native_joint_training_20261010/PV-Ground'
assert {key: value for key, value in bundle['original_sha256'].items() if key != 'whole_mask_range.py'} == prep['original_source_sha256']
assert records[str(supplemental[-1])]['sha256'] == bundle['original_sha256']['whole_mask_range.py']
for name, expected in bundle['prepared_sha256'].items():
    assert records[str(controls / name)]['sha256'] == expected
wrapper = trees[str(root / 'run_direct_control_modules_cpu_authorized.py')]
remote_code = next(node.value.value for node in wrapper.body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'code' for t in node.targets))
ast.parse(remote_code, feature_version=(3, 7))
call_names = sorted({node.func.id for tree in trees.values() for node in ast.walk(tree)
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)})

record = dict(status='DETERMINISTIC_SOURCE_HASH_AND_AST_CHECKS_PASS',
              generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              inputs=records, python_files_parsed=len(trees), listed_artifacts_read=len(inputs),
              changed_sources=changed, byte_identical_sources=identical,
              specification_differences=spec_diffs, protocol_differences=protocol_diffs,
              controllers_byte_identical=True, preparation_hashes_match_actual_files=True,
              cpu_bundle_prepared_hashes_match_actual_files=True,
              cpu_bundle_original_14_hashes_match_actual_files=True,
              whole_mask_range_source_review_pending=False,
              whole_mask_range_local_snapshot_hash_matches_cpu_bundle=True,
              calculate_diou_3d_called_in_audited_python='calculate_diou_3d' in call_names,
              verify_native_replacement_called_in_audited_python='verify_native_replacement' in call_names,
              cpu_module_witness_exists_locally=(controls / 'CPU_MODULE_WITNESS.json').exists(),
              cpu_module_execution_exists_locally=(controls / 'CPU_MODULE_EXECUTION.json').exists(),
              neural_calls=0, gpu_calls=0, ssh_calls=0, current_training_status_reads=0,
              scope='Local static parsing, byte equality and hashes only')
(out / 'DETERMINISTIC_CHECKS.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps({key: record[key] for key in ('status', 'listed_artifacts_read', 'python_files_parsed',
                 'changed_sources', 'controllers_byte_identical', 'cpu_module_witness_exists_locally',
                 'cpu_module_execution_exists_locally')}))
