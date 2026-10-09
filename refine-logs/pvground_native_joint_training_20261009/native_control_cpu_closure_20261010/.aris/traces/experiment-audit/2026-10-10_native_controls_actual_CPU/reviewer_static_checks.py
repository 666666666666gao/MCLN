"""Read-only artifact verification. Standard library only; no reviewed code is run."""
import ast
import base64
import datetime
import hashlib
import json
from pathlib import Path

TRACE = Path(__file__).resolve().parent
ROOT = TRACE.parents[3]
CONTROLS = ROOT / 'native_direct_controls_20261010'
CPU = CONTROLS / 'cpu_transport_attempt2'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_bytes())

initial = read_json(TRACE / 'input_hashes.initial.json')
assert len(initial) == 30
input_hashes = {}
for entry in initial:
    path = Path(entry['path'])
    actual = digest(path)
    assert actual == entry['sha256'], str(path)
    assert path.stat().st_size == entry['bytes'], str(path)
    input_hashes[str(path)] = 'sha256:' + actual

bundle = read_json(CONTROLS / 'CPU_BUNDLE.json')
prep = read_json(CONTROLS / 'DIRECT_CONTROL_PREPARATION.json')
old_audit = read_json(CONTROLS / 'actual_source_review/EXPERIMENT_AUDIT.json')
old_seal = read_json(CONTROLS / 'actual_source_review/SEAL.json')
old_input_hashes = old_audit['audited_input_hashes']
assert old_seal['audited_input_hashes'] == old_input_hashes
assert len(old_input_hashes) == old_audit['audited_input_count'] == old_seal['audited_input_count'] == 49
rehash_prior_inputs = {}
for name, expected in old_input_hashes.items():
    actual = 'sha256:' + digest(name)
    assert actual == expected, name
    rehash_prior_inputs[name] = actual
old_report_hashes = {}
for name in ('EXPERIMENT_AUDIT.json', 'EXPERIMENT_AUDIT.md'):
    path = CONTROLS / 'actual_source_review' / name
    actual = 'sha256:' + digest(path)
    assert old_seal['artifact_hashes'][str(path)] == actual
    old_report_hashes[str(path)] = actual

snapshot = Path(initial[29]['path'])
original_local_hashes = {}
for name, expected in bundle['original_sha256'].items():
    path = snapshot if name == 'whole_mask_range.py' else ROOT / 'source' / name
    actual = digest(path)
    assert actual == expected, name
    original_local_hashes[name] = actual
prepared_local_hashes = {}
for name, expected in bundle['prepared_sha256'].items():
    actual = digest(CONTROLS / name)
    assert actual == expected, name
    prepared_local_hashes[name] = actual
assert len(original_local_hashes) == 15
assert len(prepared_local_hashes) == 27
assert prep['original_source_sha256'] == {
    k: v for k, v in bundle['original_sha256'].items() if k != 'whole_mask_range.py'
}
assert prep['prepared_sha256'] == {
    k: v for k, v in bundle['prepared_sha256'].items() if k != 'check_direct_control_modules_cpu.py'
}
assert bundle['original_source'] == '/root/autodl-tmp/pvground_native_joint_training_20261010/PV-Ground'

raw = read_json(CPU / 'CPU_REMOTE_RAW_STDOUT.json')
stdout_bytes = (CPU / 'CPU_MODULE_STDOUT.json').read_bytes()
stderr_bytes = (CPU / 'CPU_MODULE_STDERR.txt').read_bytes()
assert base64.b64decode(raw['stdout_base64'], validate=True) == stdout_bytes
assert base64.b64decode(raw['stderr_base64'], validate=True) == stderr_bytes
assert stderr_bytes == (CPU / 'CPU_REMOTE_RAW_STDERR.txt').read_bytes() == b''
stdout = json.loads(stdout_bytes)
witness = read_json(CPU / 'CPU_MODULE_WITNESS.json')
assert stdout == witness
assert (json.dumps(stdout, indent=2) + '\n').encode() == (CPU / 'CPU_MODULE_WITNESS.json').read_bytes()
execution = read_json(CPU / 'CPU_MODULE_EXECUTION.json')
expected_execution = {k: v for k, v in raw.items() if not k.endswith('_base64')}
expected_execution.update(
    source_audit_sha256=digest(CONTROLS / 'actual_source_review/EXPERIMENT_AUDIT.json'),
    original_source_unchanged=True, current_training_queried=False,
    same_source_full_PV_GPU_preflight_pending=True, control_training_launched=False
)
assert execution == expected_execution
assert read_json(CPU / 'CPU_TRANSPORT_EXIT.json')['exit_code'] == raw['exit_code'] == execution['exit_code'] == 0
assert execution['argv'] == [
    '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python',
    '-B', '-u',
    '/root/autodl-tmp/pvground_native_direct_controls_cpu_20261010/check_direct_control_modules_cpu.py',
    '/root/autodl-tmp/pvground_native_direct_controls_cpu_20261010/CPU_BUNDLE.json',
]
assert stdout['original_source_sha256'] == bundle['original_sha256']
assert stdout['prepared_sha256'] == bundle['prepared_sha256']
assert stdout['status'] == 'ACTUAL_CPU_MODULE_CHECKS_COMPLETE_NO_FORMAL_MODEL_OR_ACCURACY'
assert stdout['fixture'] == dict(seed=2027, batch=1, candidates=256, superpoints=8, points=50000)
assert stdout['tensor_devices'] == ['cpu', 'cpu'] and stdout['cuda_visible_devices'] == execution['cuda_visible_devices'] == ''
assert stdout['A_state_tensors'] == 10 and stdout['B_state_tensors'] == 14
assert stdout['formal_accuracy'] is None
assert stdout['full_PV_constructor_or_1295_state_checked'] is False
assert stdout['native_criterion_checked'] is False
assert stdout['optimizer_scheduler_or_full_cold_recovery_checked'] is False
assert stdout['full_goal_complete'] is False

# Read the bound initializer manifest as data, not a model/checkpoint.
init_path = CONTROLS / 'common_behavior_check/init.json'
init = read_json(init_path)
assert stdout['support_payload_sha256'] == init['support_checkpoint_sha256']
assert stdout['span_payload_sha256'] == init['span_checkpoint_sha256']
elapsed = (datetime.datetime.fromisoformat(execution['finished_cst']) -
           datetime.datetime.fromisoformat(execution['started_cst'])).total_seconds()
assert elapsed > 0

proof_path = ROOT / 'read_only_static_diagnostic_20261010_attempt1/REMOTE_STATIC_WITNESS.json'
proof = read_json(proof_path)
static_prep = read_json(ROOT / 'STATIC_CPU_ATTEMPT2_PREPARATION.json')
assert digest(proof_path) == static_prep['proof_sha256']
assert proof['time_cst'] == static_prep['proof_time_cst']
assert proof['CPU_root_exists'] is False and proof['CPU_receipt_files'] == []
attempt_script = ROOT / 'run_direct_control_modules_cpu_attempt2_authorized.py'
assert digest(attempt_script) == static_prep['scripts'][str(attempt_script)]
assert read_json(CONTROLS / 'CPU_TRANSPORT_EXIT.json')['exit_code'] == 255
assert read_json(CONTROLS / 'CPU_STATIC_ROOT_READ_EXIT.json')['exit_code'] == 255
assert (CONTROLS / 'CPU_REMOTE_RAW_STDERR.txt').read_bytes() == (CONTROLS / 'CPU_STATIC_ROOT_READ_STDERR.txt').read_bytes()
assert read_json(ROOT / 'read_only_static_diagnostic_20261010_attempt1/TRANSPORT.json')['exit_code'] == 0

parsed = {}
for entry in initial:
    path = Path(entry['path'])
    if path.suffix == '.py':
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        parsed[str(path)] = dict(sha256=digest(path), top_level_nodes=len(tree.body))
transport_tree = ast.parse(attempt_script.read_text(encoding='utf-8'))
remote_code = next(
    n.value.value for n in transport_tree.body
    if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == 'code' for t in n.targets)
)
ast.parse(remote_code, filename='transport_embedded_remote_code')
assert len(parsed) == 10

# Only source data and metadata are read; no project/module imports or workloads.
report = dict(
    status='DETERMINISTIC_LOCAL_HASH_JSON_RECEIPT_AND_AST_CHECKS_PASS',
    generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    listed_input_count=30,
    audited_input_hashes=input_hashes,
    inputs_unchanged_since_initial_hash=True,
    prior_audit_input_hashes_rechecked_count=49,
    prior_audit_input_hashes=rehash_prior_inputs,
    prior_seal_report_hashes_verified=old_report_hashes,
    original_bundle_local_hash_count=15,
    prepared_bundle_local_hash_count=27,
    original_bundle_local_hashes=original_local_hashes,
    prepared_bundle_local_hashes=prepared_local_hashes,
    remote_stdout_base64_byte_equal_local=True,
    remote_stderr_base64_byte_equal_local=True,
    both_attempt2_stderr_files_empty=True,
    stdout_and_witness_json_equal=True,
    witness_is_local_pretty_reencoding_of_stdout=True,
    execution_metadata_exactly_matches_remote_receipt_plus_static_additions=True,
    attempt2_transport_and_module_exit_codes=[0, 0],
    attempt1_transport_and_static_read_exit_codes=[255, 255],
    old_failures_preserved_in_distinct_paths=True,
    source_audit_digest_matches_execution=True,
    bundle_maps_exactly_match_witness=True,
    parent_payload_hashes_match_bound_manifest=True,
    parent_payload_bytes_read_by_reviewer=False,
    supplemental_semantic_input=dict(path=str(init_path), sha256=digest(init_path)),
    static_proof_and_attempt2_script_hash_match_preparation=True,
    module_execution_elapsed_seconds=elapsed,
    observed_torch_version=stdout['torch_version'],
    observed_fixture=stdout['fixture'],
    observed_content_vs_full_mask_max_abs_difference=stdout['content_vs_full_mask_max_abs_difference'],
    neural_values_recomputed_by_reviewer=False,
    ast_files=parsed,
    ast_file_count=10,
    embedded_remote_program_ast_parsed=True,
    neural_imports_or_execution=False,
    gpu_calls=0,
    ssh_calls=0,
    current_training_status_reads=0,
    reviewed_input_mutations=0,
    reported_zero_counters_are_static_literals_not_independent_telemetry=True,
    original_remote_sources_post_execution_rehashed=False,
    old_source_audit_not_reclassified_as_gpu_admission=True,
)
(TRACE / 'DETERMINISTIC_CHECKS.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps({
    'status': report['status'], 'listed_inputs': 30,
    'prior_audit_input_hashes_rechecked': 49, 'original_bundle_hashes': 15,
    'prepared_bundle_hashes': 27, 'AST_files': 10,
    'closed_CPU_elapsed_seconds': elapsed, 'neural_calls': 0, 'SSH_calls': 0
}))
