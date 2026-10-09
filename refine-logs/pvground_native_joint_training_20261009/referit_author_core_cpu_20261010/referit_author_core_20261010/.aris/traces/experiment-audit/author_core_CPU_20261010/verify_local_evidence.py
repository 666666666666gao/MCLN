"""Bounded local evidence verification; stdlib only, never imports audited code."""
import ast
import base64
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(r'C:\Users\gb\.codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\referit_author_core_20261010')
PARENT = ROOT.parent
R2_ROOT = PARENT / 'referit_native_preparation_20261010'
TRACE = ROOT / '.aris/traces/experiment-audit/author_core_CPU_20261010'
HASHES = {}


def read(path):
    path = Path(path)
    assert path.name.lower() not in {'memory.md', 'soul.md', 'user.md', 'auth', 'auth.md'}
    assert 'askpass' not in path.name.lower() and 'memory' not in [part.lower() for part in path.parts]
    raw = path.read_bytes()
    HASHES[path.as_posix()] = hashlib.sha256(raw).hexdigest()
    return raw


def doc(path):
    return json.loads(read(path))


def sha(path):
    return hashlib.sha256(read(path)).hexdigest()


def write(name, value):
    (TRACE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def compare_hashes(mapping):
    rows = []
    for name, expected in mapping.items():
        actual = sha(name)
        rows.append(dict(path=name, expected_sha256=expected.removeprefix('sha256:'),
                         actual_sha256=actual, matches=actual == expected.removeprefix('sha256:')))
    return dict(count=len(rows), matches=sum(row['matches'] for row in rows),
                mismatches=[row for row in rows if not row['matches']], rows=rows)


TRACE.mkdir(parents=True, exist_ok=True)
request = read(ROOT / 'ACTUAL_CPU_AUDIT_REQUEST.txt')
r3_path = ROOT / 'source_review/EXPERIMENT_CODE_REVIEW_R3.json'
r3 = doc(r3_path)
r3_binding = sha(r3_path)
assert r3_binding == '87f68880f93a224aaf89fc4f5bfc67504b135c1d1187d4ae97ed85c3f0e4e287'
r3_hashes = compare_hashes(r3['audited_input_hashes'])
write('R3_input_continuity.json', r3_hashes)

cpu = ROOT / 'cpu_execution'
raw = doc(cpu / 'RAW_STDOUT.json')
receipt = doc(cpu / 'CPU_EXECUTION.json')
transport = doc(cpu / 'TRANSPORT_EXIT.json')
byte_checks = {}
for field, filename in [('stdout_base64', 'CPU_STDOUT.txt'), ('stderr_base64', 'CPU_STDERR.txt'),
                        ('result_base64', 'AUTHOR_CORE_CPU_RESULT.json')]:
    decoded = base64.b64decode(raw[field], validate=True)
    saved = read(cpu / filename)
    byte_checks[field] = dict(file=filename, decoded_bytes=len(decoded), saved_bytes=len(saved),
                             bytes_identical=decoded == saved, sha256=hashlib.sha256(saved).hexdigest())
raw_metadata = {key: value for key, value in raw.items() if not key.endswith('_base64')}
local_metadata = {key: value for key, value in receipt.items() if key != 'source_review_sha256'}
result = doc(cpu / 'AUTHOR_CORE_CPU_RESULT.json')
stdout_result = doc(cpu / 'CPU_STDOUT.txt')
port = doc(PARENT / 'NATIVE_SOURCE_PORT.json')
protocol = doc(PARENT / 'NORMAL_NATIVE_RUN_PROTOCOL.json')
receipt_checks = dict(base64_files=byte_checks, raw_metadata_equals_local=raw_metadata == local_metadata,
    source_review_sha256_bound=receipt['source_review_sha256'] == r3_binding,
    stdout_JSON_equals_independent_result_JSON=stdout_result == result,
    transport_exit_code=transport['exit_code'], child_exit_code=raw['exit_code'],
    local_receipt_exit_code=receipt['exit_code'], raw_transport_stderr_bytes=len(read(cpu / 'RAW_STDERR.txt')),
    env_spec_sha256_matches_port=receipt['env_spec_sha256'] == port['env_spec_sha256'],
    started_cst=receipt['started_cst'], finished_cst=receipt['finished_cst'],
    wall_seconds=(datetime.datetime.fromisoformat(receipt['finished_cst']) -
                  datetime.datetime.fromisoformat(receipt['started_cst'])).total_seconds(),
    source=receipt['source'], torch_version=result['torch_version'],
    CUDA_initialized=result['CUDA_initialized'], case_keys=list(result['cases']))
write('receipt_verification.json', receipt_checks)

case_checks = {}
for dataset, case in result['cases'].items():
    spec = doc(ROOT / dataset / 'init.json')
    flag_differences = {name: dict(author=value, CPU=case['CPU_factory_flags'][name])
        for name, value in case['author_training_flags'].items() if value != case['CPU_factory_flags'][name]}
    case_checks[dataset] = dict(checkpoint_sha256_matches_spec=case['author_checkpoint_sha256'] == spec['official_checkpoint_sha256'],
        official_checkpoint_sha256=spec['official_checkpoint_sha256'],
        author_core_tensors=case['author_core_tensors'], retained_core_tensors=case['exact_loaded_core_tensors'],
        full_model_state_tensors=case['full_model_state_tensors'], native_factory=case['native_factory'],
        core_key_shape_dtype_and_values_exact=case['core_key_shape_dtype_and_values_exact'],
        deterministic_position_buffer_verified=case['deterministic_position_buffer_verified'],
        A_and_B_output_zero_verified=case['A_and_B_output_zero_verified'],
        no_ScanRefer_state_recorded=case['model_architecture']['scanrefer_core_or_module_state_loaded'] is False,
        overlay_specs_null=all(spec[name] is None for name in ('g_checkpoint','support_checkpoint','span_checkpoint')),
        CPU_only=case['CPU_only'], CUDA_initialized=case['CUDA_initialized'],
        forwards=case['forwards'], optimizer_steps=case['optimizer_steps'], real_loader_rows=case['real_loader_rows'],
        seed=case['model_architecture']['seed'], flag_differences=flag_differences,
        elapsed_seconds=case['elapsed_seconds'])

prep = doc(ROOT / 'SOURCE_PREPARATION.json')
r2_sources = compare_hashes({(R2_ROOT/name).as_posix(): digest for name,digest in prep['R2_sources_and_specs_sha256'].items()})
r3_sources = compare_hashes({(ROOT/name).as_posix(): digest for name,digest in prep['current_sources_sha256'].items()})
changed = [name for name in prep['R2_sources_and_specs_sha256'] if read(ROOT/name) != read(R2_ROOT/name)]
r3_trace = ROOT / '.aris/traces/experiment-bridge/R3'
protected_reports = compare_hashes(doc(r3_trace / 'R1_R2_report_hashes_before.json'))
r2 = doc(R2_ROOT / 'source_review/EXPERIMENT_CODE_REVIEW_R2.json')
r2_hashes = compare_hashes(r2['audited_input_hashes'])
snapshot = R2_ROOT / 'source_review/R2_CLOSED_TRACKER_INPUT.md'
alias = doc(R2_ROOT / 'source_review/R2_SEALED_TRACKER_SNAPSHOT.json')
alias_rows = [dict(live_path=name, same_live_file=Path(name).samefile(R2_ROOT/'refine-logs/EXPERIMENT_TRACKER.md'),
    same_snapshot_file=Path(item['snapshot_path']).samefile(snapshot),
    archived_digest_matches=sha(snapshot) == item['sha256'], archived_sha256=sha(snapshot))
    for name,item in alias.items()]
continuity = dict(R3_declared_inputs=r3_hashes, R2_declared_inputs=r2_hashes,
    R2_sources_and_specs=r2_sources, R3_sources_and_specs=r3_sources, changed_since_R2=changed,
    protected_R1_R2_reports=protected_reports, tracker_alias_checks=alias_rows)
write('artifact_continuity.json', continuity)

comparison = doc(PARENT / 'AUTHOR_CORE_INVENTORY_LOCAL_COMPARISON.json')
inventories = {}
for name, metadata in comparison['inputs'].items():
    inventory = doc(name)
    assert sha(name) == metadata['sha256'] and len(read(name)) == metadata['bytes']
    dataset = 'scanrefer' if '_scan_' in name else ('nr3d' if '_nr_' in name else 'sr3d')
    assert all(key.startswith('module.') for key in inventory)
    inventories[dataset] = {key[7:]: value for key,value in inventory.items()}
recomputed = {}
for dataset in ('nr3d','sr3d'):
    scan, author = inventories['scanrefer'], inventories[dataset]
    only = sorted(set(author)-set(scan))
    recomputed[dataset] = dict(tensors=len(author), author_only=only, scan_only=sorted(set(scan)-set(author)),
        common_shape_dtype_mismatches=[name for name in sorted(set(scan)&set(author))
            if any(scan[name][field] != author[name][field] for field in ('shape','dtype'))],
        author_only_metadata={name:author[name] for name in only})
history = {}
for dataset, suffix in [('nr3d','pvground_nr_checkpoint_inspection_20260908_v2'),
                        ('sr3d','pvground_sr_checkpoint_inspection_20260908_v1')]:
    directory = Path(r'C:\Users\gb\.codex_mcln_g0_20260905\refine-logs') / suffix
    old_result = doc(directory/'strict_load.json')
    history[dataset] = dict(script_sha256_matches=sha(directory/'strict_load.py') == old_result['script_sha256'],
        checkpoint_sha256_matches_current_spec=old_result['checkpoint_sha256'] == case_checks[dataset]['official_checkpoint_sha256'],
        status=old_result['status'], state_tensors=old_result['state_tensors'],
        model_source=old_result['model_source'], historical_not_current_result=True)

critical = [PARENT/name for name in ('referit_author_core_20261010_check_cpu.py',
    'run_referit_author_core_cpu_authorized.py','prepare_referit_author_core.py')]
critical += sorted((ROOT/'source').rglob('*.py'))
source_records = {}
for path in critical:
    text = read(path).decode('utf-8-sig')
    tree = ast.parse(text, filename=str(path))
    source_records[path.as_posix()] = dict(sha256=sha(path), source=text,
        assert_lines=[node.lineno for node in ast.walk(tree) if isinstance(node, ast.Assert)])
runner_ast = ast.parse(read(PARENT/'run_referit_author_core_cpu_authorized.py').decode('utf-8'))
remote = next(node.value.value for node in runner_ast.body if isinstance(node,ast.Assign)
              and any(isinstance(target,ast.Name) and target.id=='code' for target in node.targets))
ast.parse(remote)
write('raw_source_evidence.json', source_records)
write('embedded_runner_source.json', dict(source=remote, AST_parse='PASS'))

for path in sorted((ROOT/'source_review').iterdir()):
    if path.is_file():
        read(path)
for path in sorted(r3_trace.iterdir()):
    if path.is_file():
        read(path)
read(ROOT/'PLAN.md')
raw_copy = TRACE/'raw_cpu_execution'
raw_copy.mkdir(exist_ok=True)
for path in sorted(cpu.iterdir()):
    if path.is_file():
        (raw_copy/path.name).write_bytes(read(path))

summary = dict(status='PASS', verified_at=datetime.datetime.now().astimezone().isoformat(),
    scope='Local byte/JSON/base64/AST verification only; audited code is never imported or executed',
    receipt=receipt_checks, cases=case_checks,
    R3_inputs=dict(count=r3_hashes['count'], matches=r3_hashes['matches'], mismatches=r3_hashes['mismatches']),
    R2_inputs=dict(count=r2_hashes['count'], matches=r2_hashes['matches'], mismatches=r2_hashes['mismatches']),
    R2_source_and_spec_count=r2_sources['count'], R2_source_and_spec_matches=r2_sources['matches'],
    protected_R1_R2_report_count=protected_reports['count'], protected_R1_R2_report_matches=protected_reports['matches'],
    changed_since_R2=changed, tracker_alias_checks=alias_rows,
    archived_inventory_recomputed=dict(scanrefer_count=len(inventories['scanrefer']), comparisons=recomputed,
        matches_archived_comparison=recomputed == comparison['comparisons']),
    historical_strict_load_attribution=history,
    AST_parsed_current_files=len(critical), AST_parsed_embedded_remote_blocks=1,
    native_source_port_file_count=len(port['files']),
    reviewer_execution=dict(SSH_calls=0, model_imports=0, model_constructors=0, forwards=0, GPU_calls=0,
        training_queries=0, input_edits=0, CPU_check_reruns=0))
assert r3_hashes['count'] == r3_hashes['matches'] == 114
assert all(item['bytes_identical'] for item in byte_checks.values())
assert raw_metadata == local_metadata and stdout_result == result
assert transport['exit_code'] == raw['exit_code'] == receipt['exit_code'] == 0
assert receipt['source_review_sha256'] == r3_binding and receipt_checks['env_spec_sha256_matches_port']
assert set(result['cases']) == {'nr3d','sr3d'} and result['CUDA_initialized'] is False
assert r2_sources['matches'] == r3_sources['matches'] == 16
assert changed == ['source/native_model_initialization.py']
assert protected_reports['matches'] == 8
assert r2_hashes['count'] == 76 and r2_hashes['matches'] == 75
assert len(r2_hashes['mismatches']) == 1 and r2_hashes['mismatches'][0]['path'].endswith('/refine-logs/EXPERIMENT_TRACKER.md')
assert sha(snapshot) == '03bca0f2ce19ac70a59c5a38449dcf322b24ee0d46895f51181b00dfccb7343a'
assert all(row['same_live_file'] and row['same_snapshot_file'] and row['archived_digest_matches'] for row in alias_rows)
assert recomputed == comparison['comparisons'] and len(inventories['scanrefer']) == 1234
assert all(row['script_sha256_matches'] and row['checkpoint_sha256_matches_current_spec'] for row in history.values())
for case in case_checks.values():
    assert case['checkpoint_sha256_matches_spec'] and case['author_core_tensors'] == 1235
    assert case['retained_core_tensors'] == 1234 and case['full_model_state_tensors'] == 1295
    assert case['CPU_only'] and case['CUDA_initialized'] is False
    assert case['forwards'] == case['optimizer_steps'] == case['real_loader_rows'] == 0
    assert case['core_key_shape_dtype_and_values_exact'] and case['deterministic_position_buffer_verified']
    assert case['A_and_B_output_zero_verified'] and case['no_ScanRefer_state_recorded'] and case['overlay_specs_null']
    assert set(case['flag_differences']) == {'butd','butd_cls','joint_det','detect_intermediate'}
write('deterministic_verification.json', summary)
write('audited_input_hashes.json', dict(sorted(HASHES.items())))
print(json.dumps({key: summary[key] for key in ('status','R3_inputs','R2_inputs','R2_source_and_spec_matches',
    'protected_R1_R2_report_matches','changed_since_R2')}, ensure_ascii=False))
print(json.dumps(dict(receipt=receipt_checks, input_hash_count=len(HASHES)), ensure_ascii=False))
