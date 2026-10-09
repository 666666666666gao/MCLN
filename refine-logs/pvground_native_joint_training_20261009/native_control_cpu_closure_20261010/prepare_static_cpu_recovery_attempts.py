"""Prepare explicit second attempts only after an authoritative static read."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
witness_path = root / 'read_only_static_diagnostic_20261010_attempt1/REMOTE_STATIC_WITNESS.json'
witness = json.loads(witness_path.read_bytes())
assert witness['document_matches_prior_confirmed'] is True
assert witness['document_matches_current_local'] is False
assert witness['CPU_root_exists'] is False and not witness['CPU_receipt_files']
assert witness['copied_evidence_sha256'] == {}
assert witness['normal_training_reads'] == 0
controls = root / 'native_direct_controls_20261010'
cpu_attempt = controls / 'cpu_transport_attempt2'
assert not cpu_attempt.exists()
cpu_attempt.mkdir()
cpu = (root / 'run_direct_control_modules_cpu_authorized.py').read_text(encoding='utf-8')
line = "controls = root / 'native_direct_controls_20261010'\n"
assert cpu.count(line) == 1
cpu = cpu.replace(line, line + "attempt_dir = controls / 'cpu_transport_attempt2'\n"
    "proof = json.loads((root / 'read_only_static_diagnostic_20261010_attempt1/REMOTE_STATIC_WITNESS.json').read_bytes())\n"
    "assert proof['CPU_root_exists'] is False and not proof['CPU_receipt_files']\n")
for name in ('CPU_REMOTE_RAW_STDOUT.json', 'CPU_REMOTE_RAW_STDERR.txt', 'CPU_TRANSPORT_EXIT.json',
             'CPU_MODULE_STDOUT.json', 'CPU_MODULE_STDERR.txt', 'CPU_MODULE_EXECUTION.json',
             'CPU_MODULE_WITNESS.json'):
    old = "(controls / '" + name + "')"
    assert old in cpu, name
    cpu = cpu.replace(old, "(attempt_dir / '" + name + "')")
ast.parse(cpu)
cpu_path = root / 'run_direct_control_modules_cpu_attempt2_authorized.py'
assert not cpu_path.exists()
cpu_path.write_text(cpu, encoding='utf-8')

sync = (root / 'sync_native_direct_control_publication_authorized.py').read_text(encoding='utf-8')
assert "assert not (root / 'NATIVE_DIRECT_CONTROL_REMOTE_SYNC_RECEIPT.json').exists()" in sync
for suffix in ('RECEIPT.json', 'STDOUT.json', 'STDERR.txt', 'EXIT.json'):
    old = 'NATIVE_DIRECT_CONTROL_REMOTE_SYNC_' + suffix
    assert old in sync, old
    sync = sync.replace(old, 'NATIVE_DIRECT_CONTROL_REMOTE_SYNC_ATTEMPT2_' + suffix)
anchor = "assert publication['remote_handoff_sync_complete'] is False\n"
assert sync.count(anchor) == 1
sync = sync.replace(anchor, anchor +
    "proof = json.loads((root / 'read_only_static_diagnostic_20261010_attempt1/REMOTE_STATIC_WITNESS.json').read_bytes())\n"
    "assert proof['document_matches_prior_confirmed'] is True and proof['copied_evidence_sha256'] == {}\n")
ast.parse(sync)
sync_path = root / 'sync_native_direct_control_publication_attempt2_authorized.py'
assert not sync_path.exists()
sync_path.write_text(sync, encoding='utf-8')
record = dict(status='EXPLICIT_SECOND_CPU_AND_STATIC_COPY_ATTEMPTS_PREPARED_FROM_ACTUAL_READ_ONLY_STATE',
    proof_path=str(witness_path), proof_sha256=hashlib.sha256(witness_path.read_bytes()).hexdigest(),
    proof_time_cst=witness['time_cst'], original_failed_receipts_untouched=True,
    CPU_checker_and_bundle_unchanged=True, source_audit_inputs_unchanged=True,
    changed_transport_scope='Separate second-attempt output artifacts and prerequisite witness only',
    scripts={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (cpu_path, sync_path)},
    normal_training_queried=False, normal_training_restarted=False,
    second_attempts_executed=False, full_goal_complete=False)
(root / 'STATIC_CPU_ATTEMPT2_PREPARATION.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status=record['status'], prior_receipts_preserved=True, normal_training_queried=False)))
