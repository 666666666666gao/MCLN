"""Prepare the next evidence publication from the already-used static publisher."""
from pathlib import Path
import ast

root = Path(__file__).resolve().parent
source = (root / 'publish_native_control_cpu_closure_authorized.py').read_text(encoding='utf-8')
replacements = {
    "assert prior['remote_current_sha256'] == prior['doc_sha256']": "prior_remote = json.loads(Path(prior['remote_sync_receipt']).read_bytes())\nassert prior_remote['remote_sync_complete'] is True and prior_remote['doc_sha256'] == prior['doc_sha256']",
    "'native_direct_controls_publication.json'": "'native_control_cpu_closure_publication.json'",
    "prior['section'] == '20.376.130'": "prior['section'] == '20.376.131'",
    "native_control_cpu_closure_publication.json').exists()": "referit_native_cpu_publication.json').exists()",
    "'ACTUAL_CONTROL_CPU_AUDIT_CLOSURE.json'": "'ACTUAL_REFERIT_CPU_AUDIT_CLOSURE.json'",
    "'native_direct_controls_20261010'": "'referit_native_preparation_20261010'",
    "'actual_CPU_review/SEAL.json'": "'actual_CPU_review/OUTPUT_SHA256.json'",
    "b'20.376.131' not in old": "b'20.376.132' not in old",
    "'HANDOFF_NATIVE_CONTROL_CPU_CLOSURE_20261010.md'": "'HANDOFF_REFERIT_NATIVE_SOURCE_CPU_20261010.md'",
    "native_control_cpu_closure_20261010/": "referit_native_source_cpu_20261010/",
    "native_control_cpu_closure_20261010')": "referit_native_source_cpu_20261010')",
    "'.tmp_native_cpu_closure'": "'.tmp_referit_native_cpu'",
    "Close bounded native-control CPU engineering review and restore static evidence sync": "Prepare native Nr3D/Sr3D targets and close bounded CPU engineering checks",
    "'native_control_cpu_closure_local_commit.json'": "'referit_native_cpu_local_commit.json'",
    "'native_control_cpu_closure_publication.json').write_text": "'referit_native_cpu_publication.json').write_text",
    "section='20.376.131'": "section='20.376.132'",
    "NATIVE_CONTROL_CPU_CLOSURE_REMOTE": "REFERIT_NATIVE_CPU_REMOTE",
    "NATIVE_CPU_CLOSURE_ALL_LOCAL_AND_REMOTE_COPIES_COMMITTED_GITHUB_PENDING": "REFERIT_NATIVE_CPU_ALL_LOCAL_AND_REMOTE_COPIES_COMMITTED_GITHUB_PENDING",
    "NATIVE_CPU_CLOSURE_ALL_COPIES_AND_GITHUB_SYNCHRONIZED": "REFERIT_NATIVE_CPU_ALL_COPIES_AND_GITHUB_SYNCHRONIZED",
}
for before, after in replacements.items():
    assert before in source, before
    source = source.replace(before, after)
start = source.index("for inputs in (audit['audited_input_hashes']")
end = source.index('repos = [', start)
source = source[:start] + """for name, metadata in seal['files'].items():
    raw = Path(name).read_bytes()
    assert len(raw) == metadata['bytes'] and hashlib.sha256(raw).hexdigest() == metadata['sha256']
assert audit['verdict'] == 'WARN' and audit['blocking_issue_count'] == 0
assert hashlib.sha256((controls / 'actual_CPU_review/EXPERIMENT_AUDIT.json').read_bytes()).hexdigest() == closure['audit_sha256']
assert hashlib.sha256((controls / 'actual_CPU_review/OUTPUT_SHA256.json').read_bytes()).hexdigest() == closure['seal_sha256']
""" + source[end:]
start = source.index('names = [')
end = source.index('assert not any(', start)
source = source[:start] + """names = [
    'HANDOFF_REFERIT_NATIVE_SOURCE_CPU_20261010.md', 'ACTUAL_REFERIT_CPU_AUDIT_CLOSURE.json',
    'record_actual_referit_cpu_audit_closure.py', 'prepare_referit_native_sources.py',
    'correct_referit_native_metric_interfaces.py', 'record_referit_cpu_target_intake.py',
    'prepare_referit_cpu_publication.py', 'publish_referit_native_cpu_authorized.py']
names.extend(path.relative_to(root).as_posix() for path in controls.rglob('*') if path.is_file())
""" + source[end:]
source = source.replace('control_training_launched=False, active_training_source_changed=False,',
    'control_training_launched=False, Nr3D_or_Sr3D_training_launched=False, active_training_source_changed=False,')
ast.parse(source)
target = root / 'publish_referit_native_cpu_authorized.py'
assert not target.exists()
target.write_text(source, encoding='utf-8')
print('Prepared static publication script; no SSH or training call.')
