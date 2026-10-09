"""Prepare static evidence publication; preserve the closed R2 artifacts."""
import ast
from pathlib import Path

root = Path(__file__).resolve().parent
source = (root/'publish_referit_native_cpu_authorized.py').read_text(encoding='utf-8')
changes = {
    "'native_control_cpu_closure_publication.json'":"'referit_native_cpu_publication.json'",
    "prior['section'] == '20.376.131'":"prior['section'] == '20.376.132'",
    "'referit_native_cpu_publication.json').exists()":"'referit_author_core_cpu_publication.json').exists()",
    "'ACTUAL_REFERIT_CPU_AUDIT_CLOSURE.json'":"'ACTUAL_AUTHOR_CORE_CPU_AUDIT_CLOSURE.json'",
    "'referit_native_preparation_20261010'":"'referit_author_core_20261010'",
    "b'20.376.132' not in old":"b'20.376.133' not in old",
    "'HANDOFF_REFERIT_NATIVE_SOURCE_CPU_20261010.md'":"'HANDOFF_REFERIT_AUTHOR_CORE_CPU_20261010.md'",
    "referit_native_source_cpu_20261010/":"referit_author_core_cpu_20261010/",
    "referit_native_source_cpu_20261010')":"referit_author_core_cpu_20261010')",
    "'.tmp_referit_native_cpu'":"'.tmp_referit_author_core_cpu'",
    "Prepare native Nr3D/Sr3D targets and close bounded CPU engineering checks":"Strict-load corresponding author cores through native Nr3D/Sr3D factories",
    "'referit_native_cpu_local_commit.json'":"'referit_author_core_cpu_local_commit.json'",
    "'referit_native_cpu_publication.json').write_text":"'referit_author_core_cpu_publication.json').write_text",
    "section='20.376.132'":"section='20.376.133'",
    "REFERIT_NATIVE_CPU_REMOTE":"REFERIT_AUTHOR_CORE_CPU_REMOTE",
    "REFERIT_NATIVE_CPU_ALL_LOCAL_AND_REMOTE_COPIES_COMMITTED_GITHUB_PENDING":"REFERIT_AUTHOR_CORE_CPU_ALL_LOCAL_AND_REMOTE_COPIES_COMMITTED_GITHUB_PENDING",
    "REFERIT_NATIVE_CPU_ALL_COPIES_AND_GITHUB_SYNCHRONIZED":"REFERIT_AUTHOR_CORE_CPU_ALL_COPIES_AND_GITHUB_SYNCHRONIZED",
    "target = repo / name":"target = Path('\\\\\\\\?\\\\' + str(repo / name))",
    "(repo / name).read_bytes()":"Path('\\\\\\\\?\\\\' + str(repo / name)).read_bytes()",
    "['git', '-C', str(repo), 'add'":"['git', '-C', str(repo), '-c', 'core.longpaths=true', 'add'",
    "['git', '-C', str(repo), 'status'":"['git', '-C', str(repo), '-c', 'core.longpaths=true', 'status'",
}
for before,after in changes.items():
    assert before in source,before
    source = source.replace(before,after)
start = source.index('names = [')
end = source.index('names.extend(',start)
source = source[:start] + """names = [
    'HANDOFF_REFERIT_AUTHOR_CORE_CPU_20261010.md', 'ACTUAL_AUTHOR_CORE_CPU_AUDIT_CLOSURE.json',
    'record_actual_author_core_cpu_audit_closure.py', 'prepare_referit_author_core.py',
    'referit_author_core_20261010_check_cpu.py', 'run_referit_author_core_cpu_authorized.py',
    'inspect_author_core_inventory_local.py', 'AUTHOR_CORE_INVENTORY_LOCAL_COMPARISON.json',
    'prepare_author_core_cpu_publication.py', 'publish_author_core_cpu_authorized.py']
""" + source[end:]
before = """for name, metadata in seal['files'].items():
    raw = Path(name).read_bytes()
    assert len(raw) == metadata['bytes'] and hashlib.sha256(raw).hexdigest() == metadata['sha256']
"""
after = """for name, digest in seal['files'].items():
    assert hashlib.sha256((controls / name).read_bytes()).hexdigest() == digest
"""
assert before in source
source = source.replace(before,after)
ast.parse(source)
target = root/'publish_author_core_cpu_authorized.py'
assert not target.exists()
target.write_text(source,encoding='utf-8')
print('Prepared static author-core publication script; no remote call.')
