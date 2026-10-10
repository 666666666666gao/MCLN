"""Reuse the completed static-publication route for the bounded face CPU note."""
import ast
from pathlib import Path

root = Path(__file__).resolve().parent
source = (root/'publish_author_core_cpu_authorized.py').read_text(encoding='utf-8')
for before,after in (
    ('referit_author_core_cpu','face_residual_cpu'),
    ('REFERIT_AUTHOR_CORE_CPU','FACE_RESIDUAL_CPU'),
    ('ACTUAL_AUTHOR_CORE_CPU','ACTUAL_FACE_RESIDUAL_CPU'),
    ('referit_author_core_20261010','face_residual_cpu_20261010'),
    ('20.376.133','20.376.134'),
    ('20.376.132','20.376.133'),
    ('referit_native_cpu_publication.json','referit_author_core_cpu_publication.json'),
    ("controls = root / 'face_residual_cpu_20261010'","controls = root / 'face_residual_preparation_20261010'"),
    ("assert closure['audit_verdict'] == 'WARN'", "assert closure['audit_verdict'] in ('PASS', 'WARN')"),
    ("assert audit['verdict'] == 'WARN'", "assert audit['verdict'] in ('PASS', 'WARN')"),
    ("CPU_audit_verdict='WARN'", "CPU_audit_verdict=closure['audit_verdict']"),
    ("(controls / name).read_bytes()", "(controls / 'actual_CPU_review' / name).read_bytes()"),
    ('Strict-load corresponding author cores through native Nr3D/Sr3D factories',
     'Check prepared face residual CPU modules without modifying active training'),
):
    assert before in source
    source = source.replace(before,after)
begin = source.index('names = [\n')
end = source.index('names.extend(',begin)
source = source[:begin] + """names = [
    'HANDOFF_FACE_RESIDUAL_CPU_20261010.md', 'ACTUAL_FACE_RESIDUAL_CPU_AUDIT_CLOSURE.json',
    'record_actual_face_residual_cpu_closure.py', 'prepare_face_cpu_transport_fix.py',
    'read_face_cpu_static_root_authorized.py', 'prepare_face_residual_cpu_publication.py',
    'publish_face_residual_cpu_authorized.py']
""" + source[end:]
old = "names.extend(path.relative_to(root).as_posix() for path in controls.rglob('*') if path.is_file())"
new = "names.extend(path.relative_to(root).as_posix() for path in controls.rglob('*') if path.is_file() and path.name != 'RAW_STDERR.txt')"
assert old in source
source = source.replace(old,new)
ast.parse(source)
target = root/'publish_face_residual_cpu_authorized.py'
assert not target.exists()
target.write_text(source,encoding='utf-8')
print('Prepared static-only face CPU publisher; no publication or training query executed.')
