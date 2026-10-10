"""Prepare a single publication recovery after verified no remote write."""
import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parent
state = json.loads((root/'FACE_CPU_PUBLICATION_STATIC_READ.json').read_bytes())
assert state['old_document'] is True and state['prefix_exists'] is False
assert state['verified_files'] == 0 and not state['mismatched']
assert json.loads((root/'FACE_RESIDUAL_CPU_REMOTE_EXIT.json').read_bytes())['exit_code'] == 255
assert not (root/'FACE_RESIDUAL_CPU_REMOTE_RECEIPT.json').exists()
source = (root/'publish_face_residual_cpu_authorized.py').read_text(encoding='utf-8')
source = source.replace('FACE_RESIDUAL_CPU_REMOTE_','FACE_RESIDUAL_CPU_REMOTE_ATTEMPT2_')
before = "prior = json.loads((root / 'referit_author_core_cpu_publication.json').read_bytes())"
after = before + "\nrecovery = json.loads((root / 'FACE_CPU_PUBLICATION_STATIC_READ.json').read_bytes())\nassert recovery['old_document'] is True and recovery['prefix_exists'] is False and recovery['verified_files'] == 0\nassert not recovery['mismatched']"
assert source.count(before) == 1
source = source.replace(before,after)
before = "    'publish_face_residual_cpu_authorized.py']"
after = "    'publish_face_residual_cpu_authorized.py', 'publish_face_residual_cpu_authorized_attempt2.py',\n    'prepare_face_cpu_publication_attempt2.py', 'read_face_cpu_publication_static_authorized.py',\n    'FACE_CPU_PUBLICATION_STATIC_READ.json', 'FACE_CPU_PUBLICATION_EXPECTED_AT_FAILURE.json',\n    'FACE_RESIDUAL_CPU_REMOTE_EXIT.json']"
assert source.count(before) == 1
source = source.replace(before,after)
ast.parse(source)
target = root/'publish_face_residual_cpu_authorized_attempt2.py'
assert not target.exists()
target.write_text(source,encoding='utf-8')
print('Prepared one static-publication recovery from actual old-doc/absent-prefix evidence; no training query.')
