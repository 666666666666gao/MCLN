"""Prepare only the isolated native factory entry and existing face candidates."""
import ast
import json
from pathlib import Path

root=Path(__file__).resolve().parent
previous=root.parent
old=previous/'face_residual_preparation_20261010'
source=root/'source'
assert not source.exists()
source.mkdir()
content=(previous/'source/train_dist_mod.py').read_bytes()
before=b'from native_model_initialization import configure_native_model\r\n        return configure_native_model('
after=b'from native_face_model_initialization import configure_native_face_model\r\n        return configure_native_face_model('
assert content.count(before)==1
(source/'train_dist_mod.py').write_bytes(content.replace(before,after))
for name in ('face_residual_span_mixer.py','native_face_model_initialization.py'):
    (source/name).write_bytes((old/name).read_bytes())
for mode,name in [('source_conditioned','source_conditioned_init.json'),
                  ('without_additional_source','without_additional_source_init.json')]:
    spec=json.loads((old/name).read_bytes())
    assert spec['face_residual_mode']==mode
    (root/(mode+'.json')).write_text(json.dumps(spec,indent=2)+'\n')
for name in ('NATIVE_SOURCE_PORT.json','NORMAL_NATIVE_RUN_PROTOCOL.json'):
    (root/name).write_bytes((previous/name).read_bytes())
terminal=json.loads((previous/'NORMAL_TERMINAL_SUMMARY_20261010.json').read_bytes())
identity=next(row for row in terminal['full_cold_recovery']['checkpoints'] if row['name']=='best')['identity']
(root/'NORMAL_E0_IDENTITY.json').write_text(json.dumps(identity,indent=2)+'\n')
for path in list(source.glob('*.py'))+[root/'check_native_face_factory_cpu.py',Path(__file__)]:
    ast.parse(path.read_text(encoding='utf-8'),feature_version=(3,7))
print('ISOLATED_NATIVE_FACE_FACTORY_CPU_SOURCE_PREPARED_NOT_EXECUTED')
