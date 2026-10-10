"""Bind an existing author-core runtime and author data flags for a limited CPU check."""
import ast
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
parent=root.parent
author=parent/'referit_author_core_20261010'
receipt=json.loads((author/'cpu_execution/AUTHOR_CORE_CPU_RESULT.json').read_bytes())
assert receipt['status']=='ACTUAL_AUTHOR_CORE_NATIVE_FACTORY_CPU_LOAD_PASS'
port=json.loads((parent/'NATIVE_SOURCE_PORT.json').read_bytes())
expected={name:value['sha256'] for name,value in port['files'].items()}
for path in (author/'source').rglob('*.py'):
    expected[path.relative_to(author/'source').as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
assert expected['src/joint_det_dataset.py']=='3afebfc69232f9547e6810563ba6dd4f34a0f770f929df2f4203b21493de0e3d'
spec=dict(warm_source='/root/autodl-tmp/pvground_referit_author_core_cpu_20261010/PV-Ground',
    data_root='/root/autodl-tmp/DATA_ROOT_mcln_meshsp/',env_spec_sha256=port['env_spec_sha256'],
    author_core_execution_sha256=hashlib.sha256((author/'cpu_execution/AUTHOR_CORE_CPU_RESULT.json').read_bytes()).hexdigest(),
    author_flags={dataset:receipt['cases'][dataset]['author_training_flags'] for dataset in ('nr3d','sr3d')},
    annotation_limit_per_source=128,real_data_execution_pending=True,GPU_training_admission=False)
for name,value in [('CHECK_SPEC.json',spec),('WARM_SOURCE_HASHES.json',expected)]:
    (root/name).write_text(json.dumps(value,indent=2)+'\n')
for path in root.glob('*.py'):
    tree=ast.parse(path.read_text(encoding='utf-8'),feature_version=(3,7))
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='remote_code' for target in node.targets):
            ast.parse(ast.literal_eval(node.value),feature_version=(3,7))
print(json.dumps(dict(status='REFERIT_LIMITED_CPU_DATA_SOURCE_PREPARED',warm_files=len(expected),GPU_training_admission=False)))
