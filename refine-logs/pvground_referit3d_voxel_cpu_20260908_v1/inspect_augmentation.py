import ast
import hashlib
import json
import os
from pathlib import Path

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
archive = repo / 'refine-logs/pvground_referit3d_voxel_cpu_20260908_v1'
selection = json.loads((repo / 'refine-logs/referit3d_appearance_getitem_20260908_v1/manifest.json').read_bytes())
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
with sftp.open(selection['model_source'] + '/appearance_source_manifest.json', 'rb') as stream:
    manifest_raw = stream.read()
assert hashlib.sha256(manifest_raw).hexdigest() == selection['source_manifest_sha256']
with sftp.open(selection['model_source'] + '/src/joint_det_dataset.py', 'rb') as stream:
    native_raw = stream.read()
assert hashlib.sha256(native_raw).hexdigest() == json.loads(manifest_raw)['files']['src/joint_det_dataset.py']
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1'
with sftp.open(runtime + '/source_bundle_receipt.json', 'rb') as stream:
    bundle = json.loads(stream.read())
with sftp.open(runtime + '/PV-Ground/src/joint_det_dataset.py', 'rb') as stream:
    upstream_raw = stream.read()
assert hashlib.sha256(upstream_raw).hexdigest() == bundle['sources']['PV-Ground']['files']['src/joint_det_dataset.py']['sha256']
info = {}
asts = {}
for name, raw in [('native', native_raw), ('upstream', upstream_raw)]:
    source = raw.decode()
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Joint3DDataset')
    selected = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in ['_augment', '_get_pc']}
    info[name] = dict(full_source_sha256=hashlib.sha256(raw).hexdigest(),
        functions={key: ast.get_source_segment(source, node) for key, node in selected.items()})
    asts[name] = {key: ast.dump(node) for key, node in selected.items()}
info['augment_ast_equal'] = asts['native']['_augment'] == asts['upstream']['_augment']
rows = json.loads((archive / 'rows.json').read_bytes())
info['outside_by_axis'] = [dict(dataset=dset, augmented=augmented,
    xyz=[sum(row['out_of_range_by_axis'][axis] for row in rows
             if row['dataset']==dset and row['augmented']==augmented) for axis in range(3)])
    for dset in ['nr3d','sr3d'] for augmented in [False,True]]
(archive / 'augmentation_source.json').write_bytes((json.dumps(info, indent=2) + '\n').encode())
print(json.dumps({key: value for key, value in info.items() if key not in ['native','upstream']}, indent=2))
print('NATIVE_AUGMENT ' + info['native']['functions']['_augment'])
sftp.close()
client.close()
