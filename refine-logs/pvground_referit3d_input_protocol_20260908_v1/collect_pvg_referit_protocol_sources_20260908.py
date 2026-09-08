import datetime
import hashlib
import json
import os
from pathlib import Path

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
archive = repo / 'refine-logs/pvground_referit3d_input_protocol_20260908_v1'
archive.mkdir()
bundle = json.loads((repo / 'refine-logs/pvground_runtime_20260908_v1/source_bundle_receipt.json').read_bytes())
manifest = json.loads((repo / 'refine-logs/scanrefer_object_appearance_pair_20260908_v1/input_manifest.json').read_bytes())
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/PV-Ground'
source = manifest['model_source']
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
with sftp.open(source + '/appearance_source_manifest.json', 'rb') as stream:
    native_manifest_raw = stream.read()
assert hashlib.sha256(native_manifest_raw).hexdigest() == manifest['source_manifest_sha256']
native_manifest = json.loads(native_manifest_raw)
(archive / 'native_source_manifest.json').write_bytes(native_manifest_raw)
splits = ['data/meta_data/%s_%s_scans.txt' % (dataset, split) for dataset in ['nr3d', 'sr3d'] for split in ['train', 'test']]
recipes = ['scripts/%s_%s.sh' % (phase, dataset) for dataset in ['nr3d', 'sr3d'] for phase in ['train', 'test']]
sources = ['src/joint_det_dataset.py', 'src/grounding_evaluator.py', 'main_utils.py', 'train_dist_mod.py']
files = {}
for label, base, names in [('upstream', runtime, recipes + splits + sources), ('native', source, splits + ['src/joint_det_dataset.py'])]:
    for name in names:
        with sftp.open(base + '/' + name, 'rb') as stream:
            raw = stream.read()
        digest = hashlib.sha256(raw).hexdigest()
        expected = bundle['sources']['PV-Ground']['files'][name]['sha256'] if label == 'upstream' else native_manifest['files'][name]
        assert digest == expected, (label, name)
        target = archive / label / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        files[label + '/' + name] = {'source_path': base + '/' + name, 'bytes': len(raw), 'sha256': digest}
receipt = {'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
           'status': 'pass', 'pv_commit': bundle['sources']['PV-Ground']['commit'],
           'native_source_manifest_sha256': hashlib.sha256(native_manifest_raw).hexdigest(),
           'files': files, 'gpu_forwards': 0, 'optimizer_steps': 0, 'weight_downloads': 0}
(archive / 'source_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
sftp.close()
client.close()
print(json.dumps({'time_cst': receipt['time_cst'], 'status': 'pass', 'files': len(files), 'archive': str(archive)}))
