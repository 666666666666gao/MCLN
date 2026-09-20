"""Download audited visualization inputs; SSH credential comes from runtime environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import paramiko

parser = argparse.ArgumentParser()
parser.add_argument('--remote', required=True)
parser.add_argument('--root', type=Path, required=True)
args = parser.parse_args()
remote, root = args.remote, args.root
root.mkdir(exist_ok=True)
(root / 'data').mkdir(exist_ok=True)
out = root / 'paper_style_dense'
out.mkdir(exist_ok=True)
(out / 'meshes').mkdir(exist_ok=True)
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
with sftp.open(remote + '/cases.json', 'rb') as f:
    raw = f.read()
manifest = json.loads(raw)
(root / 'cases.json').write_bytes(raw)
files = []
for case in manifest['cases']:
    files.append((remote + '/data/' + case['npz_file'], root / 'data' / case['npz_file'], case['npz_sha256']))
for item in manifest['mesh_sources']:
    files.append((item['remote_path'], out / item['local_file'], item['sha256']))
for source, dest, expected in files:
    with sftp.open(source, 'rb') as f:
        f.prefetch(file_size=sftp.stat(source).st_size)
        raw = f.read()
    assert hashlib.sha256(raw).hexdigest() == expected, source
    dest.write_bytes(raw)
    print('SAVED ' + dest.name, flush=True)
(out / 'mesh_sources.json').write_text(json.dumps({'files': manifest['mesh_sources'], 'use': 'Original colored mesh for display only; no prediction changes.'}, indent=2), encoding='utf-8')
sftp.close()
client.close()
print('EG_VISUAL_INPUTS_VERIFIED')
