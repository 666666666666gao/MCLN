import hashlib, json, shutil
from pathlib import Path
base = Path('/root/autodl-tmp/mcln_g0_view_pair_20260905/inputs_v3/fixed_source')
root = Path('/root/autodl-tmp/mcln_scanrefer_local_visual_preflight_20260906_v1')
manifest_raw = (base / 'g0_source_manifest.json').read_bytes()
assert hashlib.sha256(manifest_raw).hexdigest() == 'dcf333b0e1868a7eeaafaf7f0a7abdb664a34dda65966defc1ad244ce762b15d'
previous = json.loads(manifest_raw)
source = root / 'model_source'
source.mkdir()
for name, digest in previous['files'].items():
    raw = (base / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest, name
    path = source / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
for name in ['models/candidate_local_visual.py', 'models/encoder_decoder_layers.py', 'models/mcln.py']:
    (source / name).write_bytes((root / 'patches' / Path(name).name).read_bytes())
files = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
         for path in source.rglob('*') if path.is_file()}
manifest = {'parent_source_manifest_sha256': hashlib.sha256(manifest_raw).hexdigest(),
            'role': 'ScanRefer candidate local visual isolated source', 'files': files}
raw = (json.dumps(manifest, indent=2, sort_keys=True) + '
').encode()
(source / 'local_visual_source_manifest.json').write_bytes(raw)
for path in source.rglob('*'):
    if path.is_file():
        path.chmod(0o444)
print(json.dumps({'source_manifest_sha256': hashlib.sha256(raw).hexdigest(), 'files': len(files),
                  'source_bytes': sum(path.stat().st_size for path in source.rglob('*') if path.is_file())}))
