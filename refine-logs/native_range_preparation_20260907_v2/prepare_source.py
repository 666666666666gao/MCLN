import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
expected = json.loads((root / 'expected.json').read_text())
parent = Path(expected['parent_source'])
manifest = (parent / 'native_source_manifest.json').read_bytes()
assert hashlib.sha256(manifest).hexdigest() == expected['parent_manifest_sha256']
files = json.loads(manifest)['files']
assert len(files) == 616
source = root / 'model_source'
source.mkdir()
for name, digest in files.items():
    raw = (parent / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest, name
    target = source / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
for name, digest in expected['overlay_files'].items():
    raw = (root / 'overlays' / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest, name
    (source / name).write_bytes(raw)
    files[name] = digest
assert len(files) == 618
for name in files:
    (source / name).chmod(0o444)
result = dict(expected, files=files, model_source=str(source))
(source / 'native_source_manifest.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print('Prepared618sourcefiles; original native and live Scan sources unchanged', flush=True)
