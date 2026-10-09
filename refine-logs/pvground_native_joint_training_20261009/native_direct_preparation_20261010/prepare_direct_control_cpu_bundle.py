import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
controls = root / 'native_direct_controls_20261010'
path = controls / 'CPU_BUNDLE.json'
assert not path.exists()
prepared = json.loads((controls / 'DIRECT_CONTROL_PREPARATION.json').read_bytes())
port = json.loads((root / 'NATIVE_SOURCE_PORT.json').read_bytes())
source_hashes = dict(prepared['original_source_sha256'])
source_hashes['whole_mask_range.py'] = port['files']['whole_mask_range.py']['sha256']
file_hashes = dict(prepared['prepared_sha256'])
file_hashes['check_direct_control_modules_cpu.py'] = hashlib.sha256(
    (controls / 'check_direct_control_modules_cpu.py').read_bytes()).hexdigest()
for name, digest in file_hashes.items():
    assert hashlib.sha256((controls / name).read_bytes()).hexdigest() == digest
bundle = dict(original_source=port['model_source'], original_sha256=source_hashes,
              prepared_sha256=file_hashes)
path.write_text(json.dumps(bundle, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status='CPU_BUNDLE_PREPARED_NOT_EXECUTED', path=str(path),
    sha256=hashlib.sha256(path.read_bytes()).hexdigest(), original_dependencies=len(source_hashes),
    prepared_files=len(file_hashes), neural_calls=0)))
