"""Build an isolated source overlay from the unchanged sealed PV source."""
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).parent
parent = Path('/root/autodl-tmp/pvground_tail_support_source_20261002_v1')
assert hashlib.sha256((parent / 'source_port.json').read_bytes()).hexdigest() == '3b2b44241f0c1b52b5e42f6ea84e656abb1dc3f3b062c87d49f0bcbf94b2f7bc'
port = json.loads((parent / 'source_port.json').read_bytes())
for name, digest in port['files'].items():
    assert hashlib.sha256((parent / 'PV-Ground' / name).read_bytes()).hexdigest() == digest, name
shutil.copytree(parent / 'PV-Ground', root / 'PV-Ground')
overrides = {}
for name in ('models/pv_ground.py', 'models/modules.py'):
    raw = (root / 'overrides' / name).read_bytes()
    (root / 'PV-Ground' / name).write_bytes(raw)
    overrides[name] = hashlib.sha256(raw).hexdigest()
port['files'].update(overrides)
port.update(boundary_evidence_readback=True, native_semantic_head_deferred=True,
    original_source_port_sha256='3b2b44241f0c1b52b5e42f6ea84e656abb1dc3f3b062c87d49f0bcbf94b2f7bc')
raw = (json.dumps(port, indent=2, sort_keys=True) + '\n').encode()
(root / 'source_port.json').write_bytes(raw)
digest = hashlib.sha256(raw).hexdigest()
for arm in ('evidence_hidden', 'evidence_visible'):
    template = json.loads((root / (arm + '_preflight_template.json')).read_bytes())
    template['source_port_sha256'] = digest
    (root / arm / 'spec.json').write_text(json.dumps(template, indent=2, sort_keys=True) + '\n')
print(json.dumps(dict(source_port_sha256=digest, overrides=overrides, original_source_unchanged=True)))
