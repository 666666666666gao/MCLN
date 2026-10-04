"""Prepare the future fit payload; no preflight/training approval is asserted."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

local = Path(__file__).resolve().parent
preflight = local.parent / 'revision2'
bundle = local / 'runtime_bundle'
bundle.mkdir()
for path in (preflight / 'runtime_bundle').glob('*.py'):
    if path.name != 'run_readback_preflight.py':
        shutil.copyfile(path, bundle / path.name)
shutil.copyfile(local / 'run_readback_fit.py', bundle / 'run_readback_fit.py')
files = {}
for path in sorted(bundle.glob('*.py')):
    raw = path.read_bytes()
    ast.parse(raw.decode('utf-8'), feature_version=(3, 7))
    files[path.name] = hashlib.sha256(raw).hexdigest()
assert len(files) == 16
root = '/root/autodl-tmp/pvground_readback_fit_20261004'
for arm in ('evidence_hidden', 'evidence_visible'):
    spec = json.loads((preflight / (arm + '_preflight_spec.json')).read_bytes())
    spec.update(root=root + '/' + arm, fit_passes=1, updates=3723, primary_mode='bbs', primary_threshold=.5,
                runner_files=files, preflight_root=spec['root'])
    (local / (arm + '_fit_spec.json')).write_text(json.dumps(spec, indent=2) + '\n', encoding='utf-8')
for path in local.glob('*.py'):
    ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 7))
(local / 'FORMAL_DRAFT_SOURCE_CHECK.json').write_text(json.dumps(dict(
    status='SOURCE_ONLY_AST37_PASS', runner_files=files, geometry_source_changed=False,
    source_port_sha256=spec['source_port_sha256'], formal_fit_approved=False,
    formal_training_started=False, runtime_checked=False), indent=2) + '\n', encoding='utf-8')
print('FORMAL_PAYLOAD_PREPARED_ONLY; ACTUAL_PREFLIGHT_AND_SOURCE_GATE_REQUIRED')
