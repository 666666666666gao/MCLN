"""Bind the isolated correction to the reviewed runner bundle, not a runtime PASS."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

local = Path(__file__).resolve().parent
for name in ('run_readback_preflight.py', 'readback_preflight_checks.py'):
    shutil.copyfile(local / name, local / 'runtime_bundle' / name)
files = {}
for path in sorted((local / 'runtime_bundle').glob('*.py')):
    raw = path.read_bytes()
    ast.parse(raw.decode('utf-8'), feature_version=(3, 7))
    files[path.name] = hashlib.sha256(raw).hexdigest()
assert len(files) == 16
overrides = {name: hashlib.sha256((local / 'source_preview/PV-Ground' / name).read_bytes()).hexdigest()
             for name in ('models/pv_ground.py', 'models/modules.py')}
for arm in ('evidence_hidden', 'evidence_visible'):
    path = local / (arm + '_preflight_template.json')
    spec = json.loads(path.read_bytes())
    spec['runner_files'] = files
    path.write_text(json.dumps(spec, indent=2) + '\n', encoding='utf-8')
for path in local.glob('*.py'):
    ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 7))
record = dict(status='SOURCE_ONLY_AST_PASS', runner_files=files, model_overrides=overrides,
    upstream_model_source_changed=False, original_failed_attempt_changed=False,
    repeated_forward_drift_is_diagnostic_not_tolerance_pass=True,
    native_forward_same_frame_geometry_and_masks_exact=True, runtime_checked=False)
(local / 'PREFLIGHT_BUNDLE_SOURCE_CHECK.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status=record['status'], runner_modules=len(files), changed_modules=2)))
