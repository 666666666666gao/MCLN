"""Prepare the proven closed-only observer/intake for the paired output root."""
import ast
from pathlib import Path

root = Path(__file__).resolve().parent
old = root.parent / 'pvground_reference_keep_20261006'
source = (old / 'observe_fit_authorized.py').read_text(encoding='utf-8')
assert source.count("local/'control_spec.json'") == 1
source = source.replace("local/'control_spec.json'", "local/'pair_spec.json'")
source = source.replace('REFERENCE_KEEP_OBSERVATION', 'PAIRED_FACE_OBSERVATION')
source = source.replace('REFERENCE_KEEP_OBSERVER_CLOSED', 'PAIRED_FACE_OBSERVER_CLOSED')
(root / 'observe_fit_authorized.py').write_text(source, encoding='utf-8')
source = (old / 'collect_closed_fit_authorized.py').read_text(encoding='utf-8')
assert source.count("spec=json.loads((local/'control_spec.json').read_bytes())") == 1
assert source.count("root=str(PurePosixPath(spec['root']).parent)") == 1
source = source.replace("spec=json.loads((local/'control_spec.json').read_bytes())",
                        "spec=json.loads((local/'pair_spec.json').read_bytes())")
source = source.replace("root=str(PurePosixPath(spec['root']).parent)", "root=spec['root']")
(root / 'collect_closed_fit_authorized.py').write_text(source, encoding='utf-8')
for name in ('observe_fit_authorized.py', 'collect_closed_fit_authorized.py', 'launch_fit_authorized.py'):
    ast.parse((root / name).read_text(encoding='utf-8'), feature_version=(3, 7))
print('FIT_OBSERVER_INTAKE_PREPARED_NOT_EXECUTED')
