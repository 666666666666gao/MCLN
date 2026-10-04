"""Assemble source-only readback probes; this never contacts compute."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

local = Path(__file__).resolve().parent
face = local.parent / 'pvground_face_conditioned_20261004'
base_spec = json.loads((local.parent / 'pvground_boundary_distribution_20261004/distribution_spec.json').read_bytes())
bundle = local / 'runtime_bundle'
assert not bundle.exists()
bundle.mkdir()
names = ['run_readback_preflight.py', 'readback_model_factory.py', 'readback_preflight_checks.py',
    'pvground_boundary_evidence_readback.py', 'install_boundary_evidence_readback.py',
    'native_root_bbs.py', 'pvground_boundary_box_refiner.py']
shared = ['pvground_candidate_box_refiner.py', 'pvground_tail_support_box_refiner.py',
    'pvground_whole_mask_box_refiner.py', 'whole_mask_range.py', 'pvground_source_query.py',
    'pvground_observation_query.py', 'pvground_task_observation_query.py',
    'pvground_semantic_assignment.py', 'whole_model_preflight_checks.py']
for name in names:
    shutil.copyfile(local / name, bundle / name)
for name in shared:
    shutil.copyfile(face / name, bundle / name)
files = {}
for path in sorted(bundle.iterdir()):
    raw = path.read_bytes()
    ast.parse(raw.decode('utf-8'), filename=path.name, feature_version=(3, 7))
    files[path.name] = hashlib.sha256(raw).hexdigest()
root = '/root/autodl-tmp/pvground_readback_preflight_20261004'
common = {key: base_spec[key] for key in ('runtime', 'input_manifest', 'reference_fixtures',
    'training_interface_receipt', 'parent_source_port', 'env_spec_sha256', 'checkpoint_sha256',
    'base_terminal', 'base_terminal_sha256')}
common.update(batch_size=8, seed=2027, lr=1e-5, weight_decay=.0005, clip_norm=.1,
    geometry_terminal='/root/autodl-tmp/pvground_boundary_fit_20261004/distribution/terminal.pth',
    geometry_terminal_sha256='79e35068b8787a81c354c5fd3ed2bbc12fc62e06167b9cfbe781f86cbbd36c67',
    geometry_hits50=4506, geometry_frozen=True, runner_files=files,
    model_source=root + '/PV-Ground', source_port=root + '/source_port.json')
for arm, enabled in (('evidence_hidden', False), ('evidence_visible', True)):
    spec = dict(common, root=root + '/' + arm, use_geometry_evidence=enabled)
    (local / (arm + '_preflight_template.json')).write_text(json.dumps(spec, indent=2) + '\n', encoding='utf-8')
receipt = dict(status='SOURCE_BUNDLE_PYTHON37_AST_ONLY', runner_files=files,
    model_overrides={name: hashlib.sha256((local / 'source_preview/PV-Ground' / name).read_bytes()).hexdigest()
        for name in ('models/pv_ground.py', 'models/modules.py')},
    cpu_factory_constructed=False, gpu_updates=0, accuracy_result=False,
    formal_budget=dict(fit_rows=29778, effective_batch=8, full_batches=3722, tail_rows=2, updates=3723))
(local / 'PREFLIGHT_BUNDLE_SOURCE_CHECK.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status=receipt['status'], ast_files=len(files), native_execution=False)), flush=True)
