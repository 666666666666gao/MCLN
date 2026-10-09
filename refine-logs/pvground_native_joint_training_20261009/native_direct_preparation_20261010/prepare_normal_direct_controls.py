"""Prepare native A/B controls locally; never alter or query the active run."""
import ast
import datetime
import hashlib
import json
from pathlib import Path


root = Path(__file__).resolve().parent
destination = root / 'native_direct_controls_20261010'
assert not destination.exists()
port = json.loads((root / 'NATIVE_SOURCE_PORT.json').read_bytes())
baseline_spec = json.loads((root / 'init_manifests/extremal_support.json').read_bytes())
baseline_protocol = json.loads((root / 'NORMAL_NATIVE_RUN_PROTOCOL.json').read_bytes())
original_hashes = {}
sources = {}
source_bytes = {}
for path in sorted((root / 'source').rglob('*.py')):
    name = path.relative_to(root / 'source').as_posix()
    source_bytes[name] = path.read_bytes()
    original_hashes[name] = hashlib.sha256(source_bytes[name]).hexdigest()
    assert original_hashes[name] == port['files'][name]['sha256']
    sources[name] = path.read_text(encoding='utf-8')
assert len(sources) == 14


def replace_once(name, old, new):
    assert sources[name].count(old) == 1, (name, old)
    sources[name] = sources[name].replace(old, new)


replace_once('mask_support_corrector.py',
    'def __init__(self, use_box_geometry):',
    'def __init__(self, use_box_geometry, correction_mode):')
replace_once('mask_support_corrector.py',
    '        self.use_box_geometry = use_box_geometry\n',
    "        assert correction_mode in ('full', 'content_only', 'bypass')\n"
    '        self.use_box_geometry = use_box_geometry\n'
    '        self.correction_mode = correction_mode\n')
replace_once('mask_support_corrector.py',
    "        count = own.new_tensor(geometry['count']).log1p() / math.log1p(50000)\n",
    "        count = own.new_tensor(geometry['count']).log1p() / math.log1p(50000)\n"
    "        if self.correction_mode == 'content_only':\n"
    '            mask_state = torch.zeros_like(mask_state)\n'
    '            count = torch.zeros_like(count)\n')
replace_once('mask_support_corrector.py',
    "            corrected.append(self.correct_one(query[bid], supports[bid], center[bid], size[bid],\n"
    "                predictions['last_pred_masks'][bid][0], predictions['sp_last_pred_masks'][bid],\n"
    "                predictions['adaptive_weights'][bid], geometry))\n",
    "            if self.correction_mode == 'bypass':\n"
    "                corrected.append(predictions['sp_last_pred_masks'][bid])\n"
    '            else:\n'
    '                corrected.append(self.correct_one(query[bid], supports[bid], center[bid], size[bid],\n'
    "                    predictions['last_pred_masks'][bid][0], predictions['sp_last_pred_masks'][bid],\n"
    "                    predictions['adaptive_weights'][bid], geometry))\n")

replace_once('extremal_span_mixer.py',
    'def __init__(self, source_mode):',
    'def __init__(self, source_mode, mixing_mode):')
replace_once('extremal_span_mixer.py',
    '        self.source_mode = source_mode\n',
    "        assert mixing_mode in ('learned', 'fixed_half')\n"
    '        self.source_mode = source_mode\n'
    '        self.mixing_mode = mixing_mode\n')
replace_once('extremal_span_mixer.py',
    '        raw_gate = self.output(self.axis_decoder(decoder_input)).squeeze(-1)\n',
    '        raw_gate = self.output(self.axis_decoder(decoder_input)).squeeze(-1)\n'
    "        if self.mixing_mode == 'fixed_half':\n"
    '            raw_gate = torch.full_like(raw_gate, .5)\n')

replace_once('native_model_initialization.py',
    "    assert spec['span_source_mode'] in ('whole_support', 'extremal_support')\n",
    "    assert spec['span_source_mode'] in ('whole_support', 'extremal_support')\n"
    "    assert spec['support_correction_mode'] in ('full', 'content_only', 'bypass')\n"
    "    assert spec['span_mixing_mode'] in ('learned', 'fixed_half')\n")
replace_once('native_model_initialization.py',
    '    model.candidate_support_corrector = CandidateMaskSupportCorrector(False)\n',
    "    model.candidate_support_corrector = CandidateMaskSupportCorrector(False, spec['support_correction_mode'])\n")
replace_once('native_model_initialization.py',
    "    model.candidate_span_mixer = ExtremalSpanMixer(spec['span_source_mode'])\n",
    "    model.candidate_span_mixer = ExtremalSpanMixer(spec['span_source_mode'], spec['span_mixing_mode'])\n")
replace_once('native_model_initialization.py',
    '    assert len(model.state_dict()) == 1295\n',
    "    if spec['support_correction_mode'] == 'bypass':\n"
    '        model.candidate_support_corrector.requires_grad_(False)\n'
    "    if spec['span_mixing_mode'] == 'fixed_half':\n"
    '        model.candidate_span_mixer.requires_grad_(False)\n'
    '    assert len(model.state_dict()) == 1295\n')
replace_once('native_model_initialization.py',
    "        span_source_mode=spec['span_source_mode'], full_state_tensors=1295,\n",
    "        span_source_mode=spec['span_source_mode'], full_state_tensors=1295,\n"
    "        support_correction_mode=spec['support_correction_mode'],\n"
    "        span_mixing_mode=spec['span_mixing_mode'],\n"
    "        support_parameters_trainable=spec['support_correction_mode'] != 'bypass',\n"
    "        span_parameters_trainable=spec['span_mixing_mode'] != 'fixed_half',\n")

for name, source in sources.items():
    ast.parse(source, filename=name, feature_version=(3, 7))
    path = destination / 'source' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if name in ('mask_support_corrector.py', 'extremal_span_mixer.py', 'native_model_initialization.py'):
        path.write_text(source, encoding='utf-8', newline='')
    else:
        path.write_bytes(source_bytes[name])

common_spec = dict(baseline_spec, support_correction_mode='full', span_mixing_mode='learned')
variants = {
    'common_behavior_check': dict(common_spec),
    'support_bypass': dict(common_spec, support_correction_mode='bypass'),
    'support_content_only': dict(common_spec, support_correction_mode='content_only'),
    'fixed_half_geometry': dict(common_spec, span_mixing_mode='fixed_half'),
}
controller = (root / 'normal_joint_controller.py').read_text(encoding='utf-8')
old = "protocol['model_source']+'/init_manifests/'+mode+'.json'"
assert controller.count(old) == 1
controller = controller.replace(old, "protocol['native_init_spec']")
ast.parse(controller, feature_version=(3, 7))
remote_source = '/root/autodl-tmp/pvground_native_direct_controls_20261010/PV-Ground'
config_differences = {}
for arm, spec in variants.items():
    config_differences[arm] = [key for key in common_spec if common_spec[key] != spec[key]]
    expected = {
        'common_behavior_check': [],
        'support_bypass': ['support_correction_mode'],
        'support_content_only': ['support_correction_mode'],
        'fixed_half_geometry': ['span_mixing_mode'],
    }[arm]
    assert config_differences[arm] == expected
    planned_root = '/root/autodl-tmp/pvground_native_direct_controls_20261010/' + arm
    protocol = dict(baseline_protocol, model_source=remote_source,
                    native_init_spec=planned_root + '/init.json')
    path = destination / arm
    path.mkdir()
    (path / 'init.json').write_text(json.dumps(spec, indent=2) + '\n', encoding='utf-8')
    (path / 'NORMAL_NATIVE_RUN_PROTOCOL.json').write_text(
        json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    (path / 'normal_joint_controller.py').write_text(controller, encoding='utf-8')

for name, digest in original_hashes.items():
    assert hashlib.sha256((root / 'source' / name).read_bytes()).hexdigest() == digest
prepared_hashes = {str(path.relative_to(destination).as_posix()): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(destination.rglob('*')) if path.is_file()}
changed_sources = [name for name in sources
    if hashlib.sha256((destination / 'source' / name).read_bytes()).hexdigest() != original_hashes[name]]
assert set(changed_sources) == {'mask_support_corrector.py', 'extremal_span_mixer.py', 'native_model_initialization.py'}
record = dict(
    status='NORMAL_NATIVE_DIRECT_CONTROLS_SOURCE_PREPARED_NOT_CONSTRUCTED',
    time_cst=datetime.datetime.now().astimezone().isoformat(),
    config_differences=config_differences, changed_sources=changed_sources,
    original_source_sha256=original_hashes, prepared_sha256=prepared_hashes,
    active_source_unchanged=True, current_native_training_source_mutations=0,
    original_core_policy_unchanged=True, criterion_score_dataset_and_training_entry_unchanged=True,
    model_state_tensor_count_expected=1295, same_parent_state_history=True,
    actual_state_load_constructor_optimizer_or_forward_checked=False,
    common_behavior_identity_requires_actual_same_batch_check=True,
    control_initial_outputs_expected_to_differ=True,
    bypass_A_parameters_registered=27841, bypass_A_parameters_frozen=True,
    fixed_B_parameters_registered=29793, fixed_B_parameters_frozen=True,
    content_only_A_parameters_trainable=True,
    claim_scope='Equal-budget normal continuation from common trained parents; not never-A/B/C training histories',
    same_training_seed=2027, planned_epochs=3, batch_size=8, effective_batch_size=8,
    fresh_optimizer_for_each_arm=True, multiseed=False,
    source_audit_pending=True, actual_CPU_Torch_and_GPU_preflight_pending=True,
    before_launch_requires_current_normal_terminal_and_result_review=True,
    remote_experiment_directories_created=0, SSH_calls=0, neural_calls=0, training_launched=False,
    full_goal_complete=False)
(destination / 'DIRECT_CONTROL_PREPARATION.json').write_text(
    json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({key: record[key] for key in (
    'status', 'config_differences', 'changed_sources', 'active_source_unchanged', 'neural_calls', 'training_launched')}))
