"""Summarize only the actual closed real-model sanity evidence."""
import hashlib
import json
from pathlib import Path

local = Path(__file__).resolve().parent
intake = json.loads((local / 'complete_preflight/INTAKE.json').read_bytes())
assert intake['status']['status'] == 'complete' and intake['controller_exit'] == 0
assert not intake['controller_alive'] and intake['weight_files_created'] == 0
for relative, item in intake['files'].items():
    path = local / 'complete_preflight' / relative
    raw = path.read_bytes()
    assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
proof = json.loads((local / 'complete_preflight/face_conditioned/preflight.json').read_bytes())
assert proof['status'] == 'pass' and proof['optimizer_steps'] == 2 and proof['batch_size'] == 8
assert proof['head_architecture'] == 'face_conditioned' and proof['head_parameters'] == 64737
assert proof['original_g_state_unchanged'] and proof['same_cached_inputs_zero_head_common_floor_exact']
assert proof['optimizer_exact_check']['all_keys_moments_steps_and_groups_exact']
assert proof['weight_files_created'] == proof['formal_rows'] == 0
assert all(value == 0 for value in proof['zero_output_differences'].values())
assert len(proof['native_call_witnesses']) == 4
assert len(proof['steps']) == 2
for step in proof['steps']:
    assert step['direct_geometry_output_gradient'] > 0 and step['edge_alone_output_gradient'] > 0
    assert not step['semantic_loss_requires_grad'] and not step['mask_outputs_require_grad']
    assert not step['range_observation_requires_grad']
gradients = proof['steps'][1]['p3_gradients']
assert len(gradients) == 25 and all(value > 0 for value in gradients.values())
record = dict(status='REAL_MODEL_PREFLIGHT_PASS', measured_head_parameters=64737,
    measured_trainable_tensors=25, real_optimizer_updates=2, real_batch_size=8,
    uniform_output_common_floor_exact=True, frozen_original_g_unchanged=True,
    direct_geometry_output_gradients=[step['direct_geometry_output_gradient'] for step in proof['steps']],
    edge_output_gradients=[step['edge_alone_output_gradient'] for step in proof['steps']],
    second_step_all25_gradients_positive=True, actual_model_optimizer_restore_exact=True,
    allocator_peak_bytes=proof['peak_allocated_bytes'], reserved_peak_bytes=proof['peak_reserved_bytes'],
    serialization_bytes=proof['serialization_bytes'], wall_seconds=proof['runner_wall_seconds_through_restore'],
    weight_files_created=0, formal_accuracy_available=False)
(local / 'PREFLIGHT_ANALYSIS.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps(record))
