"""Prepare isolated native PV model/loss sources without touching the active fit."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
tmp = root.parent
span = tmp / 'pvground_extremal_span_evidence_20261009/runner_v1'
prior = tmp / 'pvground_selected_mask_training_20261009'
base = prior / 'postrun_results/audit_sources'
g_source = Path('C:/Users/gb/.codex_pvground_cs_20261002/refine-logs/pvground_scanrefer_finetune_20260918_semantic_assignment_v1/pvground_semantic_assignment.py')
native_train = Path('C:/Users/gb/.codex_pvground_cs_20261002/refine-logs/pvground_pretrained_resources_20260908_v1/source/train_dist_mod.py')
dest = root / 'source'
assert not dest.exists()
(dest / 'models').mkdir(parents=True)
inputs = {}

def original(path):
    raw = path.read_bytes()
    inputs[str(path)] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    return raw.decode('utf-8').replace('\r\n', '\n')

def replace_once(text, old, new):
    assert text.count(old) == 1, old[:80]
    return text.replace(old, new, 1)

model = original(base / 'models_pv_ground.py')
model = replace_once(model, '        self.candidate_support_corrector = None',
    '        self.candidate_support_corrector = None\n'
    '        self.candidate_span_mixer = None\n'
    '        self.use_predicted_mask_reference = False\n'
    '        self.use_g_supervision = False\n'
    '        self.use_selected_mask_supervision = False')
tail = '''        if self.use_predicted_mask_reference:
            from native_mask_geometry import native_mask_geometry
            if self.candidate_support_corrector is None:
                from whole_mask_range import member_statistics
                xyz = inputs['points'][:, 1:4].view(inputs['batch_size'], 50000, 3).detach().cpu().numpy()
                geometries = [member_statistics(xyz[bid], end_points['superpoints'][bid].detach().cpu().numpy(), bins=32)
                              for bid in range(inputs['batch_size'])]
            else:
                geometries = end_points['support_member_geometry']
            native_mask_geometry(end_points, geometries)

        if self.candidate_span_mixer is not None:
            assert self.use_predicted_mask_reference
            center, size, evidence = self.candidate_span_mixer(
                tail_geometry_query, super_features, end_points, geometries)
            end_points['span_mask_center'] = end_points['last_center']
            end_points['span_mask_size'] = end_points['last_pred_size']
            end_points['last_center'], end_points['last_pred_size'] = center, size
            end_points['span_axis_gate'] = torch.stack([row['axis_gate'] for row in evidence])
            end_points['span_raw_axis_gate'] = torch.stack([row['raw_axis_gate'] for row in evidence])
            end_points['span_source_fraction'] = torch.stack([row['source_fraction'] for row in evidence])

        end_points['native_joint_training'] = self.training
        end_points['use_g_supervision'] = self.use_g_supervision
        end_points['use_selected_mask_supervision'] = self.use_selected_mask_supervision

'''
model = replace_once(model, '        return end_points\n\n    def init_bn_momentum(self):',
    tail + '        return end_points\n\n    def init_bn_momentum(self):')
(dest / 'models/pv_ground.py').write_text(model, encoding='utf-8')

mixer = original(span / 'extremal_span_mixer.py')
mixer = replace_once(mixer, 'Prepared source only. The existing frozen parent supplies native regression,',
    'Native joint-training source. The trainable PV model supplies native regression,')
mixer = replace_once(mixer, '        # Parents are frozen. The native evaluator/layout already floors the',
    '        # The native evaluator/layout already floors the')
mixer = replace_once(mixer, '        native_center = native_center.detach()\n', '')
mixer = replace_once(mixer, 'native_size.detach().clamp_min(1e-6)', 'native_size.clamp_min(1e-6)')
mixer = replace_once(mixer, '        mask_center, mask_size = mask_center.detach(), mask_size.detach()\n', '')
mixer = replace_once(mixer, '        text, own, alpha = text.detach(), own.detach(), alpha.detach()\n', '')
mixer = replace_once(mixer, 'support[:, slots].T.detach()', 'support[:, slots].T')
mixer = replace_once(mixer, 'self.query_projection(query.detach())', 'self.query_projection(query)')
(dest / 'extremal_span_mixer.py').write_text(mixer, encoding='utf-8')

reference = original(span / 'mask_reference.py')
function = next(node for node in ast.parse(reference).body if isinstance(node, ast.FunctionDef) and node.name == 'extent_reference')
reference_function = ast.get_source_segment(reference, function)
geometry = '"""Discrete prediction-only Mask reference inside native model.forward."""\nimport torch\n\n\n' + reference_function + '''


def native_mask_geometry(predictions, geometries):
    native_center = predictions['last_center']
    native_size = predictions['last_pred_size']
    centers, sizes, validities = [], [], []
    for bid, geometry in enumerate(geometries):
        center, size, valid = extent_reference(native_center[bid], native_size[bid],
            predictions['last_pred_masks'][bid][0], predictions['sp_last_pred_masks'][bid],
            predictions['adaptive_weights'][bid], geometry)
        centers.append(center)
        sizes.append(size.clamp_min(1e-6))
        validities.append(valid)
    predictions['native_coarse_center'] = native_center
    predictions['native_coarse_size'] = native_size
    predictions['mask_reference_valid'] = torch.stack(validities)
    predictions['mask_reference_center'] = torch.stack(centers)
    predictions['mask_reference_size'] = torch.stack(sizes)
    predictions['last_center'] = predictions['mask_reference_center']
    predictions['last_pred_size'] = predictions['mask_reference_size']
'''
(dest / 'native_mask_geometry.py').write_text(geometry, encoding='utf-8')
support = original(span / 'mask_support_corrector.py')
support = support[:support.index('\ndef install_support_correction(')]
(dest / 'mask_support_corrector.py').write_text(support, encoding='utf-8')
selected_loss = original(prior / 'selected_query_mask_objective.py')
selected_loss = selected_loss[:selected_loss.index('\ndef selected_output_gradient_witness(')]
selected_loss = replace_once(selected_loss, 'score_path_frozen=True',
    'score_path_frozen=False, selection_gradient_stopped=True')
(dest / 'selected_query_mask_objective.py').write_text(selected_loss, encoding='utf-8')
(dest / 'pvground_semantic_assignment.py').write_text(original(g_source), encoding='utf-8')

losses = original(base / 'models_losses.py')
losses = replace_once(losses, '        losses, _ = set_criterion(output, target)',
    '        losses, indices = set_criterion(output, target)\n'
    "        if prefix == 'last_':\n"
    '            last_indices = indices')
extra = '''    if end_points['native_joint_training']:
        if end_points['use_g_supervision']:
            from pvground_semantic_assignment import semantic_assignment_correction
            correction, record = semantic_assignment_correction(end_points, end_points, last_indices, set_criterion.eos_coef)
            loss = loss + correction
            end_points['g_reassigned_queries'] = record['reassigned_queries']
        if end_points['use_selected_mask_supervision']:
            from selected_query_mask_objective import selected_query_mask_loss
            extra, record = selected_query_mask_loss(end_points, end_points, last_indices, target)
            loss = loss + extra
            end_points['selected_query_mask_extra_loss'] = extra
            end_points['selected_query_mask_extra_rows'] = record['extra_rows']

'''
losses = replace_once(losses, "    end_points['loss_ce'] = loss_ce", extra + "    end_points['loss_ce'] = loss_ce")
(dest / 'models/losses.py').write_text(losses, encoding='utf-8')
(dest / 'train_dist_mod.py').write_text(original(native_train), encoding='utf-8')
(dest / 'main_utils.py').write_text(original(base / 'main_utils.py'), encoding='utf-8')

files = {}
for path in dest.rglob('*.py'):
    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    files[path.relative_to(dest).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
receipt = dict(status='NATIVE_MODEL_AND_CRITERION_SOURCE_PREPARED_NOT_DEPLOYED',
    input_sources=inputs, source_files=files, native_model_class_modified=True,
    final_box_loss_uses_deployed_output=True, connected_feature_and_native_box_graph=True,
    hard_reference_selection_remains_discrete=True, parameter_factory_and_entry_pending=True,
    actual_gpu_preflight_passed=False, native_joint_training_started=False,
    active_frozen_pair_changed=False, new_weights_created=0, accuracy_result=False)
(root / 'SOURCE_PREPARATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({key:value for key,value in receipt.items() if key not in ('input_sources', 'source_files')}))
