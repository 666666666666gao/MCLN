"""Prepare only the changed files; do not deploy or edit the active source."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
original = root.parent / 'source'
overlay = root / 'source_overlay_v2'
assert not overlay.exists()
overlay.mkdir()
changes = {}


def amend(name, substitutions):
    raw = (original / name).read_bytes()
    text = raw.decode('utf-8').replace('\r\n', '\n')
    for before, after in substitutions:
        assert text.count(before) == 1, (name, before)
        text = text.replace(before, after)
    ast.parse(text)
    output = overlay / name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(text.encode('utf-8'))
    changes[name] = dict(original_sha256=hashlib.sha256(raw).hexdigest(),
                         patched_sha256=hashlib.sha256(output.read_bytes()).hexdigest())


amend('native_model_initialization.py', [
    ("    model.use_selected_mask_supervision = spec['use_selected_mask_supervision']",
     "    model.use_selected_mask_supervision = spec['use_selected_mask_supervision']\n"
     "    model.use_selected_geometry_supervision = spec['use_selected_geometry_supervision']"),
    ("        use_selected_mask_supervision=model.use_selected_mask_supervision,",
     "        use_selected_mask_supervision=model.use_selected_mask_supervision,\n"
     "        use_selected_geometry_supervision=model.use_selected_geometry_supervision,")])
amend('models/pv_ground.py', [
    ('        self.use_selected_mask_supervision = False',
     '        self.use_selected_mask_supervision = False\n        self.use_selected_geometry_supervision = False'),
    ("        end_points['use_selected_mask_supervision'] = self.use_selected_mask_supervision",
     "        end_points['use_selected_mask_supervision'] = self.use_selected_mask_supervision\n"
     "        end_points['use_selected_geometry_supervision'] = self.use_selected_geometry_supervision")])
amend('models/losses.py', [
    ("            end_points['selected_query_mask_extra_rows'] = record['extra_rows']",
     "            end_points['selected_query_mask_extra_rows'] = record['extra_rows']\n"
     "        if end_points['use_selected_geometry_supervision']:\n"
     "            from native_root_bbs import native_root_bbs\n"
     "            from selected_support_geometry_objective import selected_support_geometry_loss\n"
     "            scores = native_root_bbs(end_points['last_sem_cls_scores'].detach(), end_points)\n"
     "            extra, record = selected_support_geometry_loss(\n"
     "                end_points, end_points, last_indices, target, set_criterion, scores)\n"
     "            loss = loss + extra\n"
     "            end_points['selected_geometry_extra_loss'] = extra\n"
     "            end_points['selected_geometry_extra_rows'] = record['extra_rows']\n"
     "            end_points['selected_geometry_extra_record'] = record")])
(overlay / 'selected_support_geometry_objective.py').write_bytes((root / 'selected_support_geometry_objective.py').read_bytes())
ast.parse((overlay / 'selected_support_geometry_objective.py').read_text())

initial = json.loads((root.parent / 'init_manifests/extremal_support.json').read_bytes())
for mode in ('native_control', 'selected_geometry'):
    spec = dict(initial)
    spec.update(use_g_supervision=True, use_selected_mask_supervision=False,
                use_selected_geometry_supervision=mode == 'selected_geometry')
    (root / (mode + '_initialization.json')).write_text(json.dumps(spec, indent=2) + '\n', encoding='utf-8')
result = dict(status='ISOLATED_LOSS_FACTORY_FORWARD_OVERLAY_PREPARED_NOT_DEPLOYED',
    original_source=str(original), changed_source=str(overlay), files=changes,
    new_inference_parameters=0, original_states=1295, initialization_same_except_loss_flag=True,
    new_neural_forwards=0, GPU_calls=0, remote_queries=0, active_source_changes=0,
    complete_normal_entry_prepared=False, GPU_preflight_completed=False, formal_training_started=False,
    remaining_integration=['complete native loop/decline policy and single-card queue',
                           'fresh source review', 'real factory/criterion GPU two-step check',
                           'full common E0 and runtime training admission'])
(root / 'SOURCE_OVERLAY_V2.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps({key: result[key] for key in ('status', 'new_neural_forwards', 'active_source_changes')}))
