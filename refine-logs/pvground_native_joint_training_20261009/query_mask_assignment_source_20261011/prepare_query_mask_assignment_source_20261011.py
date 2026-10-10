"""Prepare the minimal native criterion source switch; no training or deployment."""
import ast
import datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
out = root / 'query_mask_assignment_20261011'
assert not out.exists()
diagnostic = json.loads((root / 'matcher_mask_role_cpu_20261011/ACTUAL_CHECK_SUMMARY.json').read_bytes())
assert diagnostic['status'] == 'ACTUAL_NATIVE_MATCHER_CPU_CHECK_AUDITED'
assert diagnostic['actual_review_blocking_findings'] == []
base = json.loads((root / 'normal_controls_20261010/CONTROL_PREPARATION.json').read_bytes())['source_sha256']
out.mkdir()
source = out / 'source'
source.mkdir()
hashes = {}
for name, digest in base.items():
    raw = (root / 'source' / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest
    if name == 'models/losses.py':
        text = raw.decode('utf-8')
        before = '                 soft_token=False):'
        assert text.count(before) == 1
        text = text.replace(before, '                 soft_token=False, mask_source="text"):')
        before = '        self.cost_masks = 0.0002  # mask weight'
        assert text.count(before) == 1
        text = text.replace(before, before + '\n        self.mask_source = mask_source')
        before = '                out_mask = outputs["pred_masks"][idx].squeeze(0)  # [Q, super_num]'
        assert text.count(before) == 1
        text = text.replace(before,
            '                out_mask = (outputs["sp_pred_masks"][idx] if self.mask_source == "query"\n'
            '                            else outputs["pred_masks"][idx].squeeze(0))  # [Q, super_num]')
        raw = text.encode('utf-8')
    elif name == 'main_utils.py':
        text = raw.decode('utf-8')
        before = "    parser.add_argument('--lr_modules', type=float, default=1e-5)"
        assert text.count(before) == 1
        text = text.replace(before, before + '\n'
            "    parser.add_argument('--matcher_mask_source', choices=('text', 'query'), default='text',\n"
            "                        help='Training matcher Mask-cost source; inference outputs are unchanged')")
        before = '        matcher = HungarianMatcher(1, 0, 2, args.use_soft_token_loss)'
        assert text.count(before) == 1
        text = text.replace(before,
            '        matcher = HungarianMatcher(1, 0, 2, args.use_soft_token_loss,\n'
            '                                   mask_source=args.matcher_mask_source)')
        raw = text.encode('utf-8')
    path = source / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    ast.parse(raw.decode('utf-8'), feature_version=(3, 7))
    hashes[name] = hashlib.sha256(raw).hexdigest()
assert [name for name in base if base[name] != hashes[name]] == ['main_utils.py', 'models/losses.py']
(out / 'BASE_SOURCE_HASHES.json').write_text(json.dumps(base, indent=2) + '\n')
(out / 'SOURCE_HASHES.json').write_text(json.dumps(hashes, indent=2) + '\n')
spec = dict(status='ISOLATED_NATIVE_QUERY_MASK_ASSIGNMENT_SOURCE_PREPARED_NOT_ADMITTED',
    prepared_cst=datetime.datetime.now().astimezone().isoformat(),
    warm_source='/root/autodl-tmp/pvground_native_joint_training_20261010/PV-Ground',
    env_spec_sha256='966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c',
    common_parent_scope='Same retained E0 core/G/A/B state, including disclosed earlier C adaptation history',
    training_control_mask_source='text', training_strategy_mask_source='query',
    changed_source_files=['main_utils.py', 'models/losses.py'],
    changed_cost='Last-layer direct Mask cost source only; original hard threshold, point mapping, L1 metric and coefficient retained',
    class_cost=1, bbox_cost=0, giou_cost=2, mask_cost=0.0002,
    intermediate_layer_matching='Unchanged; no Mask predictions in the six earlier prefixes',
    target_policy='All original valid GT; native one-to-one matching, without root-only remapping or extra positives',
    selected_mask_supervision=False, G_supervision='unchanged',
    added_model_parameters=0, model_forward_changed=False, original_inference_score='last/bbs',
    candidate_queries=256, candidate_pruning=False, inference_GT_inputs_added=False,
    seed=2027, multiseed=False, normal_training_planned_epochs=3,
    batch_size=8, effective_batch_size=8, accumulation=1,
    lr=1e-6, lr_backbone=1e-6, lr_modules=1e-5, weight_decay=0.0005, clip_norm=0.1,
    planned_formal_rows=9508, final_method_admitted=False,
    actual_source_review_completed=False, actual_CPU_check_completed=False,
    native_GPU_preflight_completed=False, training_started=False,
    active_training_source_mutations=0, current_training_queries=0,
    next_main_observation_cst='2026-10-11T05:00:40.968644+08:00', full_goal_complete=False)
(out / 'PLAN.json').write_text(json.dumps(spec, indent=2) + '\n')
print(json.dumps(dict(status=spec['status'], changed_source_files=spec['changed_source_files'],
    added_model_parameters=0, active_training_source_mutations=0, training_started=False)))
