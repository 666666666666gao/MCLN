"""Reuse the actual native C-off preflight for a future serial matcher test.

Source preparation only. The generated protocols explicitly deny GPU execution.
"""
import ast
import datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
prior = root / 'normal_controls_20261010'
query = root / 'query_mask_assignment_20261011'
out = root / 'query_mask_assignment_gpu_20261011'
cpu = json.loads((query / 'CPU_CHECK_SUMMARY.json').read_bytes())
assert cpu['status'] == 'ISOLATED_NATIVE_QUERY_MASK_ASSIGNMENT_CPU_CHECK_AUDITED'
assert not cpu['actual_review_blocking_findings']
assert not out.exists()
out.mkdir()
hashes = json.loads((query / 'SOURCE_HASHES.json').read_bytes())
for name, digest in hashes.items():
    raw = (query / 'source' / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest
    target = out / 'source' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
(out / 'SOURCE_HASHES.json').write_text(json.dumps(hashes, indent=2) + '\n')
(out / 'init.json').write_bytes((prior / 'init.json').read_bytes())
remote_root = '/root/autodl-tmp/pvground_query_mask_assignment_20261011'
common = json.loads((prior / 'NORMAL_NATIVE_RUN_PROTOCOL.json').read_bytes())
for mode in ('text', 'query'):
    protocol = dict(common)
    protocol.update(status='SOURCE_ONLY_NATIVE_MATCHER_GPU_PREFLIGHT_NOT_ADMITTED',
        model_source=remote_root + '/PV-Ground', native_init_spec=remote_root + '/init.json',
        matcher_mask_source=mode, serial_gpu_preflight_admitted=False, full_training_admitted=False,
        current_c_off_must_be_terminal=True, actual_normal_model_constructed=False,
        actual_normal_gpu_preflight=False, normal_joint_training_started=False,
        intended_updates=2, intended_real_rows=16, actual_updates=0,
        new_accuracy_result=False, source_sha256=hashes, full_goal_complete=False)
    (out / (mode + '.json')).write_text(json.dumps(protocol, indent=2) + '\n')

source = (prior / 'native_c_off_preflight.py').read_text(encoding='utf-8')
changes = [
    ("protocol = json.loads(Path(options.protocol).read_bytes())",
     "protocol = json.loads(Path(options.protocol).read_bytes())\n"
     "assert protocol['serial_gpu_preflight_admitted'], 'Source preparation is not GPU admission'\n"
     "assert protocol['matcher_mask_source'] in ('text', 'query')"),
    ("'--native_init_spec', protocol['native_init_spec'],",
     "'--native_init_spec', protocol['native_init_spec'],\n"
     "    '--matcher_mask_source', protocol['matcher_mask_source'],"),
    ("assert args.lr == args.lr_backbone == 1e-6 and args.lr_modules == 1e-5",
     "assert args.lr == args.lr_backbone == 1e-6 and args.lr_modules == 1e-5\n"
     "assert args.matcher_mask_source == protocol['matcher_mask_source']"),
    ("criterion, set_criterion = tester.get_criterion(args)",
     "criterion, set_criterion = tester.get_criterion(args)\n"
     "assert set_criterion.matcher.mask_source == args.matcher_mask_source\n"
     "assert (set_criterion.matcher.cost_class, set_criterion.matcher.cost_bbox,\n"
     "        set_criterion.matcher.cost_giou, set_criterion.matcher.cost_masks) == (1, 0, 2, .0002)"),
    ("assert saved['retained_metrics'] is None and saved['architecture'] == model.module.native_training_architecture",
     "assert saved['retained_metrics'] is None and saved['architecture'] == model.module.native_training_architecture\n"
     "assert saved['config'].matcher_mask_source == args.matcher_mask_source"),
    ("load_checkpoint(args, restored_model, restored_optimizer, restored_scheduler)",
     "load_checkpoint(args, restored_model, restored_optimizer, restored_scheduler)\n"
     "restored_criterion, restored_set_criterion = tester.get_criterion(args)\n"
     "assert restored_set_criterion.matcher.mask_source == saved['config'].matcher_mask_source"),
    ("receipt = dict(status='ACTUAL_NATIVE_C_OFF_TWO_STEP_INITIAL_E0_EQUALITY_AND_FULL_RECOVERY_COMPLETE',",
     "receipt = dict(status='ACTUAL_NATIVE_MATCHER_TWO_STEP_AND_FULL_RECOVERY_COMPLETE',\n"
     "    matcher_mask_source=args.matcher_mask_source,\n"
     "    matcher_source_saved_and_reconstructed=True, native_last_matching_checks=matching_records,"),
]
for before, after in changes:
    assert source.count(before) == 1
    source = source.replace(before, after)
before = 'original_compute_loss = tester._compute_loss'
assert source.count(before) == 1
matching = '''from models.losses import HungarianMatcher, box_cxcyczwhd_to_xyzxyz, _iou3d_par
original_matching = set_criterion.matcher.forward
other_source = 'text' if args.matcher_mask_source == 'query' else 'query'
other_matcher = HungarianMatcher(1, 0, 2, args.use_soft_token_loss, mask_source=other_source)
matching_records = []


@torch.no_grad()
def observed_matching(outputs, targets):
    configured = original_matching(outputs, targets)
    if 'pred_masks' not in outputs:
        return configured
    alternative = other_matcher(outputs, targets)
    rows = []
    for b, target in enumerate(targets):
        current_query, current_gt = configured[b]
        other_query, other_gt = alternative[b]
        count = len(target['boxes'])
        assert current_query.unique().numel() == current_gt.unique().numel() == count
        assert other_query.unique().numel() == other_gt.unique().numel() == count
        current_order = current_gt.argsort()
        other_order = other_gt.argsort()
        current_query, current_gt = current_query[current_order], current_gt[current_order]
        other_query, other_gt = other_query[other_order], other_gt[other_order]
        assert torch.equal(current_gt, other_gt)
        metrics = {}
        for name, queries, mask_source in (('configured', current_query, args.matcher_mask_source),
                                          ('alternative', other_query, other_source)):
            boxes = outputs['pred_boxes'][b, queries]
            gt_boxes = target['boxes'][current_gt]
            iou, _ = _iou3d_par(box_cxcyczwhd_to_xyzxyz(boxes), box_cxcyczwhd_to_xyzxyz(gt_boxes))
            own = outputs['sp_pred_masks'][b][queries]
            member_ids = outputs['superpoints'][b].unsqueeze(0).expand(count, -1)
            own_points = torch.gather(own, 1, member_ids) > 0
            gt_mask = target['masks'][current_gt] > 0
            intersection = (own_points & gt_mask).sum(-1).float()
            union = (own_points | gt_mask).sum(-1).float()
            matching_mask = (outputs['pred_masks'][b].squeeze(0)[queries] if mask_source == 'text'
                             else outputs['sp_pred_masks'][b][queries])
            matching_points = torch.gather(matching_mask, 1, member_ids) > 0
            metrics[name] = dict(box_iou=torch.diag(iou).tolist(), own_mask_intersection=intersection.tolist(),
                own_mask_union=union.tolist(),
                own_query_mask_cost=(.0002 * (own_points != gt_mask).sum(-1).float()).tolist(),
                matching_mask_source=mask_source,
                matching_mask_cost=(.0002 * (matching_points != gt_mask).sum(-1).float()).tolist())
        rows.append(dict(batch_row=b, valid_gt_count=count, gt=current_gt.tolist(),
            configured_query=current_query.tolist(), alternative_query=other_query.tolist(),
            changed_gt_pairs=int((current_query != other_query).sum()), metrics=metrics))
    matching_records.append(dict(training_step=current_step[0] + 1, configured_source=args.matcher_mask_source,
        alternative_source=other_source, rows=rows))
    return configured


set_criterion.matcher.forward = observed_matching
'''
source = source.replace(before, matching + '\n' + before)
before = 'assert current_step[0] == 1 and len(loss_records) == 2'
assert source.count(before) == 1
source = source.replace(before, before + "\nassert len(matching_records) == 2\nset_criterion.matcher.forward = original_matching")
ast.parse(source, feature_version=(3, 7))
helper = out / 'native_query_mask_matcher_preflight.py'
helper.write_text(source, encoding='utf-8')
plan = dict(status='NATIVE_MATCHER_REAL_BATCH_GPU_PREFLIGHT_SOURCE_PREPARED_NOT_ADMITTED',
    prepared_cst=datetime.datetime.now().astimezone().isoformat(),
    copied_native_source_files=14, native_source_equal_to_reviewed_query_variant=True,
    prepared_helper=str(helper), helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
    reused_actual_c_off_preflight_sha256=hashlib.sha256((prior / 'native_c_off_preflight.py').read_bytes()).hexdigest(),
    planned_remote_root=remote_root, remote_directories_created=0,
    planned_modes=['text', 'query'], planned_real_rows_per_mode=16,
    original_train_one_epoch_planned=True, full_native_criterion_planned=True,
    same_initial_1295_state_expected=True, fresh_optimizer_per_mode=True,
    mode_in_checkpoint_config_and_fresh_criterion_reconstruction_checked=True,
    comparative_mask_and_box_diagnostics='Same native outputs/targets, configured versus other matching source; return configured assignments only',
    all_valid_GT_native_one_to_one=True, original_G_and_C_off_retained=True,
    original_RoBERTa_freeze=True, normal_core_backbone_G_A_B_trainable=True,
    batch_size=8, effective_batch_size=8, seed=2027, multiseed=False,
    lr=1e-6, lr_backbone=1e-6, lr_modules=1e-5,
    actual_CPU_or_GPU_model_calls=0, actual_model_forward_calls=0, actual_updates=0,
    source_review_completed=False, actual_GPU_preflight_completed=False,
    serial_gpu_preflight_admitted=False, full_training_admitted=False,
    current_training_queries=0, current_training_source_mutations=0,
    queue_or_launcher_created=False, formal_accuracy=None, full_goal_complete=False,
    next_main_observation_cst='2026-10-11T05:00:40.968644+08:00')
(out / 'PREPARATION.json').write_text(json.dumps(plan, indent=2) + '\n')
print(json.dumps(dict(status=plan['status'], source_review_completed=False,
    actual_GPU_preflight_completed=False, current_training_queries=0)))
