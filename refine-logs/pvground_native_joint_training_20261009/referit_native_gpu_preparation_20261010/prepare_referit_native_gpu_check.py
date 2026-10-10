"""Prepare, but do not admit or launch, the C-off ReferIt native GPU check."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parent
previous = root / 'referit_same_query_20261010'
destination = root / 'referit_native_gpu_preparation_20261010'
assert not destination.exists()
destination.mkdir()
shutil.copytree(str(previous / 'source'), str(destination / 'source'))
for dataset in ('nr3d', 'sr3d'):
    spec = json.loads((previous / dataset / 'init.json').read_bytes())
    assert spec['new_module_initialization'] == 'fresh' and spec['use_selected_mask_supervision']
    spec['use_selected_mask_supervision'] = False
    path = destination / 'source/init_manifests' / (dataset + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

content = (root / 'native_joint_preflight.py').read_text(encoding='utf-8')
replacements = [
    ("parser.add_argument('--mode', required=True, choices=['extremal_support'])", "parser.add_argument('--dataset', required=True, choices=['nr3d', 'sr3d'])"),
    ("protocol = json.loads(Path(options.protocol).read_bytes())", "protocol = json.loads(Path(options.protocol).read_bytes())\nassert protocol['final_method_admitted'], 'Source preparation is not GPU admission'\nassert not protocol['use_selected_mask_supervision']"),
    ("'--native_init_spec', str(source / 'init_manifests' / (options.mode + '.json')),", "'--dataset', options.dataset, '--test_dataset', options.dataset,\n    '--native_init_spec', str(source / 'init_manifests' / (options.dataset + '.json')),"),
    ("assert args.dataset == ['scanrefer'] and args.test_dataset == 'scanrefer'", "assert args.dataset == [options.dataset] and args.test_dataset == options.dataset"),
    ("assert not args.joint_det and not args.detect_intermediate and not args.augment_det", "assert args.joint_det and args.detect_intermediate and args.butd_cls\nassert not args.butd and not args.butd_gt and not args.augment_det\nassert not args.debug and not args.eval and not args.eval_train"),
    ("assert len(train_loader.dataset) == 36665 and len(test_loader.dataset) == 9508\nassert len(train_loader) == 4583 and train_loader.drop_last", "assert train_loader.drop_last and train_loader.dataset.split == 'train'\nassert train_loader.dataset.augment and train_loader.dataset.joint_det\nassert test_loader.dataset.split == 'val' and not test_loader.dataset.augment\nassert len(train_loader) == len(train_loader.dataset) // args.batch_size\nassert set(anno['dataset'] for anno in train_loader.dataset.annos) == {options.dataset, 'scannet'}"),
    ("batches = list(itertools.islice(train_loader, 2))\nassert len(batches) == 2 and all(len(batch['utterances']) == 8 for batch in batches)\nassert all(set(batch['language_dataset']) == {'scanrefer'} for batch in batches)", """from torch.utils.data import DataLoader, Subset
referring = [i for i, anno in enumerate(train_loader.dataset.annos) if anno['dataset'] == options.dataset][:8]
detection = [i for i, anno in enumerate(train_loader.dataset.annos) if anno['dataset'] == 'scannet'][:8]
assert len(referring) == len(detection) == 8
if options.dataset == 'sr3d':
    referring[0] = next(i for i, anno in enumerate(train_loader.dataset.annos)
        if anno['dataset'] == 'sr3d' and anno['auxi_entity'] is not None and anno['anchor_ids'])
selected_indices = [index for pair in zip(referring, detection) for index in pair]
check_loader = DataLoader(Subset(train_loader.dataset, selected_indices), batch_size=8,
    shuffle=False, num_workers=0, drop_last=True)
batches = list(check_loader)
assert len(batches) == 2 and all(len(batch['utterances']) == 8 for batch in batches)
assert all(batch['language_dataset'] == [options.dataset] * 8 for batch in batches)
assert all(batch['sample_dataset'] == [options.dataset, 'scannet'] * 4 for batch in batches)
if options.dataset == 'sr3d':
    assert int(batches[0]['box_label_mask'][0].sum()) == 2
for batch in batches:
    assert batch['point_clouds'].shape == (8, 50000, 6)
    assert torch.equal(batch['all_detected_boxes'], batch['all_bboxes'])
    assert torch.equal(batch['all_detected_bbox_label_mask'], batch['all_bbox_label_mask'])
    assert bool((batch['box_label_mask'][1::2].sum(-1) > 1).all())"""),
    ("    if 'backbone_net' in name:", "    if '.source_query_read.' in name:\n        return 'G_reader'\n    if 'backbone_net' in name:"),
    ("return 'original_core_and_G_reader'", "return 'original_core'"),
    ("assert len(model.state_dict()) == 1295", "assert len(model.state_dict()) == 1295\nassert not model.use_selected_mask_supervision"),
    ("assert not initial['native_joint_training']", "assert not initial['native_joint_training']\n    assert not model.module.native_training_architecture['scanrefer_core_or_module_state_loaded']\n    assert model.module.native_training_architecture['dataset'] == options.dataset"),
    ("original_compute_loss = tester._compute_loss", """original_compute_loss = tester._compute_loss
original_set_forward = set_criterion.forward
matching_records = []


def observed_set_forward(outputs, targets):
    losses, indices = original_set_forward(outputs, targets)
    matching_records.append((indices, targets))
    return losses, indices


set_criterion.forward = observed_set_forward"""),
    ("    loss, end_points = original_compute_loss(end_points, native_criterion, native_set_criterion, native_args)\n    assert torch.isfinite(loss)", """    matching_records.clear()
    assert not end_points['use_selected_mask_supervision']
    loss, end_points = original_compute_loss(end_points, native_criterion, native_set_criterion, native_args)
    assert torch.isfinite(loss) and len(matching_records) == 7
    assert 'selected_query_mask_extra_rows' not in end_points
    assert 'selected_query_mask_extra_loss' not in end_points
    # Native prefix order is proposal_, last_, then the five intermediate heads.
    indices, targets = matching_records[1]
    from pvground_semantic_assignment import qualified_unmatched, semantic_assignment_correction
    selected = qualified_unmatched(end_points, end_points, indices)
    assert not bool(selected[1::2].any())
    for bid, (queries, target_slots) in enumerate(indices):
        assert not bool(selected[bid, queries].any())
        assert len(queries) == int(end_points['box_label_mask'][bid].sum())
        assert torch.equal(target_slots.sort().values,
            torch.arange(len(queries), device=target_slots.device))
        assert len(targets[bid]['boxes']) == len(queries)
    correction, assignment = semantic_assignment_correction(end_points, end_points,
        indices, native_set_criterion.eos_coef, native_args.num_decoder_layers)
    direct_gradient = torch.autograd.grad(correction,
        end_points['last_sem_cls_scores'], retain_graph=True)[0]
    assert bool((direct_gradient[~selected] == 0).all())
    assert int(end_points['g_reassigned_queries']) == assignment['reassigned_queries']
    protection = dict(selected_unmatched=int(selected.sum()),
        detection_rows_unchanged=True, all_original_matched_queries_unchanged=True,
        original_multi_GT_matching_complete=True,
        valid_GT_counts=[int(row.sum()) for row in end_points['box_label_mask']],
        matched_counts=[len(queries) for queries, _ in indices],
        nonselected_G_direct_logit_gradient_zero=True)"""),
    ("        selected_query_mask_extra_rows=int(end_points['selected_query_mask_extra_rows']),\n        selected_query_mask_extra_loss=float(end_points['selected_query_mask_extra_loss'].detach())))", "        selected_mask_supervision_disabled=True, multi_GT_protection=protection))"),
    ("step['original_core_and_G_reader']", "step['original_core']"),
    ("assert step['support_corrector'] > 0", "assert step['support_corrector'] > 0 and step['G_reader'] > 0"),
    ("('backbone', 'original_core_and_G_reader', 'support_corrector', 'span_output', 'span_internal')", "('backbone', 'original_core', 'G_reader', 'support_corrector', 'span_output', 'span_internal')"),
    ("options.mode", "'extremal_support'"),
    ("'ACTUAL_NATIVE_NORMAL_TRAINER_TWO_STEP_AND_FULL_RECOVERY_COMPLETE'", "'ACTUAL_NATIVE_REFERIT_C_OFF_TWO_MIXED_BATCHES_AND_FULL_RECOVERY_COMPLETE'"),
    ("Core, backbone, support positive each step; Span output and internal positive across two steps", "Core, backbone, G reader and support positive each step; Span output and internal positive across two steps"),
    ("span_initialization='retained_pretrained_checkpoint'", "span_initialization='fresh_corresponding_author_core'"),
    ("pretrained_span_initial_mix_formula_verified=True", "fresh_span_initial_mix_formula_verified=True"),
    ("span_initialization_identity=json.loads(Path(args.native_init_spec).read_bytes())['span_checkpoint_sha256']", "span_initialization_identity=json.loads(Path(args.native_init_spec).read_bytes())['official_checkpoint_sha256']"),
    ("    real_training_batches=2, native_train_one_epoch_called=True,", "    real_training_batches=2, native_train_one_epoch_called=True,\n    dataset=options.dataset, referencing_rows=8, joint_detection_rows=8,\n    full_native_dataset_counts_observed=True, full_epoch_completed=False,\n    selected_annotation_indices=selected_indices, debug_used=False,\n    actual_normal_dataset_loading_and_augmentation_used=True,"),
]
for old, new in replacements:
    assert old in content, old
    content = content.replace(old, new)
content = content.replace('Two real batches through the normal PV trainer',
    'Two real mixed ReferIt/detection batches through the normal PV trainer')
content = content.replace("import itertools\n", '')
ast.parse(content, feature_version=(3, 7))
(destination / 'native_referit_preflight.py').write_text(content, encoding='utf-8')

arguments = ['--model', 'PVGround', '--data_root', '/root/autodl-tmp/DATA_ROOT_mcln_meshsp/',
    '--use_color', '--use_soft_token_loss', '--use_contrastive_align', '--self_attend',
    '--butd_cls', '--joint_det', '--detect_intermediate', '--rng_seed', '2027',
    '--batch_size', '8', '--num_workers', '4', '--num_target', '256',
    '--num_decoder_layers', '6', '--num_encoder_layers', '3',
    '--self_position_embedding', 'loc_learned', '--lr', '1e-6', '--lr_backbone', '1e-6',
    '--lr_modules', '1e-5', '--weight_decay', '0.0005', '--clip_norm', '0.1',
    '--max_epoch', '3', '--start_epoch', '1', '--lr-scheduler', 'step',
    '--lr_decay_epochs', '280', '340', '--warmup-epoch', '-1']
protocol = dict(model_source='/root/autodl-tmp/pvground_referit_native_gpu_preflight_20261011/PV-Ground',
    common_arguments=arguments, final_method_admitted=False,
    use_selected_mask_supervision=False, source_review_completed=False,
    active_ScanRefer_controller_terminal_verified=False,
    single_A100_idle_and_current_resource_intake_verified=False,
    source_deployed=False, formal_accuracy=None, full_goal_complete=False)
(destination / 'PREPARATION_PROTOCOL.json').write_text(json.dumps(protocol, indent=2)+'\n', encoding='utf-8')
hashes = {path.relative_to(destination).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (destination / 'source').rglob('*') if path.is_file()}
(destination / 'SOURCE_HASHES.json').write_text(json.dumps(hashes, indent=2)+'\n', encoding='utf-8')
(destination / 'SOURCE_PREPARATION.json').write_text(json.dumps(dict(
    status='REFERIT_C_OFF_NATIVE_GPU_CHECK_SOURCE_PREPARED_NOT_ADMITTED',
    source_files_copied_unchanged=15, initialization_only_change='C disabled in provisional Nr/Sr manifests',
    check_script_sha256=hashlib.sha256((destination / 'native_referit_preflight.py').read_bytes()).hexdigest(),
    base_same_query_publication_section='20.376.143',
    real_execution_completed=False, source_deployed=False, GPU_training_admission=False,
    final_method_selected=False, formal_accuracy=None, full_goal_complete=False), indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(status='REFERIT_C_OFF_GPU_CHECK_SOURCE_PREPARED_NOT_EXECUTED',
    source_files=len(hashes), GPU_training_admission=False, formal_accuracy=None)))
