"""Two real mixed ReferIt/detection batches through the normal PV trainer, then complete recovery.

Engineering preflight only: no formal accuracy, no carried-over fitted state.
Run only after the original serial GPU experiment has closed and admission.
"""
import argparse
import copy
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--protocol', required=True)
parser.add_argument('--dataset', required=True, choices=['nr3d', 'sr3d'])
parser.add_argument('--output', required=True)
options = parser.parse_args()
protocol = json.loads(Path(options.protocol).read_bytes())
assert protocol['final_method_admitted'], 'Source preparation is not GPU admission'
assert not protocol['use_selected_mask_supervision']
source = Path(protocol['model_source'])
output = Path(options.output)
assert source.is_dir() and not output.exists()
output.mkdir(parents=True)
os.chdir(str(source))
sys.path.insert(0, str(source))
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from main_utils import parse_option, save_checkpoint, load_checkpoint
from train_dist_mod import TrainTester
from pcdet.config import cfg_from_yaml_file, cfg as model_cfg
from utils import get_scheduler

sys.argv = [str(source / 'train_dist_mod.py')] + protocol['common_arguments'] + [
    '--dataset', options.dataset, '--test_dataset', options.dataset,
    '--native_init_spec', str(source / 'init_manifests' / (options.dataset + '.json')),
    '--log_dir', str(output), '--exp', 'engineering_two_batch_preflight', '--print_freq', '1']
args = parse_option()
assert args.dataset == [options.dataset] and args.test_dataset == options.dataset
assert args.batch_size == 8 and args.rng_seed == 2027 and args.self_attend
assert args.joint_det and args.detect_intermediate and args.butd_cls
assert not args.butd and not args.butd_gt and not args.augment_det
assert not args.debug and not args.eval and not args.eval_train
assert not args.frozen and not args.small_lr and not args.checkpoint_path
assert args.lr == args.lr_backbone == 1e-6 and args.lr_modules == 1e-5
random.seed(args.rng_seed)
np.random.seed(args.rng_seed)
torch.manual_seed(args.rng_seed)
torch.cuda.manual_seed_all(args.rng_seed)
torch.cuda.set_device(0)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
dist.init_process_group(backend='nccl', init_method='env://',
                        timeout=datetime.timedelta(seconds=5400))
cfg_from_yaml_file('wandb_config.yaml', model_cfg)
tester = TrainTester(args, model_cfg)
train_loader, test_loader = tester.get_loaders(args)
assert train_loader.drop_last and train_loader.dataset.split == 'train'
assert train_loader.dataset.augment and train_loader.dataset.joint_det
assert test_loader.dataset.split == 'val' and not test_loader.dataset.augment
assert len(train_loader) == len(train_loader.dataset) // args.batch_size
assert set(anno['dataset'] for anno in train_loader.dataset.annos) == {options.dataset, 'scannet'}
train_loader.sampler.set_epoch(1)
train_loader.generator.manual_seed(args.rng_seed + 1)
from torch.utils.data import DataLoader, Subset
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
    assert bool((batch['box_label_mask'][1::2].sum(-1) > 1).all())
criterion, set_criterion = tester.get_criterion(args)
model = tester.get_model(args)
assert len(model.state_dict()) == 1295
assert not model.use_selected_mask_supervision
assert not any(parameter.requires_grad for parameter in model.text_encoder.parameters())
optimizer = tester.get_optimizer(args, model)
scheduler = get_scheduler(optimizer, len(train_loader), args)
model = DistributedDataParallel(model.cuda(), device_ids=[0],
    broadcast_buffers=False, find_unused_parameters=True)


def group(name):
    assert name.startswith('module.')
    name = name[7:]
    if name.startswith('candidate_span_mixer.output.'):
        return 'span_output'
    if name.startswith('candidate_span_mixer.'):
        return 'span_internal'
    if name.startswith('candidate_support_corrector.'):
        return 'support_corrector'
    if '.source_query_read.' in name:
        return 'G_reader'
    if 'backbone_net' in name:
        return 'backbone'
    assert 'text_encoder' not in name
    return 'original_core'


before = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
          if parameter.requires_grad}
assert before and not any('text_encoder' in name for name in before)
model.eval()
with torch.no_grad():
    initial_batch = tester._to_gpu(copy.deepcopy(batches[0]))
    initial = model(tester._get_inputs(initial_batch, training=False))
    assert not initial['native_joint_training']
    assert not model.module.native_training_architecture['scanrefer_core_or_module_state_loaded']
    assert model.module.native_training_architecture['dataset'] == options.dataset
    assert initial['last_center'].shape == initial['last_pred_size'].shape == (8, 256, 3)
    gate = initial['span_axis_gate']
    assert gate.shape == (8, 256, 3) and torch.isfinite(gate).all()
    assert ((gate >= 0) & (gate <= 1)).all()
    expected_center = (1 - gate) * initial['span_mask_center'] + gate * initial['native_coarse_center']
    expected_size = (1 - gate) * initial['span_mask_size'] + gate * initial['native_coarse_size'].clamp_min(1e-6)
    assert torch.equal(initial['last_center'], expected_center)
    assert torch.equal(initial['last_pred_size'], expected_size)
    assert torch.isfinite(initial['last_center']).all() and (initial['last_pred_size'] > 0).all()
del initial, initial_batch
torch.cuda.empty_cache()

gradient = [dict(), dict()]
loss_records = []
current_step = [-1]
hooks = []
for name, parameter in model.named_parameters():
    if parameter.requires_grad:
        category = group(name)
        def observe(value, category=category):
            assert current_step[0] in (0, 1) and torch.isfinite(value).all()
            gradient[current_step[0]].setdefault(category, []).append(value.detach().square().sum())
        hooks.append(parameter.register_hook(observe))
original_compute_loss = tester._compute_loss
original_set_forward = set_criterion.forward
matching_records = []


def observed_set_forward(outputs, targets):
    losses, indices = original_set_forward(outputs, targets)
    matching_records.append((indices, targets))
    return losses, indices


set_criterion.forward = observed_set_forward


def observed_loss(end_points, native_criterion, native_set_criterion, native_args):
    current_step[0] += 1
    assert current_step[0] in (0, 1) and end_points['native_joint_training']
    assert end_points['last_center'].requires_grad and end_points['last_pred_size'].requires_grad
    assert end_points['last_sem_cls_scores'].requires_grad
    matching_records.clear()
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
        nonselected_G_direct_logit_gradient_zero=True)
    raw_gate = end_points['span_raw_axis_gate'].detach()
    loss_records.append(dict(step=current_step[0] + 1, total_loss=float(loss.detach()),
        raw_gate_min=float(raw_gate.min()), raw_gate_max=float(raw_gate.max()),
        raw_gate_negative_count=int((raw_gate < 0).sum()), raw_gate_zero_count=int((raw_gate == 0).sum()),
        raw_gate_above_one_count=int((raw_gate > 1).sum()), raw_gate_elements=raw_gate.numel(),
        g_reassigned_queries=int(end_points['g_reassigned_queries']),
        selected_mask_supervision_disabled=True, multi_GT_protection=protection))
    return loss, end_points


class TwoNativeBatches:
    def __len__(self):
        return 2

    def __iter__(self):
        return (copy.deepcopy(batch) for batch in batches)


tester._compute_loss = observed_loss
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
started = time.monotonic()
tester.train_one_epoch(1, TwoNativeBatches(), model, criterion, set_criterion,
                       optimizer, scheduler, args)
torch.cuda.synchronize()
seconds = time.monotonic() - started
assert current_step[0] == 1 and len(loss_records) == 2
for hook in hooks:
    hook.remove()
gradient_norms = [{key: float(torch.stack(values).sum().sqrt()) for key, values in step.items()}
                  for step in gradient]
updates = {}
for name, parameter in model.named_parameters():
    if name in before:
        delta = parameter.detach().cpu() - before[name]
        category = group(name)
        row = updates.setdefault(category, dict(changed_parameters=0, changed_elements=0, maximum_change=0.0))
        changed = int(torch.count_nonzero(delta))
        row['changed_parameters'] += int(changed > 0)
        row['changed_elements'] += changed
        row['maximum_change'] = max(row['maximum_change'], float(delta.abs().max()))
diagnostic = dict(status='TWO_ACTUAL_NATIVE_UPDATES_BEFORE_ACCEPTANCE',
    source_mode='extremal_support', gradient_norms=gradient_norms, updates=updates, losses=loss_records,
    gradient_acceptance='Core, backbone, G reader and support positive each step; Span output and internal positive across two steps',
    actual_updates=2, formal_accuracy=None, normal_epoch_training_started=False)
(output / 'NATIVE_M0_TRAINING_DIAGNOSTIC.json').write_text(json.dumps(diagnostic, indent=2) + '\n')
for step in gradient_norms:
    assert step['backbone'] > 0 and step['original_core'] > 0
    assert step['support_corrector'] > 0 and step['G_reader'] > 0
assert sum(step['span_output'] for step in gradient_norms) > 0
assert sum(step['span_internal'] for step in gradient_norms) > 0
assert all(updates[name]['changed_elements'] > 0 for name in
    ('backbone', 'original_core', 'G_reader', 'support_corrector', 'span_output', 'span_internal'))
del before
peak = dict(allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved())
args._retained_native_metrics = None  # Engineering-only checkpoint, no fabricated formal score.
save_checkpoint(args, 0, model, optimizer, scheduler)
checkpoint = Path(args.log_dir) / 'best.pth'
saved = torch.load(str(checkpoint), map_location='cpu')
assert saved['retained_metrics'] is None and saved['architecture'] == model.module.native_training_architecture
assert len(saved['model']) == 1295 and saved['optimizer']['state']
assert all(int(state['step']) == 2 for state in saved['optimizer']['state'].values())
checkpoint_bytes = checkpoint.stat().st_size
checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
del model, optimizer, scheduler
gc.collect()
torch.cuda.empty_cache()
args.checkpoint_path = str(checkpoint)
restored_model = tester.get_model(args)  # Architecture only, never reloads official/G/support weights.
restored_optimizer = tester.get_optimizer(args, restored_model)
restored_scheduler = get_scheduler(restored_optimizer, len(train_loader), args)
restored_model = DistributedDataParallel(restored_model.cuda(), device_ids=[0],
    broadcast_buffers=False, find_unused_parameters=True)
load_checkpoint(args, restored_model, restored_optimizer, restored_scheduler)
restored_state = restored_model.state_dict()
assert set(restored_state) == set(saved['model'])
assert all(torch.equal(value.cpu(), saved['model'][name]) for name, value in restored_state.items())
restored_adam = restored_optimizer.state_dict()
assert restored_adam['param_groups'] == saved['optimizer']['param_groups']
assert set(restored_adam['state']) == set(saved['optimizer']['state'])
for index, state in restored_adam['state'].items():
    expected = saved['optimizer']['state'][index]
    assert set(state) == set(expected)
    for name, value in state.items():
        if torch.is_tensor(value):
            assert torch.equal(value.cpu(), expected[name])
        else:
            assert value == expected[name]
assert restored_scheduler.state_dict() == saved['scheduler'] and args.start_epoch == 1
assert random.getstate() == saved['rng']['python']
numpy_rng = np.random.get_state()
assert numpy_rng[0] == saved['rng']['numpy'][0] and numpy_rng[2:] == saved['rng']['numpy'][2:]
assert np.array_equal(numpy_rng[1], saved['rng']['numpy'][1])
assert torch.equal(torch.get_rng_state(), saved['rng']['torch'])
cuda_rng = torch.cuda.get_rng_state_all()
assert len(cuda_rng) == len(saved['rng']['cuda'])
assert all(torch.equal(value, expected) for value, expected in zip(cuda_rng, saved['rng']['cuda']))
receipt = dict(status='ACTUAL_NATIVE_REFERIT_C_OFF_TWO_MIXED_BATCHES_AND_FULL_RECOVERY_COMPLETE',
    source_mode='extremal_support', time_cst=datetime.datetime.now().astimezone().isoformat(),
    real_train_dataset_rows=len(train_loader.dataset), real_formal_dataset_rows=len(test_loader.dataset),
    normal_epoch_updates=len(train_loader), actual_updates=2, actual_input_rows=16,
    real_training_batches=2, native_train_one_epoch_called=True,
    dataset=options.dataset, referencing_rows=8, joint_detection_rows=8,
    full_native_dataset_counts_observed=True, full_epoch_completed=False,
    selected_annotation_indices=selected_indices, debug_used=False,
    actual_normal_dataset_loading_and_augmentation_used=True,
    original_trainable_core_jointly_updated=True, original_roberta_frozen=True,
    full_state_tensors=1295, gradient_norms=gradient_norms, updates=updates, losses=loss_records,
    gradient_acceptance=diagnostic['gradient_acceptance'],
    fresh_span_initial_mix_formula_verified=True, full_recovery_exact=True,
    span_initialization='fresh_corresponding_author_core',
    span_initialization_identity=json.loads(Path(args.native_init_spec).read_bytes())['official_checkpoint_sha256'],
    checkpoint_sha256=checkpoint_sha, checkpoint_bytes=checkpoint_bytes,
    saved_retained_metrics=None, restored_optimizer_states=len(restored_adam['state']),
    step_seconds=seconds, peak_memory_bytes=peak, formal_accuracy=None,
    actual_imports={name: str(sys.modules[name].__file__) for name in
        ('train_dist_mod', 'main_utils', 'native_model_initialization', 'models.pv_ground', 'models.losses')},
    preflight_weights_used_for_formal_training=False, actual_normal_epoch_training_started=False)
(output / 'NATIVE_M0_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(dict(status=receipt['status'], source_mode='extremal_support',
    actual_updates=2, checkpoint_bytes=checkpoint_bytes, step_seconds=seconds, formal_accuracy=None)), flush=True)
dist.destroy_process_group()
