"""Two real batches through the normal PV trainer, then complete recovery.

Engineering preflight only: no formal accuracy, no carried-over fitted state.
Run only after the original serial GPU experiment has closed and admission.
"""
import argparse
import copy
import datetime
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--protocol', required=True)
parser.add_argument('--mode', required=True, choices=['extremal_support'])
parser.add_argument('--output', required=True)
parser.add_argument('--initial_checkpoint', required=True)
options = parser.parse_args()
protocol = json.loads(Path(options.protocol).read_bytes())
assert protocol['serial_gpu_preflight_admitted'], 'Source preparation is not GPU admission'
assert protocol['matcher_mask_source'] in ('text', 'query')
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
    '--native_init_spec', protocol['native_init_spec'],
    '--matcher_mask_source', protocol['matcher_mask_source'],
    '--log_dir', str(output), '--exp', 'engineering_two_batch_preflight', '--print_freq', '1']
args = parse_option()
assert args.dataset == ['scanrefer'] and args.test_dataset == 'scanrefer'
assert args.batch_size == 8 and args.rng_seed == 2027 and args.self_attend
assert not args.joint_det and not args.detect_intermediate and not args.augment_det
assert not args.frozen and not args.small_lr and not args.checkpoint_path
assert args.lr == args.lr_backbone == 1e-6 and args.lr_modules == 1e-5
assert args.matcher_mask_source == protocol['matcher_mask_source']
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
assert len(train_loader.dataset) == 36665 and len(test_loader.dataset) == 9508
assert len(train_loader) == 4583 and train_loader.drop_last
train_loader.sampler.set_epoch(1)
train_loader.generator.manual_seed(args.rng_seed + 1)
batches = list(itertools.islice(train_loader, 2))
assert len(batches) == 2 and all(len(batch['utterances']) == 8 for batch in batches)
assert all(set(batch['language_dataset']) == {'scanrefer'} for batch in batches)
criterion, set_criterion = tester.get_criterion(args)
assert set_criterion.matcher.mask_source == args.matcher_mask_source
assert (set_criterion.matcher.cost_class, set_criterion.matcher.cost_bbox,
        set_criterion.matcher.cost_giou, set_criterion.matcher.cost_masks) == (1, 0, 2, .0002)
model = tester.get_model(args)
assert len(model.state_dict()) == 1295
assert not model.use_selected_mask_supervision and model.use_g_supervision
spec = json.loads(Path(args.native_init_spec).read_bytes())
assert not spec['use_selected_mask_supervision']
parent = torch.load(options.initial_checkpoint, map_location='cpu')
assert parent['epoch'] == 0 and parent['retained_metrics']['hits025'] == 5677 and parent['retained_metrics']['hits050'] == 4920
assert parent['architecture']['use_selected_mask_supervision']
parent_architecture = dict(parent['architecture'])
parent_architecture['use_selected_mask_supervision'] = False
assert parent_architecture == model.native_training_architecture
initial_state = model.state_dict()
assert {'module.' + name for name in initial_state} == set(parent['model'])
assert all(torch.equal(value, parent['model']['module.' + name]) for name, value in initial_state.items())
del parent, initial_state
initial_model_state_exactly_matches_normal_E0 = True
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
    if 'backbone_net' in name:
        return 'backbone'
    assert 'text_encoder' not in name
    return 'original_core_and_G_reader'


before = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
          if parameter.requires_grad}
assert before and not any('text_encoder' in name for name in before)
model.eval()
with torch.no_grad():
    initial_batch = tester._to_gpu(copy.deepcopy(batches[0]))
    initial = model(tester._get_inputs(initial_batch, training=False))
    assert not initial['native_joint_training']
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
from models.losses import HungarianMatcher, box_cxcyczwhd_to_xyzxyz, _iou3d_par
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

original_compute_loss = tester._compute_loss


def observed_loss(end_points, native_criterion, native_set_criterion, native_args):
    current_step[0] += 1
    assert current_step[0] in (0, 1) and end_points['native_joint_training']
    assert not end_points['use_selected_mask_supervision']
    assert end_points['last_center'].requires_grad and end_points['last_pred_size'].requires_grad
    assert end_points['last_sem_cls_scores'].requires_grad
    loss, end_points = original_compute_loss(end_points, native_criterion, native_set_criterion, native_args)
    assert torch.isfinite(loss)
    assert 'selected_query_mask_extra_rows' not in end_points
    assert 'selected_query_mask_extra_loss' not in end_points
    raw_gate = end_points['span_raw_axis_gate'].detach()
    loss_records.append(dict(step=current_step[0] + 1, total_loss=float(loss.detach()),
        raw_gate_min=float(raw_gate.min()), raw_gate_max=float(raw_gate.max()),
        raw_gate_negative_count=int((raw_gate < 0).sum()), raw_gate_zero_count=int((raw_gate == 0).sum()),
        raw_gate_above_one_count=int((raw_gate > 1).sum()), raw_gate_elements=raw_gate.numel(),
        g_reassigned_queries=int(end_points['g_reassigned_queries']),
        selected_mask_supervision_disabled=True))
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
assert len(matching_records) == 2
set_criterion.matcher.forward = original_matching
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
    source_mode=options.mode, gradient_norms=gradient_norms, updates=updates, losses=loss_records,
    gradient_acceptance='Core, backbone, support positive each step; Span output and internal positive across two steps',
    actual_updates=2, formal_accuracy=None, normal_epoch_training_started=False)
(output / 'NATIVE_M0_TRAINING_DIAGNOSTIC.json').write_text(json.dumps(diagnostic, indent=2) + '\n')
for step in gradient_norms:
    assert step['backbone'] > 0 and step['original_core_and_G_reader'] > 0
    assert step['support_corrector'] > 0
assert sum(step['span_output'] for step in gradient_norms) > 0
assert sum(step['span_internal'] for step in gradient_norms) > 0
assert all(updates[name]['changed_elements'] > 0 for name in
    ('backbone', 'original_core_and_G_reader', 'support_corrector', 'span_output', 'span_internal'))
del before
peak = dict(allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved())
args._retained_native_metrics = None  # Engineering-only checkpoint, no fabricated formal score.
save_checkpoint(args, 0, model, optimizer, scheduler)
checkpoint = Path(args.log_dir) / 'best.pth'
saved = torch.load(str(checkpoint), map_location='cpu')
assert saved['retained_metrics'] is None and saved['architecture'] == model.module.native_training_architecture
assert saved['config'].matcher_mask_source == args.matcher_mask_source
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
restored_criterion, restored_set_criterion = tester.get_criterion(args)
assert restored_set_criterion.matcher.mask_source == saved['config'].matcher_mask_source
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
receipt = dict(status='ACTUAL_NATIVE_MATCHER_TWO_STEP_AND_FULL_RECOVERY_COMPLETE',
    matcher_mask_source=args.matcher_mask_source,
    matcher_source_saved_and_reconstructed=True, native_last_matching_checks=matching_records,
    selected_mask_supervision_disabled=True,
    initial_model_state_exactly_matches_normal_E0=initial_model_state_exactly_matches_normal_E0,
    source_mode=options.mode, time_cst=datetime.datetime.now().astimezone().isoformat(),
    real_train_dataset_rows=len(train_loader.dataset), real_formal_dataset_rows=len(test_loader.dataset),
    normal_epoch_updates=len(train_loader), actual_updates=2, actual_input_rows=16,
    real_training_batches=2, native_train_one_epoch_called=True,
    original_trainable_core_jointly_updated=True, original_roberta_frozen=True,
    full_state_tensors=1295, gradient_norms=gradient_norms, updates=updates, losses=loss_records,
    gradient_acceptance=diagnostic['gradient_acceptance'],
    pretrained_span_initial_mix_formula_verified=True, full_recovery_exact=True,
    span_initialization='retained_pretrained_checkpoint',
    span_initialization_identity=json.loads(Path(args.native_init_spec).read_bytes())['span_checkpoint_sha256'],
    checkpoint_sha256=checkpoint_sha, checkpoint_bytes=checkpoint_bytes,
    saved_retained_metrics=None, restored_optimizer_states=len(restored_adam['state']),
    step_seconds=seconds, peak_memory_bytes=peak, formal_accuracy=None,
    actual_imports={name: str(sys.modules[name].__file__) for name in
        ('train_dist_mod', 'main_utils', 'native_model_initialization', 'models.pv_ground', 'models.losses')},
    preflight_weights_used_for_formal_training=False, actual_normal_epoch_training_started=False)
(output / 'NATIVE_M0_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(dict(status=receipt['status'], source_mode=options.mode,
    actual_updates=2, checkpoint_bytes=checkpoint_bytes, step_seconds=seconds, formal_accuracy=None)), flush=True)
dist.destroy_process_group()
