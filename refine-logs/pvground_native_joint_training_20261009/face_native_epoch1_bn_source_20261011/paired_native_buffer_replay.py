"""Read-only native PV replay; fixed parameters, two actual BN buffer sets."""
import argparse
import copy
import datetime
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
parser.add_argument('--source-manifest', required=True)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--max-batches', type=int, choices=(0, 1), required=True)
options = parser.parse_args()
protocol = json.loads(Path(options.protocol).read_bytes())
manifest = json.loads(Path(options.source_manifest).read_bytes())
source = Path(protocol['model_source'])
assert source == Path('/root/autodl-tmp/pvground_face_native_gpu_v2_20261011/PV-Ground')
assert protocol['face_residual_mode'] == 'source_conditioned'
assert manifest['model_source'] == str(source)
for identity in manifest['source_files'].values():
    assert hashlib.sha256(Path(identity['path']).read_bytes()).hexdigest() == identity['sha256']
checkpoint = Path(options.checkpoint)
output = Path(options.output)
assert checkpoint.is_file() and not output.exists()
checkpoint_digest = hashlib.sha256()
with checkpoint.open('rb') as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        checkpoint_digest.update(block)
output.mkdir(parents=True)
os.chdir(str(source))
sys.path.insert(0, str(source))

import numpy as np
import torch
import torch.distributed as dist
from batchnorm_buffer_intervention import capture_batchnorm_buffers, install_batchnorm_buffers
from main_utils import parse_option
from train_dist_mod import TrainTester
from pcdet.config import cfg_from_yaml_file, cfg as model_cfg
from src.grounding_evaluator import GroundingEvaluator
from models.losses import _iou3d_par, box_cxcyczwhd_to_xyzxyz
from native_root_bbs import native_root_bbs

# No checkpoint argument here: reconstruct the protected official/G/A/B E0
# before loading the trained terminal into the SAME model.
sys.argv = [str(source / 'train_dist_mod.py')] + protocol['common_arguments'] + [
    '--native_init_spec', protocol['native_init_spec'], '--log_dir', str(output / 'logs'),
    '--exp', 'bn_buffer_diagnostic', '--print_freq', '128', '--eval']
args = parse_option()
assert args.dataset == ['scanrefer'] and args.test_dataset == 'scanrefer'
assert args.rng_seed == 2027 and args.batch_size == 8 and args.eval
assert not args.checkpoint_path and not args.butd_cls
assert not args.joint_det and not args.detect_intermediate and not args.augment_det
random.seed(2027)
np.random.seed(2027)
torch.manual_seed(2027)
torch.cuda.manual_seed_all(2027)
torch.cuda.set_device(0)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
dist.init_process_group(backend='nccl', init_method='env://',
    timeout=datetime.timedelta(seconds=5400))
assert dist.get_world_size() == 1
cfg_from_yaml_file('wandb_config.yaml', model_cfg)
tester = TrainTester(args, model_cfg)
train_loader, test_loader = tester.get_loaders(args)
assert train_loader is None and len(test_loader.dataset) == 9508 and len(test_loader) == 1189
assert list(test_loader.sampler) == list(range(9508))
model = tester.get_model(args).cuda().eval()
assert len(model.state_dict()) == 1301
initial_buffers = capture_batchnorm_buffers(model)
assert initial_buffers
saved = torch.load(str(checkpoint), map_location='cpu')
assert saved['epoch'] == 3 and saved['architecture'] == model.native_training_architecture
assert all(name.startswith('module.') for name in saved['model'])
trained_state = {name[7:]: value for name, value in saved['model'].items()}
saved_epoch = saved['epoch']
del saved
model.load_state_dict(trained_state, strict=True)
assert all(torch.equal(value.detach().cpu(), trained_state[name])
           for name, value in model.state_dict().items())
trained_buffers = capture_batchnorm_buffers(model)
assert set(trained_buffers) == set(initial_buffers)
buffer_changes = sum(not torch.equal(trained_buffers[key], initial_buffers[key])
                     for key in trained_buffers)
criterion, set_criterion = tester.get_criterion(args)
branches = ('trained_buffers', 'initial_buffers')
evaluators = {name: GroundingEvaluator(only_root=True, thresholds=[.25, .5],
    topks=[1, 5, 10], prefixes=['last_'], filter_non_gt_boxes=args.butd_cls,
    logger=tester.logger, model=args.model) for name in branches}
stats = {name: {} for name in branches}
rows = []
started = time.monotonic()
torch.cuda.reset_peak_memory_stats()


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def row_evidence(predictions):
    scores = native_root_bbs(predictions['last_sem_cls_scores'], predictions)
    selected = scores.argsort(1, True)[:, 0]
    result, all_ious = [], []
    for bid, selected_query in enumerate(selected.tolist()):
        assert bool(predictions['box_label_mask'][bid, 0])
        truth = torch.cat([predictions['center_label'][bid, 0, :3],
                           predictions['size_gts'][bid, 0]])[None]
        boxes = torch.cat([predictions['last_center'][bid], predictions['last_pred_size'][bid]], -1)
        ious, _ = _iou3d_par(box_cxcyczwhd_to_xyzxyz(truth), box_cxcyczwhd_to_xyzxyz(boxes))
        assert ious.shape == (1, 256) and torch.isfinite(ious).all()
        own = predictions['sp_last_pred_masks'][bid][selected_query]
        text = predictions['last_pred_masks'][bid][0, selected_query]
        alpha = predictions['adaptive_weights'][bid]
        fused = alpha * text + (1 - alpha) * own
        assert own.ndim == fused.ndim == 1
        slots = predictions['superpoints'][bid].long()
        truth_mask = predictions['gt_masks'][bid, 0].bool()
        mask_ious = []
        for logits in (own, fused):
            point_mask = logits[slots] > 0
            union = (point_mask | truth_mask).sum()
            assert int(union) > 0
            mask_ious.append(float((point_mask & truth_mask).sum().float() / union))
        result.append(dict(selected_query=selected_query,
            selected_box=boxes[selected_query].cpu().tolist(),
            selected_iou=float(ious[0, selected_query]), full256_max_iou=float(ious.max()),
            selected_own_mask_iou=mask_ious[0], selected_fused_mask_iou=mask_ious[1]))
        all_ious.append(ious[0].cpu())
    return result, all_ious


iterator = test_loader if options.max_batches == 0 else itertools.islice(test_loader, 1)
with (output / 'paired_rows.jsonl').open('w', encoding='utf-8') as stream, torch.no_grad():
    for batch_index, cached in enumerate(iterator):
        assert set(cached['language_dataset']) == {'scanrefer'}
        common_rng = rng_state()
        branch_rows, branch_ious = {}, {}
        for branch in branches:
            install_batchnorm_buffers(model, trained_buffers if branch == branches[0] else initial_buffers)
            restore_rng(common_rng)
            stats[branch], predictions = tester._main_eval_branch(batch_index, copy.deepcopy(cached),
                test_loader, model, stats[branch], criterion, set_criterion, args)
            assert not model.training and not predictions['native_joint_training']
            # Retain the primary branch's stream for the next batch; the workers
            # generated cached inputs only once, outside this two-branch loop.
            if branch == branches[0]:
                primary_next_rng = rng_state()
            evaluators[branch].evaluate(predictions, 'last_')
            branch_rows[branch], branch_ious[branch] = row_evidence(predictions)
            del predictions
        restore_rng(primary_next_rng)
        assert len(branch_rows[branches[0]]) == len(branch_rows[branches[1]]) == len(cached['utterances'])
        for bid in range(len(cached['utterances'])):
            row = dict(index=len(rows), scan_id=cached['scan_ids'][bid],
                target_id=int(cached['target_id'][bid]), utterance=cached['utterances'][bid],
                trained_buffers=branch_rows[branches[0]][bid],
                initial_buffers=branch_rows[branches[1]][bid],
                trained_query_under_initial_buffers_iou=float(branch_ious[branches[1]][bid][branch_rows[branches[0]][bid]['selected_query']]),
                initial_query_under_trained_buffers_iou=float(branch_ious[branches[0]][bid][branch_rows[branches[1]][bid]['selected_query']]))
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        stream.flush()
        if (batch_index + 1) % 128 == 0:
            print(json.dumps(dict(completed_batches=batch_index + 1, rows=len(rows),
                elapsed_seconds=time.monotonic() - started)), flush=True)

install_batchnorm_buffers(model, trained_buffers)
assert all(torch.equal(value.detach().cpu(), trained_state[name])
           for name, value in model.state_dict().items())
expected_rows = 9508 if options.max_batches == 0 else 8
assert len(rows) == expected_rows
metrics, transitions = {}, {}
for branch, evaluator in evaluators.items():
    evaluator.synchronize_between_processes()
    metrics[branch] = dict(rows=expected_rows, primary_score='last/bbs')
    for threshold, suffix in ((.25, '025'), (.5, '050')):
        key = ('last_', threshold, 1, 'bbs')
        assert int(evaluator.gts[key]) == expected_rows
        hits = sum(row[branch]['selected_iou'] > threshold for row in rows)
        assert int(evaluator.dets[key]) == hits
        metrics[branch]['hits' + suffix] = hits
        metrics[branch]['acc' + suffix + '_percent'] = 100.0 * hits / expected_rows
        metrics[branch]['full256_hits' + suffix] = sum(row[branch]['full256_max_iou'] > threshold for row in rows)
    metrics[branch]['native_mask_miou_pos'] = float(evaluator.dets['mask_pos'] / evaluator.gts['mask_sem'])
    metrics[branch]['native_mask_miou_sem'] = float(evaluator.dets['mask_sem'] / evaluator.gts['mask_sem'])
    metrics[branch]['native_semantic_alignment_mask_hits025'] = int(evaluator.dets['overall_mask'])
    metrics[branch]['native_semantic_alignment_mask_hits050'] = int(evaluator.dets['overall50_mask'])
for threshold, suffix in ((.25, '025'), (.5, '050')):
    repaired = sum(row['trained_buffers']['selected_iou'] <= threshold < row['initial_buffers']['selected_iou'] for row in rows)
    broken = sum(row['initial_buffers']['selected_iou'] <= threshold < row['trained_buffers']['selected_iou'] for row in rows)
    assert repaired - broken == metrics['initial_buffers']['hits' + suffix] - metrics['trained_buffers']['hits' + suffix]
    transitions[suffix] = dict(repaired=repaired, broken=broken, net=repaired - broken)
result = dict(status='NATIVE_PAIRED_BUFFER_REPLAY_COMPLETE',
    scope='FULL_9508_DIAGNOSTIC_REPLAY' if options.max_batches == 0 else 'REAL_FIRST_BATCH_8_SANITY',
    completed_cst=datetime.datetime.now().astimezone().isoformat(), saved_epoch=saved_epoch,
    checkpoint=dict(path=str(checkpoint), bytes=checkpoint.stat().st_size, sha256=checkpoint_digest.hexdigest()),
    model_source=str(source), architecture=model.native_training_architecture,
    metrics=metrics, transitions=transitions,
    selected_query_changes=sum(row['trained_buffers']['selected_query'] != row['initial_buffers']['selected_query'] for row in rows),
    buffer_count=len(initial_buffers), changed_buffer_count=buffer_changes,
    full_trained_state_restored_exactly=True, all_non_bn_states_unchanged=True,
    paired_input_read_once=True, same_instance_and_trained_parameters=True,
    branch_rng_restored=True, optimizer_updates=0, written_weights=0,
    elapsed_seconds=time.monotonic() - started,
    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
    official_metrics_replaced=False, best_promoted=False, normal_training_fix_validated=False,
    limitations=['Independent validation replay may use different sampled points from the historical E3 evaluation.',
                 'Any change establishes a fixed-checkpoint forward effect, not a successful normal-training recipe.'])
(output / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
dist.destroy_process_group()
