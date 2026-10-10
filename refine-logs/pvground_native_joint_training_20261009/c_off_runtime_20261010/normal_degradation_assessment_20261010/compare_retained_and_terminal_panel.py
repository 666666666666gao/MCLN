"""Read-only first-64 validation panel, same cached inputs and two saved models."""
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
parser.add_argument('--checkpoints', required=True)
parser.add_argument('--output', required=True)
options = parser.parse_args()
protocol = json.loads(Path(options.protocol).read_bytes())
checkpoints = json.loads(Path(options.checkpoints).read_bytes())
assert checkpoints['status'] == 'NORMAL_BEST_AND_FIXED_E3_FULL_COLD_RECOVERY_COMPLETE'
assert [(row['name'], row['saved_epoch']) for row in checkpoints['checkpoints']] == [('best', 0), ('latest', 3)]
source = Path(protocol['model_source'])
output = Path(options.output)
assert source == Path('/root/autodl-tmp/pvground_native_joint_training_20261010/PV-Ground')
assert not output.exists()
output.mkdir()
os.chdir(str(source))
sys.path.insert(0, str(source))
import numpy as np
import torch
import torch.distributed as dist
from main_utils import parse_option
from train_dist_mod import TrainTester
from pcdet.config import cfg_from_yaml_file, cfg as model_cfg
from models.losses import _iou3d_par, box_cxcyczwhd_to_xyzxyz
from native_root_bbs import native_root_bbs

sys.argv = [str(source / 'train_dist_mod.py')] + protocol['common_arguments'] + [
    '--native_init_spec', str(source / 'init_manifests/extremal_support.json'),
    '--checkpoint_path', checkpoints['checkpoints'][0]['identity']['path'],
    '--log_dir', str(output), '--exp', 'read_only_validation_panel', '--eval']
args = parse_option()
assert args.dataset == ['scanrefer'] and args.test_dataset == 'scanrefer'
assert args.rng_seed == 2027 and args.batch_size == 8 and args.eval
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
cfg_from_yaml_file('wandb_config.yaml', model_cfg)
tester = TrainTester(args, model_cfg)
train_loader, test_loader = tester.get_loaders(args)
assert train_loader is None and len(test_loader.dataset) == 9508
assert list(itertools.islice(iter(test_loader.sampler), 64)) == list(range(64))
model = tester.get_model(args).cuda().eval()
assert len(model.state_dict()) == 1295
batches = list(itertools.islice(test_loader, 8))
assert len(batches) == 8 and all(len(batch['utterances']) == 8 for batch in batches)
assert all(set(batch['language_dataset']) == {'scanrefer'} for batch in batches)
assert all(batch['point_clouds'].shape == (8, 50000, 6) for batch in batches)
identities = []
for batch in batches:
    for row in range(8):
        identities.append(dict(panel_index=len(identities), scan_id=batch['scan_ids'][row],
            target_id=int(batch['target_id'][row]), utterance=batch['utterances'][row]))
assert len(identities) == 64
state = dict(python=random.getstate(), numpy=np.random.get_state(),
             torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())
results = []
started = time.monotonic()
torch.cuda.reset_peak_memory_stats()


def box_iou(center, size, truth):
    box = torch.cat([center, size], -1)
    iou, _ = _iou3d_par(box_cxcyczwhd_to_xyzxyz(truth), box_cxcyczwhd_to_xyzxyz(box))
    assert iou.shape == (1, 256) and torch.isfinite(iou).all()
    return box, iou.squeeze(0)


for checkpoint in checkpoints['checkpoints']:
    identity = checkpoint['identity']
    path = Path(identity['path'])
    assert path.stat().st_size == identity['bytes']
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    assert digest.hexdigest() == identity['sha256']
    saved = torch.load(str(path), map_location='cpu')
    assert saved['epoch'] == checkpoint['saved_epoch'] and saved['architecture'] == model.native_training_architecture
    assert all(name.startswith('module.') for name in saved['model'])
    model.load_state_dict({name[7:]: value for name, value in saved['model'].items()}, strict=True)
    del saved
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])
    arrays = {name: [] for name in ('native_box', 'reference_box', 'final_box',
        'native_iou', 'reference_iou', 'final_iou', 'own_mask_iou', 'fused_mask_iou',
        'bbs_score', 'axis_gate', 'raw_axis_gate', 'reference_valid')}
    rows = []
    with torch.no_grad():
        for batch_index, cached in enumerate(batches):
            batch = tester._to_gpu(copy.deepcopy(cached))
            predictions = model(tester._get_inputs(batch, training=False))
            assert not model.training and not predictions['native_joint_training']
            assert predictions['last_center'].shape == predictions['last_pred_size'].shape == (8, 256, 3)
            scores = native_root_bbs(predictions['last_sem_cls_scores'], batch)
            selected = scores.argsort(-1, descending=True)[:, 0]
            for bid in range(8):
                assert bool(batch['box_label_mask'][bid, 0])
                truth_box = torch.cat([batch['center_label'][bid, 0, :3], batch['size_gts'][bid, 0]])[None]
                native_box, native_iou = box_iou(predictions['native_coarse_center'][bid],
                    predictions['native_coarse_size'][bid].clamp_min(1e-6), truth_box)
                reference_box, reference_iou = box_iou(predictions['mask_reference_center'][bid],
                    predictions['mask_reference_size'][bid], truth_box)
                final_box, final_iou = box_iou(predictions['last_center'][bid],
                    predictions['last_pred_size'][bid], truth_box)
                own = predictions['sp_last_pred_masks'][bid]
                text = predictions['last_pred_masks'][bid][0]
                alpha = predictions['adaptive_weights'][bid]
                fused = alpha * text + (1-alpha) * own
                slots = predictions['superpoints'][bid].long()
                truth_mask = batch['gt_masks'][bid, 0].bool()[None]
                mask_ious = []
                for logits in (own, fused):
                    point_mask = logits[:, slots] > 0
                    intersection = (point_mask & truth_mask).sum(-1).float()
                    union = (point_mask | truth_mask).sum(-1).float()
                    assert (union > 0).all()
                    mask_ious.append(intersection / union)
                gate = predictions['span_axis_gate'][bid]
                raw_gate = predictions['span_raw_axis_gate'][bid]
                valid = predictions['mask_reference_valid'][bid]
                selected_query = int(selected[bid])
                values = (native_box, reference_box, final_box, native_iou, reference_iou,
                    final_iou, mask_ious[0], mask_ious[1], scores[bid], gate, raw_gate, valid)
                for name, value in zip(arrays, values):
                    arrays[name].append(value.detach().cpu().numpy())
                rows.append(dict(identities[batch_index*8+bid], checkpoint=checkpoint['name'],
                    selected_query=selected_query, selected_native_iou=float(native_iou[selected_query]),
                    selected_reference_iou=float(reference_iou[selected_query]),
                    selected_final_iou=float(final_iou[selected_query]),
                    selected_own_mask_iou=float(mask_ious[0][selected_query]),
                    selected_fused_mask_iou=float(mask_ious[1][selected_query]),
                    selected_axis_gate=gate[selected_query].tolist(),
                    selected_raw_axis_gate=raw_gate[selected_query].tolist(),
                    selected_reference_valid=bool(valid[selected_query]),
                    full256_native_max_iou=float(native_iou.max()),
                    full256_reference_max_iou=float(reference_iou.max()),
                    full256_final_max_iou=float(final_iou.max())))
            del predictions, batch, scores, selected
    assert len(rows) == 64 and all(len(value) == 64 for value in arrays.values())
    arrays = {name: np.stack(values) for name, values in arrays.items()}
    np.savez_compressed(str(output / (checkpoint['name'] + '_panel.npz')), **arrays)
    (output / (checkpoint['name'] + '_rows.json')).write_text(json.dumps(rows, indent=2) + '\n')
    results.append(dict(checkpoint=checkpoint['name'], saved_epoch=checkpoint['saved_epoch'],
        identity=identity, rows=64,
        selected_hits025=sum(row['selected_final_iou'] > .25 for row in rows),
        selected_hits050=sum(row['selected_final_iou'] > .5 for row in rows),
        selected_reference_hits050=sum(row['selected_reference_iou'] > .5 for row in rows),
        selected_native_hits050=sum(row['selected_native_iou'] > .5 for row in rows),
        selected_fused_mask_hits050=sum(row['selected_fused_mask_iou'] > .5 for row in rows),
        full256_final_qualified_rows=sum(row['full256_final_max_iou'] > .5 for row in rows),
        all_query_gate_mean=float(arrays['axis_gate'].mean()),
        all_query_gate_zero_fraction=float((arrays['axis_gate'] == 0).mean()),
        all_query_gate_one_fraction=float((arrays['axis_gate'] == 1).mean())))
receipt = dict(status='READ_ONLY_RETAINED_E0_VS_FIXED_E3_SAME64_VALIDATION_INPUTS_COMPLETE',
    time_cst=datetime.datetime.now().astimezone().isoformat(), results=results,
    primary_score='last/bbs', full256_retained=True, real_dataset_ground_truth=True,
    same_cached_inputs=True, same_inference_random_states=True,
    root_correspondence='Native ScanRefer valid GT slot0; Query slot equality is not physical instance identity',
    new_optimizer_steps=0, model_or_source_mutations=0, checkpoint_mutations=0,
    formal_full9508_result=False, fixed_panel_representative_of_full_validation=False,
    actual_neural_forwards=16, elapsed_seconds=time.monotonic()-started,
    peak_memory_bytes=dict(allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved()))
(output / 'PANEL_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
dist.destroy_process_group()
