"""Measure repeated inference on one original validation batch, without training."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models.cs_mcln_modules import _member_mean
from models.losses import _iou3d_par, box_cxcyczwhd_to_xyzxyz
from models.source_choice_adapter import compute_default_source_scores
from scripts.train_cs_mcln_scanrefer import (
    batch_loss, experiment_args, load_exact_e71, set_seed,
)
from src.grounding_evaluator import GroundingEvaluator
from train_dist_mod import TrainTester


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ious_to_root(boxes, root):
    return _iou3d_par(
        box_cxcyczwhd_to_xyzxyz(root.unsqueeze(0)),
        box_cxcyczwhd_to_xyzxyz(boxes),
    )[0][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--row-id', type=int, default=2449)
    opt = parser.parse_args()
    reference = json.loads(opt.reference.read_text())
    assert reference['sample_count'] == 9508
    assert reference['checkpoint_sha256'] == file_sha256(opt.checkpoint)
    set_seed(2027)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    parent = torch.load(str(opt.parent), map_location='cpu')
    config = experiment_args(parent['config'], 'cs', opt.data_root)
    model = TrainTester.get_model(config)
    initial_load = load_exact_e71(model, parent['model'], 'cs')
    del parent
    saved = torch.load(str(opt.checkpoint), map_location='cpu')
    assert saved['initial_load'] == initial_load and saved['epoch'] == 15
    batch_size = int(saved['batch_size'])
    assert batch_size == reference['batch_size'] == 12
    model.load_state_dict(saved['model'], strict=True)
    del saved
    model = model.cuda().eval()
    parameters = list(model.parameters())
    original_grad_flags = [parameter.requires_grad for parameter in parameters]
    model.requires_grad_(False)
    criterion, set_criterion = TrainTester.get_criterion(config)
    original_eval = config.eval
    config.eval = True
    train, validation = TrainTester.get_datasets(config)
    config.eval = original_eval
    assert train is None and len(validation) == 9508
    first = opt.row_id // batch_size * batch_size
    row_ids = list(range(first, first + batch_size))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(validation, row_ids),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    raw = next(iter(loader))
    evaluator = GroundingEvaluator(
        only_root=True, thresholds=[0.25, 0.5], topks=[1],
        prefixes=['last_'], filter_non_gt_boxes=False, model='MCLN',
        eval_use_selector_choice_scores=False,
    )
    tensor_names = ('last_center', 'last_pred_size', 'last_sem_cls_scores')
    recorded = []
    base = None
    with torch.no_grad():
        for repeat in range(4):
            if repeat == 3:
                for parameter, flag in zip(parameters, original_grad_flags):
                    parameter.requires_grad_(flag)
            loss, outputs = batch_loss(model, raw, criterion, set_criterion, config, False)
            assert bool(torch.isfinite(loss))
            outputs['last_pred_size'] = outputs['last_pred_size'].clamp(min=1e-6)
            tensors = {name: outputs[name].detach().cpu().clone() for name in tensor_names}
            if base is None:
                base = tensors
            delta = {name: float((tensors[name] - base[name]).abs().max())
                     for name in tensor_names}
            scores = compute_default_source_scores(outputs, outputs)
            chosen = evaluator._position_top_indices(
                scores, torch.ones_like(scores, dtype=torch.bool),
                'default_query_axis', 256,
            )[:, 0]
            rows = []
            for bid, row_id in enumerate(row_ids):
                root = torch.cat((outputs['center_label'][bid, 0], outputs['size_gts'][bid, 0]))
                boxes = torch.cat((outputs['last_center'][bid], outputs['last_pred_size'][bid]), dim=-1)
                query_index = int(chosen[bid])
                rows.append({'row_id': row_id, 'query_index': query_index,
                             'iou': float(ious_to_root(boxes, root)[query_index]),
                             'v5_iou': reference['rows'][row_id]['post_selected_iou'],
                             'v5_query_index': reference['rows'][row_id]['query_index']})
            recorded.append({'repeat': repeat, 'parameter_flags': (
                'frozen' if repeat < 3 else 'original_training_parameter_flags'),
                'delta_from_first': delta, 'rows': rows})
            print('REPLAY_PASS', json.dumps(recorded[-1]), flush=True)
            del outputs, tensors
        bid = opt.row_id - first
        values = raw['point_clouds'][bid].cuda()
        member_ids = raw['superpoints'][bid].cuda().long()
        count = int(member_ids.max()) + 1
        mean, numbers = _member_mean(values, member_ids, count)
        reference_mean = mean.detach().cpu()
        reference_numbers = numbers.detach().cpu()
        member_deltas = []
        for repeat in range(16):
            mean, numbers = _member_mean(values, member_ids, count)
            assert torch.equal(numbers.detach().cpu(), reference_numbers)
            member_deltas.append(float((mean.detach().cpu() - reference_mean).abs().max()))
    assert all(parameter.grad is None for parameter in model.parameters())
    result = {'schema': 'cs-mcln-replay-stability-v1', 'optimizer_steps': 0,
              'checkpoint_sha256': reference['checkpoint_sha256'],
              'reference_sha256': file_sha256(opt.reference), 'batch_size': batch_size,
              'row_ids': row_ids, 'diagnostic_row_selection': (
                  'original batch containing the v5 selected IoU closest to 0.25; '
                  'runtime diagnosis only, not a training or validation selection rule'),
              'passes': recorded, 'member_mean_max_abs_deltas': member_deltas,
              'repeated_forward_bitwise_equal': all(
                  value == 0 for item in recorded[1:3]
                  for value in item['delta_from_first'].values()),
              'note': 'This panel cannot identify the historically disputed row without historical row outputs.'}
    opt.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('STABILITY_RESULT', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
