"""Read-only CS-MCLN M3 geometry diagnosis on a fixed ScanRefer train panel."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models.losses import _iou3d_par, box_cxcyczwhd_to_xyzxyz
from models.source_choice_adapter import compute_default_source_scores
from scripts.train_cs_mcln_scanrefer import (
    batch_loss, experiment_args, load_exact_e71, set_seed,
)
from src.grounding_evaluator import GroundingEvaluator
from train_dist_mod import TrainTester


PANEL_SALT = 'CS-MCLN-SCANREFER-TRAIN-SCENES-20260925'


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


def outside_mass(xyz, mass, center, size):
    inside = ((xyz - center).abs() <= size.clamp_min(1e-6) / 2).all(dim=-1)
    return float(mass[~inside].sum() / mass.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--panel-scenes', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=4)
    opt = parser.parse_args()
    assert opt.panel_scenes > 0 and opt.batch_size > 0

    set_seed(2027)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    parent = torch.load(str(opt.parent), map_location='cpu')
    config = experiment_args(parent['config'], 'cs', opt.data_root)
    model = TrainTester.get_model(config)
    initial_load = load_exact_e71(model, parent['model'], 'cs')
    del parent
    saved = torch.load(str(opt.checkpoint), map_location='cpu')
    assert saved['arm'] == 'cs' and saved['initial_load'] == initial_load
    assert saved['metrics']['samples'] == 9508
    model.load_state_dict(saved['model'], strict=True)
    checkpoint_epoch = int(saved['epoch'])
    checkpoint_metrics = saved['metrics']
    del saved
    model = model.cuda().eval().requires_grad_(False)
    criterion, set_criterion = TrainTester.get_criterion(config)
    train, validation = TrainTester.get_datasets(config)
    assert len(train) == 48655 and len(validation) == 9508
    train.augment = False

    first_by_scene = {}
    for index, annotation in enumerate(train.annos):
        if annotation['dataset'] == 'scanrefer':
            first_by_scene.setdefault(annotation['scan_id'], index)
    scene_ids = sorted(first_by_scene, key=lambda scene: hashlib.sha256(
        (PANEL_SALT + '\0' + scene).encode('utf-8')
    ).hexdigest())[:opt.panel_scenes]
    assert len(scene_ids) == opt.panel_scenes
    panel_ids = [first_by_scene[scene] for scene in scene_ids]
    panel = torch.utils.data.Subset(train, panel_ids)
    loader = torch.utils.data.DataLoader(
        panel, batch_size=opt.batch_size, shuffle=False, num_workers=0,
    )
    evaluator = GroundingEvaluator(
        only_root=True, thresholds=[0.25, 0.5], topks=[1],
        prefixes=['last_'], filter_non_gt_boxes=False, model='MCLN',
        eval_use_selector_choice_scores=False,
    )

    captured = {}

    def capture_refiner(module, arguments, result):
        # Called in each of the last two decoder layers; keep the final call.
        captured['mask_query'] = arguments[1]
        captured['super_features'] = arguments[2]
        captured['super_xyz_list'] = arguments[3]
        captured['pre_center'] = arguments[4]
        captured['pre_size'] = arguments[5]
        captured['post_center'], captured['post_size'] = result

    hook = model.cs_box_refiner.register_forward_hook(capture_refiner)
    rows = []
    with torch.no_grad():
        for start, raw in enumerate(loader):
            loss, outputs = batch_loss(
                model, raw, criterion, set_criterion, config, False,
            )
            assert bool(torch.isfinite(loss))
            outputs['last_pred_size'] = outputs['last_pred_size'].clamp(min=1e-6)
            assert torch.equal(captured['post_center'], outputs['last_center'])
            assert torch.equal(
                captured['post_size'].clamp(min=1e-6),
                outputs['last_pred_size'],
            )
            evaluator.evaluate(outputs, 'last_')
            scores = compute_default_source_scores(outputs, outputs)
            chosen = evaluator._position_top_indices(
                scores, torch.ones_like(scores, dtype=torch.bool),
                'default_query_axis', 1,
            )[:, 0]
            for bid in range(chosen.shape[0]):
                panel_index = start * opt.batch_size + bid
                query_index = int(chosen[bid])
                root = torch.cat((
                    outputs['center_label'][bid, 0],
                    outputs['size_gts'][bid, 0],
                ))
                before = torch.cat((
                    captured['pre_center'][bid],
                    captured['pre_size'][bid].clamp(min=1e-6),
                ), dim=-1)
                after = torch.cat((
                    outputs['last_center'][bid],
                    outputs['last_pred_size'][bid],
                ), dim=-1)
                before_iou = ious_to_root(before, root)
                after_iou = ious_to_root(after, root)
                xyz = captured['super_xyz_list'][bid].squeeze(0)
                features = captured['super_features'][bid].transpose(0, 1)
                logits = captured['mask_query'][bid, query_index] @ features.T
                mass = logits.sigmoid()
                pre_box = before[query_index]
                post_box = after[query_index]
                rows.append({
                    'panel_index': panel_index,
                    'train_row_id': panel_ids[panel_index],
                    'scan_id': scene_ids[panel_index],
                    'target_id': int(outputs['target_id'][bid]),
                    'query_index': query_index,
                    'score': float(scores[bid, query_index]),
                    'pre_selected_iou': float(before_iou[query_index]),
                    'post_selected_iou': float(after_iou[query_index]),
                    'pre_raw256_oracle_iou': float(before_iou.max()),
                    'post_raw256_oracle_iou': float(after_iou.max()),
                    'center_shift_m': float((post_box[:3] - pre_box[:3]).norm()),
                    'center_shift_over_gt_size': float(
                        (post_box[:3] - pre_box[:3]).norm()
                        / root[3:].norm().clamp_min(1e-6)
                    ),
                    'max_abs_log_size_ratio': float(
                        (post_box[3:] / pre_box[3:]).log().abs().max()
                    ),
                    'm3_mass_outside_gt_box': outside_mass(
                        xyz, mass, root[:3], root[3:]
                    ),
                    'm3_mass_outside_pre_box': outside_mass(
                        xyz, mass, pre_box[:3], pre_box[3:]
                    ),
                    'superpoint_count': int(xyz.shape[0]),
                })
            captured.clear()
    hook.remove()
    assert len(rows) == opt.panel_scenes

    summary = {}
    for threshold, suffix in ((0.25, '025'), (0.5, '050')):
        pre = [row['pre_selected_iou'] > threshold for row in rows]
        post = [row['post_selected_iou'] > threshold for row in rows]
        summary['selected_pre_hits' + suffix] = sum(pre)
        summary['selected_post_hits' + suffix] = sum(post)
        summary['same_query_repairs' + suffix] = sum(
            not old and new for old, new in zip(pre, post)
        )
        summary['same_query_damages' + suffix] = sum(
            old and not new for old, new in zip(pre, post)
        )
        summary['raw256_pre_oracle_hits' + suffix] = sum(
            row['pre_raw256_oracle_iou'] > threshold for row in rows
        )
        summary['raw256_post_oracle_hits' + suffix] = sum(
            row['post_raw256_oracle_iou'] > threshold for row in rows
        )
        assert summary['selected_post_hits' + suffix] == int(
            evaluator.dets[('last_', threshold, 1, 'bbs')]
        )
        assert int(evaluator.gts[('last_', threshold, 1, 'bbs')]) == len(rows)
    for name in ('center_shift_m', 'center_shift_over_gt_size',
                 'max_abs_log_size_ratio', 'm3_mass_outside_gt_box',
                 'm3_mass_outside_pre_box'):
        summary[name + '_median'] = statistics.median(row[name] for row in rows)
    assert all(parameter.grad is None for parameter in model.parameters())
    result = {
        'schema': 'cs-mcln-m3-fixed-train-panel-v1',
        'panel_salt': PANEL_SALT,
        'panel_definition': 'one first ScanRefer expression per hashed train scene',
        'augmentation': False,
        'train_row_ids': panel_ids,
        'checkpoint_epoch': checkpoint_epoch,
        'checkpoint_sha256': file_sha256(opt.checkpoint),
        'checkpoint_formal_metrics': checkpoint_metrics,
        'sample_count': len(rows),
        'optimizer_steps': 0,
        'note': 'GT is used only for offline diagnosis; raw256 oracle is not deployable.',
        'summary': summary,
        'rows': rows,
    }
    opt.output.parent.mkdir(parents=True, exist_ok=True)
    opt.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(opt.output), 'summary': summary}), flush=True)


if __name__ == '__main__':
    main()
