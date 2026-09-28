"""Read-only ScanRefer training-batch audit for native BBS quality targets."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models.cs_native_root_quality import root_matched_quality_loss
from models.source_choice_adapter import compute_default_source_scores
from models.source_moe import compute_query_box_ious
from scripts.train_cs_mcln_scanrefer import (
    experiment_args, load_exact_e71, set_seed,
)
from src.grounding_evaluator import GroundingEvaluator
from train_dist_mod import TrainTester


class AuditGroundingEvaluator(GroundingEvaluator):
    """Record the actual native evaluator ranking without changing it."""

    def __init__(self, expected_scores):
        super().__init__(
            only_root=True, thresholds=[0.25, 0.5], topks=[1],
            prefixes=['last_'], filter_non_gt_boxes=False, model='MCLN',
        )
        self.expected_scores = expected_scores.detach()
        self.selected_queries = []

    def _position_top_indices(self, scores, valid, axis_mode, max_topk):
        row = len(self.selected_queries)
        torch.testing.assert_close(scores[0], self.expected_scores[row])
        top = super()._position_top_indices(scores, valid, axis_mode, max_topk)
        expected_top = super()._position_top_indices(
            self.expected_scores[row:row + 1], valid, axis_mode, max_topk,
        )
        assert torch.equal(top, expected_top)
        self.selected_queries.append(int(top[0, 0]))
        return top


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    set_seed(2027)
    parent = torch.load(str(args.checkpoint), map_location='cpu')
    config = experiment_args(parent['config'], 'native', args.data_root)
    model = TrainTester.get_model(config)
    load_exact_e71(model, parent['model'], 'native')
    del parent
    model = model.cuda().eval()
    criterion, set_criterion = TrainTester.get_criterion(config)
    train, _ = TrainTester.get_datasets(config)
    scan_rows = [
        i for i, anno in enumerate(train.annos)
        if anno['dataset'] == 'scanrefer'
    ]
    assert len(scan_rows) >= 12
    selected = [scan_rows[k * len(scan_rows) // 12] for k in range(12)]
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train, selected), batch_size=12,
        shuffle=False, num_workers=0,
    )
    raw = next(iter(loader))
    batch = TrainTester._to_gpu(raw)
    assert all(name == 'scanrefer' for name in batch['sample_dataset'])
    inputs = TrainTester._get_inputs(batch)
    inputs['train'] = False
    with torch.no_grad():
        output = model(inputs)
        for key, value in batch.items():
            assert key not in output, key
            output[key] = value
        scores = compute_default_source_scores(output, output)
        score_map = (
            output['positive_map'][:, 0].gt(0).float()
            + output['modify_positive_map'][:, 0]
            + output['pron_positive_map'][:, 0]
            + output['rel_positive_map'][:, 0]
            - output['other_entity_map'][:, 0]
        )
        evaluator_scores = (
            output['last_sem_cls_scores'].softmax(-1)
            * score_map[:, None]
        ).sum(-1)
        torch.testing.assert_close(scores, evaluator_scores)
        match_boxes = torch.cat((
            output['last_center'], output['last_pred_size'],
        ), dim=-1)
        boxes = torch.cat((
            output['last_center'],
            output['last_pred_size'].clamp_min(1e-6),
        ), dim=-1)
        gt_boxes = torch.cat((
            output['center_label'][:, :, :3], output['size_gts'],
        ), dim=-1)
        gt_valid = output['box_label_mask'].bool()
        assert bool(gt_valid[:, 0].all())
        assert bool((gt_valid.sum(dim=1) == 1).all())
        ious = compute_query_box_ious(
            boxes, gt_boxes[:, :1], gt_valid[:, :1],
        )
        targets = [
            {
                'boxes': gt_boxes[b, gt_valid[b]],
                'labels': output['sem_cls_label'][b, gt_valid[b]],
                'positive_map': output['positive_map'][b, gt_valid[b]],
                'masks': output['gt_masks'][b, gt_valid[b]],
            }
            for b in range(scores.shape[0])
        ]
        matches = set_criterion.matcher({
            'pred_logits': output['last_sem_cls_scores'],
            'pred_boxes': match_boxes,
            'pred_masks': output['last_pred_masks'],
            'superpoints': output['superpoints'],
        }, targets)

    semantic_logits = output['last_sem_cls_scores'].detach().requires_grad_(True)
    output['last_sem_cls_scores'] = semantic_logits
    native_loss, output = TrainTester._compute_loss(
        output, criterion, set_criterion, config,
    )
    native_gradient = torch.autograd.grad(native_loss, semantic_logits)[0]
    assert bool(torch.isfinite(native_gradient).all())
    native_gradient_norm = float(native_gradient.norm())
    assert native_gradient_norm > 0
    gradient_audit = {}
    for temperature in (0.05, 0.1, 0.25, 0.5, 1.0):
        quality_loss = root_matched_quality_loss(
            compute_default_source_scores(output, output), boxes, gt_boxes,
            matches, batch['sample_dataset'], temperature,
        )
        quality_gradient = torch.autograd.grad(quality_loss, semantic_logits)[0]
        assert bool(torch.isfinite(quality_gradient).all())
        norm = float(quality_gradient.norm())
        gradient_audit[str(temperature)] = {
            'loss': float(quality_loss.detach()),
            'semantic_logit_gradient_l2': norm,
            'gradient_l2_ratio_to_native': norm / native_gradient_norm,
        }
    with torch.no_grad():
        evaluator = AuditGroundingEvaluator(scores)
        evaluation_output = dict(output)
        evaluation_output['last_pred_size'] = boxes[..., 3:]
        evaluator.evaluate_bbox_by_pos_align(evaluation_output, 'last_')
        assert len(evaluator.selected_queries) == len(selected)
        selected_ious = ious.gather(
            1, torch.tensor(evaluator.selected_queries, device=ious.device)[:, None],
        ).squeeze(1)
        evaluator_counts = {}
        for threshold in (0.25, 0.5):
            key = ('last_', threshold, 1, 'bbs')
            assert evaluator.gts[key] == len(selected)
            assert evaluator.dets[key] == int((selected_ious > threshold).sum())
            evaluator_counts[str(threshold)] = {
                'hits': int(evaluator.dets[key]), 'samples': int(evaluator.gts[key]),
            }

    rows = []
    for b, (source, target) in enumerate(matches):
        assert len(source) == len(target) == 1 and int(target[0]) == 0
        matched = int(source[0])
        score = scores[b]
        iou = ious[b]
        best = evaluator.selected_queries[b]
        eligible = iou > 0.5
        utility = iou + 2 * (iou > 0.25).float() + (iou > 0.5).float()
        quality_row = None
        if float(iou[matched]) > 0.25:
            quality_eligible = iou < 0.25
            quality_eligible[source] = False
            quality_eligible[matched] = True
            root_position = int((
                quality_eligible.nonzero(as_tuple=True)[0] == matched
            ).nonzero(as_tuple=True)[0].item())
            quality_row = {
                'eligible_count': int(quality_eligible.sum()),
                'root_target_probability': float(
                    (utility[quality_eligible] / 0.25)
                    .softmax(dim=0)[root_position]
                ),
                'root_score_probability_by_temperature': {
                    str(temperature): float(
                        (score[quality_eligible] / temperature)
                        .softmax(dim=0)[root_position]
                    )
                    for temperature in (0.05, 0.1, 0.25, 0.5, 1.0)
                },
            }
        rows.append({
            'dataset_index': selected[b],
            'matched_query': matched,
            'matched_iou': float(iou[matched]),
            'matched_score': float(score[matched]),
            'deployed_top_query': best,
            'deployed_top_iou': float(iou[best]),
            'score_min': float(score.min()),
            'score_median': float(score.median()),
            'score_max': float(score.max()),
            'score_spread': float(score.max() - score.min()),
            'native_top_probability_temperature_1': float(
                score.softmax(dim=0).max()
            ),
            'native_top_probability_temperature_025': float(
                (score / 0.25).softmax(dim=0).max()
            ),
            'v99_utility_top_probability_temperature_025': float(
                (utility / 0.25).softmax(dim=0).max()
            ),
            'qualified_050_count': int(eligible.sum()),
            'qualified_050_unmatched_count': int(eligible.sum()) - int(eligible[matched]),
            'best_iou': float(iou.max()),
            'matched_root_quality': quality_row,
        })
    result = {
        'source': 'E71 core; 12 fixed ScanRefer training rows; eval-mode forward',
        'model_updates': 0,
        'sample_dataset': 'scanrefer',
        'query_count': int(scores.shape[1]),
        'qualified_root_rows': sum(row['matched_iou'] > 0.25 for row in rows),
        'native_loss': float(native_loss.detach()),
        'native_semantic_logit_gradient_l2': native_gradient_norm,
        'root_quality_loss_by_score_temperature': {
            temperature: value['loss'] for temperature, value in gradient_audit.items()
        },
        'root_quality_gradient_by_score_temperature': gradient_audit,
        'actual_native_evaluator_bbs_counts': evaluator_counts,
        'rows': rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({
        'rows': len(rows),
        'qualified_050_unmatched_total': sum(
            row['qualified_050_unmatched_count'] for row in rows
        ),
        'output': str(args.output),
    }))


if __name__ == '__main__':
    main()
