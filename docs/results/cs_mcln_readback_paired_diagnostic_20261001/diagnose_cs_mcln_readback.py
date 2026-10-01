"""Compare native REC choices before/after R on one fixed set of predictions."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models.losses import _iou3d_par, box_cxcyczwhd_to_xyzxyz
from scripts.train_cs_mcln_scanrefer import (
    batch_loss, experiment_args, load_exact_e71, set_seed,
)
from src.grounding_evaluator import GroundingEvaluator
from train_dist_mod import TrainTester


class ScoreRecordingEvaluator(GroundingEvaluator):
    """Capture the actual native score tensor without recomputing token maps."""

    def _position_top_indices(self, scores, valid, axis_mode, max_topk):
        selected = GroundingEvaluator._position_top_indices(
            scores, valid, axis_mode, max_topk,
        )
        ranking = GroundingEvaluator._position_top_indices(
            scores, valid, axis_mode, 256,
        )
        assert torch.equal(selected, ranking[:, :selected.shape[1]])
        self.batch_scores.append(scores)
        self.batch_rankings.append(ranking)
        return selected


def bypass_readback(model, captured, outputs):
    """Replay only the eval-mode semantic head; keep every box/Mask unchanged."""
    head = model.prediction_heads[-1]
    assert not head.training
    replay = {}
    head.write_semantic_scores(
        captured['after'].transpose(1, 2).contiguous(), replay, 'last_',
    )
    assert torch.equal(replay['last_sem_cls_scores'], outputs['last_sem_cls_scores'])
    bypass = dict(outputs)
    head.write_semantic_scores(
        captured['before'].transpose(1, 2).contiguous(), bypass, 'last_',
    )
    assert all(bypass[key] is value for key, value in outputs.items()
               if key != 'last_sem_cls_scores')
    return bypass


def root_ious(boxes, root):
    return _iou3d_par(
        box_cxcyczwhd_to_xyzxyz(root.unsqueeze(0)),
        box_cxcyczwhd_to_xyzxyz(boxes),
    )[0][0]


def first_qualified_rank(ordered_ious, threshold):
    qualified = (ordered_ious > threshold).nonzero().flatten()
    return int(qualified[0]) + 1 if qualified.numel() else None


def summarize(rows):
    result = {'samples': len(rows)}
    for threshold, suffix in ((0.25, '025'), (0.5, '050')):
        readback = [row['readback_iou'] > threshold for row in rows]
        bypass = [row['bypass_iou'] > threshold for row in rows]
        result['readback_hits' + suffix] = sum(readback)
        result['bypass_hits' + suffix] = sum(bypass)
        result['readback_repairs' + suffix] = sum(
            new and not old for new, old in zip(readback, bypass)
        )
        result['readback_damages' + suffix] = sum(
            old and not new for new, old in zip(readback, bypass)
        )
        result['net_readback_hits' + suffix] = sum(readback) - sum(bypass)
        result['raw256_oracle_hits' + suffix] = sum(
            row['raw256_oracle_iou'] > threshold for row in rows
        )
        result['pre_m3_raw256_oracle_hits' + suffix] = sum(
            row['pre_m3_raw256_oracle_iou'] > threshold for row in rows
        )
        result['readback_qualified_not_selected' + suffix] = sum(
            not hit and row['raw256_oracle_iou'] > threshold
            for hit, row in zip(readback, rows)
        )
    result['changed_query_count'] = sum(
        row['readback_query'] != row['bypass_query'] for row in rows
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--limit-expressions', type=int)
    opt = parser.parse_args()
    assert not opt.output.exists()
    set_seed(2027)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    parent = torch.load(str(opt.parent), map_location='cpu')
    config = experiment_args(parent['config'], 'cs_readback', opt.data_root)
    model = TrainTester.get_model(config)
    initial_load = load_exact_e71(model, parent['model'], 'cs_readback')
    del parent
    saved = torch.load(str(opt.checkpoint), map_location='cpu')
    assert saved['arm'] == 'cs_readback' and saved['initial_load'] == initial_load
    assert saved['seed'] == 2027 and saved['batch_size'] == opt.batch_size == 12
    assert saved['metrics']['samples'] == 9508
    model.load_state_dict(saved['model'], strict=True)
    checkpoint_epoch, checkpoint_metrics = int(saved['epoch']), saved['metrics']
    del saved
    model = model.cuda().eval().requires_grad_(False)
    criterion, set_criterion = TrainTester.get_criterion(config)
    _, validation = TrainTester.get_datasets(config)
    assert len(validation) == 9508 and not validation.augment
    count = len(validation)
    if opt.limit_expressions is not None:
        assert 0 < opt.limit_expressions <= count
        count = opt.limit_expressions
    panel = torch.utils.data.Subset(validation, range(count))
    loader = torch.utils.data.DataLoader(
        panel, batch_size=opt.batch_size, shuffle=False, num_workers=0,
    )
    evaluators = [ScoreRecordingEvaluator(
        only_root=True, thresholds=[0.25, 0.5], topks=[1],
        prefixes=['last_'], filter_non_gt_boxes=False, model='MCLN',
        eval_use_selector_choice_scores=False,
    ) for _ in range(2)]
    captured = {}

    def capture(module, arguments, result):
        assert not captured
        captured.update(before=arguments[0], evidence=arguments[1], after=result)

    hook = model.cs_geometry_readback.register_forward_hook(capture)
    rows = []
    with torch.no_grad():
        for batch_index, raw in enumerate(loader):
            loss, outputs = batch_loss(
                model, raw, criterion, set_criterion, config, False,
            )
            assert bool(torch.isfinite(loss))
            outputs['last_pred_size'] = outputs['last_pred_size'].clamp(min=1e-6)
            bypass = bypass_readback(model, captured, outputs)
            boxes = torch.cat((outputs['last_center'], outputs['last_pred_size']), -1)
            assert bool(torch.isfinite(boxes).all()) and boxes.shape[1] == 256
            resolved, override, valid, axis = evaluators[0]._resolve_position_candidates(
                outputs, 'last_', boxes,
            )
            assert resolved is boxes and override is None
            assert axis == 'default_query_axis' and bool(valid.all())
            variants = (outputs, bypass)
            scores, rankings = [], []
            for evaluator, variant in zip(evaluators, variants):
                evaluator.batch_scores, evaluator.batch_rankings = [], []
                evaluator.evaluate_bbox_by_pos_align(variant, 'last_')
                assert len(evaluator.batch_scores) == boxes.shape[0]
                score = torch.cat(evaluator.batch_scores, dim=0)
                assert bool(torch.isfinite(score).all())
                scores.append(score)
                rankings.append(torch.cat(evaluator.batch_rankings, dim=0))
            evidence = captured['evidence']
            coarse = torch.cat((evidence[..., :3], evidence[..., -9:-6].clamp(min=1e-6)), -1)
            for bid in range(boxes.shape[0]):
                row_id = batch_index * opt.batch_size + bid
                annotation = validation.annos[row_id]
                root = torch.cat((outputs['center_label'][bid, 0], outputs['size_gts'][bid, 0]))
                ious = root_ious(boxes[bid], root)
                coarse_ious = root_ious(coarse[bid], root)
                readback_query, bypass_query = [int(order[bid, 0]) for order in rankings]
                row = {
                    'row_id': row_id, 'scan_id': annotation['scan_id'],
                    'target_id': int(outputs['target_id'][bid]),
                    'annotation_word_count': len(annotation['utterance'].split()),
                    'readback_query': readback_query, 'bypass_query': bypass_query,
                    'readback_iou': float(ious[readback_query]),
                    'bypass_iou': float(ious[bypass_query]),
                    'readback_score': float(scores[0][bid, readback_query]),
                    'bypass_score': float(scores[1][bid, bypass_query]),
                    'pre_m3_readback_query_iou': float(coarse_ious[readback_query]),
                    'raw256_oracle_iou': float(ious.max()),
                    'pre_m3_raw256_oracle_iou': float(coarse_ious.max()),
                }
                for name, order in zip(('readback', 'bypass'), rankings):
                    ordered = ious[order[bid]]
                    for threshold, suffix in ((0.25, '025'), (0.5, '050')):
                        row[name + '_first_qualified_rank' + suffix] = first_qualified_rank(ordered, threshold)
                    row[name + '_top16_oracle_iou'] = float(ordered[:16].max())
                rows.append(row)
            captured.clear()
            print('PAIRED_ROWS', len(rows), count, flush=True)
    hook.remove()
    assert len(rows) == count
    summary = summarize(rows)
    for name, evaluator in zip(('readback', 'bypass'), evaluators):
        for threshold, suffix in ((0.25, '025'), (0.5, '050')):
            assert summary[name + '_hits' + suffix] == int(evaluator.dets[('last_', threshold, 1, 'bbs')])
            assert int(evaluator.gts[('last_', threshold, 1, 'bbs')]) == count
    assert all(parameter.grad is None for parameter in model.parameters())
    groups = {}
    for name, low, high in (('words_lt9', 0, 8), ('words9to12', 9, 12), ('words_ge13', 13, None)):
        group = [row for row in rows if row['annotation_word_count'] >= low
                 and (high is None or row['annotation_word_count'] <= high)]
        groups[name] = summarize(group)
    result = {
        'schema': 'cs-readback-fixed-prediction-paired-rec-v1',
        'split': 'ScanRefer validation', 'full_validation': count == 9508,
        'checkpoint_epoch': checkpoint_epoch, 'checkpoint_formal_metrics': checkpoint_metrics,
        'sample_count': count, 'optimizer_steps': 0, 'batch_size': opt.batch_size,
        'candidate_scope': 'same native last/bbs 256 queries; filter_non_gt_boxes=False',
        'comparison': 'one model forward; eval semantic head on pre/post-R Query; all other outputs shared',
        'scope_limit': 'direct final R readback effect only; not a retrained no-R ablation or a causal training analysis',
        'oracle_note': 'GT is used only for offline IoU/coverage; raw256 oracle is not deployable',
        'summary': summary, 'word_groups': groups, 'rows': rows,
    }
    if count == 9508:
        result['formal_result_match'] = {
            'hits' + suffix: summary['readback_hits' + suffix] == checkpoint_metrics['hits' + suffix]
            for suffix in ('025', '050')
        }
    opt.output.parent.mkdir(parents=True, exist_ok=True)
    opt.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(opt.output), 'summary': summary}), flush=True)
    if count == 9508:
        assert all(result['formal_result_match'].values()), result['formal_result_match']


if __name__ == '__main__':
    main()
