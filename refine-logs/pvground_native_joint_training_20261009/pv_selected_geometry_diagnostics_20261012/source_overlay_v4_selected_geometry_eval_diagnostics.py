"""Read existing native evaluation outputs; never change an answer or loss."""
import hashlib
import json
from pathlib import Path

import torch

from models.losses import _iou3d_par, box_cxcyczwhd_to_xyzxyz
from native_root_bbs import native_root_bbs


def tensor_sha256(value):
    array = value.detach().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


class SelectedGeometryEvalDiagnostics:
    def __init__(self, log_directory, epoch):
        self.directory = Path(log_directory)
        self.epoch = epoch
        self.rows = []

    @torch.no_grad()
    def append(self, points):
        score = native_root_bbs(points['last_sem_cls_scores'], points)
        assert score.shape[1] == 256 and torch.isfinite(score).all()
        order = score.argsort(dim=-1, descending=True)
        boxes = torch.cat([points['last_center'], points['last_pred_size']], -1)
        native = torch.cat([points['native_coarse_center'], points['native_coarse_size'].clamp_min(1e-6)], -1)
        mask_endpoint = torch.cat([points['span_mask_center'], points['span_mask_size']], -1)
        truth = torch.cat([points['center_label'][:, 0, :3], points['size_gts'][:, 0]], -1)
        assert boxes.shape == native.shape == mask_endpoint.shape == (len(score), 256, 6)
        assert (boxes[..., 3:] > 0).all() and (mask_endpoint[..., 3:] > 0).all()
        assert points['box_label_mask'][:, 0].bool().all()
        assert set(points['language_dataset']) == {'scanrefer'}
        for bid in range(len(score)):
            query = int(order[bid, 0])
            ious, _ = _iou3d_par(box_cxcyczwhd_to_xyzxyz(truth[bid:bid + 1]),
                                box_cxcyczwhd_to_xyzxyz(boxes[bid]))
            ious = ious[0]
            endpoint = torch.stack([native[bid, query], mask_endpoint[bid, query]])
            endpoint_ious, _ = _iou3d_par(box_cxcyczwhd_to_xyzxyz(truth[bid:bid + 1]),
                                         box_cxcyczwhd_to_xyzxyz(endpoint))
            mapping = points['superpoints'][bid].long()
            own = points['sp_last_pred_masks'][bid][query][mapping]
            text = points['last_pred_masks'][bid][0][query][mapping]
            alpha = points['adaptive_weights'][bid]
            own_foreground = own > 0
            fused_foreground = alpha * text + (1 - alpha) * own > 0
            gt_mask = points['gt_masks'][bid, 0].bool()
            assert mapping.shape == gt_mask.shape
            own_intersection = int((own_foreground & gt_mask).sum())
            own_union = int((own_foreground | gt_mask).sum())
            fused_intersection = int((fused_foreground & gt_mask).sum())
            fused_union = int((fused_foreground | gt_mask).sum())
            text_hash = hashlib.sha256(points['utterances'][bid].encode()).hexdigest()
            row = dict(row_index=len(self.rows), scan_id=points['scan_ids'][bid],
                target_id=int(points['target_id'][bid]), utterance_sha256=text_hash,
                point_cloud_sha256=tensor_sha256(points['point_clouds'][bid]),
                gt_mask_sha256=tensor_sha256(gt_mask),
                selected_query=query, native_bbs_score=float(score[bid, query]),
                selected_box=boxes[bid, query].cpu().tolist(), root_gt_box=truth[bid].cpu().tolist(),
                native_box=native[bid, query].cpu().tolist(), mask_endpoint_box=mask_endpoint[bid, query].cpu().tolist(),
                final_iou=float(ious[query]), native_iou=float(endpoint_ious[0, 0]),
                mask_endpoint_iou=float(endpoint_ious[0, 1]), best256_iou=float(ious.max()),
                own_mask_intersection=own_intersection, own_mask_union=own_union,
                fused_mask_intersection=fused_intersection, fused_mask_union=fused_union)
            ranked_iou = ious[order[bid]]
            for suffix, threshold in (('025', .25), ('050', .5)):
                qualified = ranked_iou > threshold
                ranks = qualified.nonzero().flatten()
                row['qualified256_count' + suffix] = int(qualified.sum())
                row['first_qualified_rank' + suffix] = int(ranks[0]) + 1 if len(ranks) else None
            self.rows.append(row)

    def finish(self, metrics):
        assert metrics['rows'] == len(self.rows) == 9508 and metrics['primary_score'] == 'last/bbs'
        summary = dict(epoch=self.epoch, rows=len(self.rows), primary_score='last/bbs',
            candidates_per_expression=256, GT_used_only_in_diagnostics=True,
            new_neural_forwards=0, answers_changed=False, losses_changed=False)
        for suffix, threshold in (('025', .25), ('050', .5)):
            hits = sum(row['final_iou'] > threshold for row in self.rows)
            assert hits == metrics['hits' + suffix]
            summary[suffix] = dict(hits=hits,
                full256_covered=sum(row['best256_iou'] > threshold for row in self.rows),
                errors_with_qualified_candidate=sum(row['final_iou'] <= threshold < row['best256_iou'] for row in self.rows),
                no_qualified_candidate=sum(row['best256_iou'] <= threshold for row in self.rows),
                final_vs_native_repairs=sum(row['native_iou'] <= threshold < row['final_iou'] for row in self.rows),
                final_vs_native_breaks=sum(row['final_iou'] <= threshold < row['native_iou'] for row in self.rows),
                final_vs_mask_endpoint_repairs=sum(row['mask_endpoint_iou'] <= threshold < row['final_iou'] for row in self.rows),
                final_vs_mask_endpoint_breaks=sum(row['final_iou'] <= threshold < row['mask_endpoint_iou'] for row in self.rows))
        summary['selected_own_and_fused_support_qualified'] = sum(
            2 * row['own_mask_intersection'] > row['own_mask_union']
            and 2 * row['fused_mask_intersection'] > row['fused_mask_union'] for row in self.rows)
        if self.epoch:
            initial = [json.loads(line) for line in (self.directory / 'native_rows_epoch000.jsonl').read_text().splitlines()]
            assert len(initial) == len(self.rows)
            for previous, current in zip(initial, self.rows):
                for name in ('row_index', 'scan_id', 'target_id', 'utterance_sha256', 'root_gt_box'):
                    assert previous[name] == current[name]
            same_input = [previous['point_cloud_sha256'] == current['point_cloud_sha256']
                          and previous['gt_mask_sha256'] == current['gt_mask_sha256']
                          for previous, current in zip(initial, self.rows)]
            summary['versus_E0'] = dict(same_expression_order_and_GT_boxes=True,
                exact_same_input_rows=sum(same_input), all_inputs_identical=all(same_input),
                interpretation='expression-level changes; exact-input pairing only for matching input fingerprints')
            for suffix, threshold in (('025', .25), ('050', .5)):
                summary['versus_E0'][suffix] = dict(
                    repairs=sum(previous['final_iou'] <= threshold < current['final_iou'] for previous, current in zip(initial, self.rows)),
                    breaks=sum(current['final_iou'] <= threshold < previous['final_iou'] for previous, current in zip(initial, self.rows)),
                    selected_query_changes=sum(previous['selected_query'] != current['selected_query'] for previous, current in zip(initial, self.rows)),
                    exact_input_repairs=sum(equal and previous['final_iou'] <= threshold < current['final_iou'] for equal, previous, current in zip(same_input, initial, self.rows)),
                    exact_input_breaks=sum(equal and current['final_iou'] <= threshold < previous['final_iou'] for equal, previous, current in zip(same_input, initial, self.rows)))
        path = self.directory / ('native_rows_epoch%03d.jsonl' % self.epoch)
        with path.open('x', encoding='utf-8') as stream:
            for row in self.rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
        summary['row_file'] = path.name
        summary['row_file_bytes'] = path.stat().st_size
        summary['row_file_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        with (self.directory / ('native_diagnostic_epoch%03d.json' % self.epoch)).open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(summary, indent=2) + '\n')
        return summary
