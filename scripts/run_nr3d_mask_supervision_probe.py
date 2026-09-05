"""Observe native Hungarian Mask supervision and gradients on16 fixed fit rows."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    opt = parser.parse_args()
    addon = opt.manifest.parent
    manifest = json.loads(opt.manifest.read_text())
    source = Path(manifest['model_source'])

    def verify_inputs():
        source_manifest = source / 'g0_source_manifest.json'
        assert file_sha(source_manifest) == manifest['source_manifest_sha256']
        for name, digest in json.loads(source_manifest.read_text())['files'].items():
            assert file_sha(source / name) == digest, name
        for name, digest in manifest['files'].items():
            assert file_sha(addon / name) == digest, name
        for name, metadata in manifest['data_files'].items():
            assert file_sha(Path(name)) == metadata['sha256'], name
        assert file_sha(Path(manifest['checkpoint'])) == manifest['checkpoint_sha256']

    verify_inputs()
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    import scripts
    scripts.__path__ = [str(addon / 'scripts')] + list(scripts.__path__)
    from main_utils import parse_option
    from train_dist_mod import TrainTester
    from src.joint_det_dataset import Joint3DDataset
    from src.grounding_evaluator import GroundingEvaluator
    from models.losses import _iou3d_par, box_cxcyczwhd_to_xyzxyz
    from models.rec_evaluator_filter import build_detector_overlap_valid
    from scripts.nr3d_mask_branch_diagnostic import diagnose_root_candidates, superpoint_mask_ious
    from scripts.nr3d_mask_supervision_probe import (
        gradient_connection, paired_query_gradients, query_gradient_support, superpoint_neighborhood_counts,
    )
    from scripts.run_nr3d_view_pair_role import read_train_rows

    raw_rows = read_train_rows(Path('/root/autodl-tmp/DATA_ROOT'))
    row_ids = manifest['fit_row_ids']
    assert len(row_ids) == len({raw_rows[i]['scan_id'] for i in row_ids}) == 16
    chosen, scenes = [], set()
    for i, row in enumerate(raw_rows):
        fold = int(hashlib.sha256((manifest['split_salt'] + '\0' + row['scan_id']).encode()).hexdigest()[:8], 16) % 5
        if fold != 0 and row['scan_id'] not in scenes:
            chosen.append(i)
            scenes.add(row['scan_id'])
            if len(chosen) == 16:
                break
    assert chosen == row_ids
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    checkpoint = torch.load(manifest['checkpoint'], map_location='cpu')
    assert checkpoint['evaluation_only'] and 'optimizer' not in checkpoint
    sys.argv = [sys.argv[0]]
    args = parse_option()
    vars(args).update(vars(checkpoint['config']))
    assert args.mask_loss_scale == args.consistency_loss_scale == 1
    assert not args.source_choice_selector_train_only and not args.use_source_moe
    model = TrainTester.get_model(args).cuda().eval()
    state = {name[7:]: value for name, value in checkpoint['model'].items()}
    model.load_state_dict(state, strict=True)
    assert model.decoder_query_adapter is None
    del checkpoint

    def verify_state():
        for name, value in model.state_dict().items():
            assert torch.equal(value.detach().cpu(), state[name]), name
        assert all(parameter.grad is None for parameter in model.parameters())

    verify_state()

    class FixedFit(Joint3DDataset):
        def _scene_graph_parse(self, annos):
            annos[:] = [annos[i] for i in row_ids]
            super()._scene_graph_parse(annos)

    dataset = FixedFit(dataset_dict={'nr3d': 1}, test_dataset='nr3d', split='train',
                       data_path='/root/autodl-tmp/DATA_ROOT/', use_color=args.use_color,
                       detect_intermediate=args.detect_intermediate, butd_cls=args.butd_cls,
                       skip_missing_superpoints=args.skip_missing_superpoints)
    dataset.augment = False
    assert len(dataset) == 16
    for anno, row_id in zip(dataset.annos, row_ids):
        assert anno['scan_id'] == raw_rows[row_id]['scan_id']
        assert anno['target_id'] == int(raw_rows[row_id]['target_id'])
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0,
                                         generator=torch.Generator().manual_seed(0))
    criterion, set_criterion = TrainTester.get_criterion(args)
    evaluator = GroundingEvaluator(only_root=True, prefixes=['last_'], topks=[1],
                                   filter_non_gt_boxes=True, eval_use_selector_choice_scores=True)
    captured = {}

    def feature_hook(name):
        def hook(module, inputs, outputs):
            captured[name] = outputs
            if name == 'mask_seed_features':
                captured['point_features_at_mask_projection'] = inputs[0]
        return hook

    def grouper_hook(module, inputs, outputs):
        captured['groups'].append(outputs)

    def criterion_hook(module, inputs, outputs):
        if 'sp_pred_masks' in inputs[0]:
            assert 'last_criterion' not in captured
            captured['last_criterion'] = (inputs[0], inputs[1], outputs[0], outputs[1])

    handles = [model.decoder[-1].register_forward_hook(feature_hook('decoder_query')),
               model.x_query.register_forward_hook(feature_hook('query_projection_output')),
               model.x_mask.register_forward_hook(feature_hook('mask_seed_features')),
               model.super_grouper.register_forward_hook(grouper_hook),
               set_criterion.register_forward_hook(criterion_hook)]
    original_seg = model._seg_seeds_prediction

    def capture_seg(query, mask_feats, end_points, prefix=''):
        captured['seg_inputs'].append((query, mask_feats))
        return original_seg(query, mask_feats, end_points, prefix)

    model._seg_seeds_prediction = capture_seg
    parameter_prefixes = ('x_query.', 'x_mask.', 'rel_encoder.', 'prediction_heads.5.')
    parameters = {name: value for name, value in model.named_parameters()
                  if name.startswith(parameter_prefixes) and value.requires_grad}

    def observe_batch(raw_batch, batch_ids):
        captured.clear()
        captured.update(groups=[], seg_inputs=[])
        batch = TrainTester._to_gpu(raw_batch)
        inputs = TrainTester._get_inputs(batch)
        inputs['train'] = False
        outputs = model(inputs)
        outputs.update(batch)
        total, outputs = TrainTester._compute_loss(outputs, criterion, set_criterion, args)
        assert torch.isfinite(total)
        last_input, targets, mask_terms, indices = captured['last_criterion']
        assert len(captured['groups']) == len(captured['seg_inputs']) == 4
        assert all(len(target['boxes']) == 1 for target in targets)
        components = {'loss_mask': 10, 'loss_dice': 2, 'sp_loss_mask': 5, 'sp_loss_dice': 1,
                      'adaptive_weight_loss_mask': 10, 'adaptive_weight_loss_dice': 2,
                      'corresponding_loss_mask': 10, 'corresponding_loss_dice': 2}
        mask_loss = sum(outputs[name] * coefficient for name, coefficient in components.items())
        for name in components:
            assert torch.equal(outputs[name], mask_terms[name]), name
        grounding_loss = (outputs['loss_ce'] + 5 * outputs['loss_bbox'] + outputs['loss_giou']
                          + outputs['loss_sem_align']) / (args.num_decoder_layers + 1)
        tensors = {name: captured[name] for name in ['decoder_query', 'query_projection_output',
                                                    'mask_seed_features', 'point_features_at_mask_projection']}
        for bid in range(4):
            tensors['raw_query_logits_' + str(bid)] = outputs['sp_last_pred_masks'][bid]
            tensors['text_logits_' + str(bid)] = outputs['last_pred_masks'][bid]
            tensors['alpha_' + str(bid)] = outputs['adaptive_weights'][bid]
            tensors['grouped_mask_features_' + str(bid)] = captured['groups'][bid][0]
        assert all(tensor.requires_grad for tensor in tensors.values())
        probe_values = list(tensors.values()) + list(parameters.values())
        gradients = torch.autograd.grad(mask_loss, probe_values, retain_graph=True, allow_unused=True)
        feature_gradients = dict(zip(tensors, gradients[:len(tensors)]))
        parameter_gradients = dict(zip(parameters, gradients[len(tensors):]))
        grounding_gradient, = torch.autograd.grad(grounding_loss, [captured['decoder_query']])
        assert torch.isfinite(grounding_gradient).all()
        feature_report = {name: gradient_connection(gradient) for name, gradient in feature_gradients.items()}
        parameter_report = {name: gradient_connection(gradient) for name, gradient in parameter_gradients.items()}
        with torch.no_grad():
            rows = diagnose_root_candidates(outputs, evaluator)
            boxes = torch.cat([outputs['last_center'], outputs['last_pred_size'].clamp(min=1e-6)], -1)
            legal = build_detector_overlap_valid(
                boxes, torch.ones(boxes.shape[:2], device=boxes.device, dtype=torch.bool),
                outputs['all_detected_boxes'], outputs['all_detected_bbox_label_mask'].bool(), iou_threshold=.25)
            roots = torch.cat([batch['center_label'][:, :1, :3], batch['size_gts'][:, :1]], -1)
            for bid, (row, row_id) in enumerate(zip(rows, batch_ids)):
                q, sp = captured['seg_inputs'][bid]
                assert torch.equal(torch.einsum('bnd,bdm->bnm', q, sp).squeeze(0), outputs['sp_last_pred_masks'][bid])
                root_ious = _iou3d_par(box_cxcyczwhd_to_xyzxyz(roots[bid]), box_cxcyczwhd_to_xyzxyz(boxes[bid]))[0][0]
                gt_mask = batch['gt_masks'][bid, 0].bool()
                superpoints = outputs['superpoints'][bid].long()
                raw_mask_ious = superpoint_mask_ious(outputs['sp_last_pred_masks'][bid], superpoints, gt_mask)
                matched, target_indices = indices[bid]
                assert matched.numel() == target_indices.numel() == 1 and int(target_indices[0]) == 0
                matched_ids = matched.tolist()
                support = query_gradient_support(feature_gradients['raw_query_logits_' + str(bid)])
                assert set(support['nonzero_query_ids']) <= set(matched_ids)
                good_ids = (legal[bid] & (root_ious > .5)).nonzero().reshape(-1).tolist()
                row.update(fit_row_id=row_id, scan_id=raw_rows[row_id]['scan_id'],
                           target_id=int(raw_rows[row_id]['target_id']), raw_token_count=len(ast.literal_eval(raw_rows[row_id]['tokens'])),
                           matched_query_ids=matched_ids, matched_target_indices=target_indices.tolist(),
                           raw_query_mask_gradient=support, good_legal_box_query_ids=good_ids,
                           good_box_queries_without_direct_mask_gradient=sorted(set(good_ids) - set(support['nonzero_query_ids'])),
                           all_query_box_ious=root_ious.tolist(), all_raw_query_mask_ious=raw_mask_ious.tolist(),
                           matched_box_iou=float(root_ious[matched[0]]), matched_raw_mask_iou=float(raw_mask_ious[matched[0]]),
                           last_query_gradients=paired_query_gradients(feature_gradients['decoder_query'][bid], grounding_gradient[bid]),
                           superpoint_neighborhoods=superpoint_neighborhood_counts(
                               superpoints, gt_mask, outputs['seed_inds'][bid].long(),
                               captured['groups'][bid][1].squeeze(0).long(), sp.shape[-1]),
                           input_point_cloud_sha256=hashlib.sha256(inputs['point_clouds'][bid].detach().cpu().numpy().tobytes()).hexdigest(),
                           projected_query_norm_mean=float(q.norm(dim=-1).mean()),
                           superpoint_feature_norm_mean=float(sp.norm(dim=1).mean()))
        result = {'fit_row_ids': batch_ids, 'rows': rows,
                  'losses': dict(total=float(total), mask=float(mask_loss), grounding=float(grounding_loss),
                                 **{name: float(outputs[name]) for name in components}),
                  'mask_feature_gradients': feature_report, 'mask_parameter_gradients': parameter_report}
        captured.clear()
        return result

    started = time.time()
    batches = []
    for index, batch in enumerate(loader):
        result = observe_batch(batch, row_ids[index * 4:(index + 1) * 4])
        batches.append(result)
        print('M3 BATCH', json.dumps({'batch': index + 1, 'fit_row_ids': result['fit_row_ids'],
              'losses': result['losses'], 'elapsed_seconds': time.time() - started}), flush=True)
    for handle in handles:
        handle.remove()
    model._seg_seeds_prediction = original_seg
    verify_state()
    verify_inputs()
    receipt = {'schema': 'mcln-nr3d-mask-supervision-probe-v1', 'status': 'complete', 'batches': batches,
               'fit_row_ids': row_ids, 'model_forwards': 4, 'gradient_probes': 8,
               'optimizer_steps': 0, 'checkpoint_writes': 0, 'formal_rows': 0, 'heldout_rows': 0,
               'model_mode': 'eval_with_native_autograd_graph', 'model_requires_grad_flags_changed': False,
               'source_data_and_protected_state_unchanged': True, 'native_matching_not_replaced': True,
               'direct_query_mask_dot_product_parity': True, 'formal_promotion': False,
               'manifest_sha256': file_sha(opt.manifest), 'elapsed_seconds': time.time() - started,
               'max_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
               'matcher': {name: getattr(set_criterion.matcher, name) for name in ['cost_class', 'cost_bbox', 'cost_giou', 'cost_masks']}}
    with (addon / 'receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print('M3 COMPLETE', json.dumps({key: value for key, value in receipt.items() if key != 'batches'}), flush=True)


if __name__ == '__main__':
    main()
