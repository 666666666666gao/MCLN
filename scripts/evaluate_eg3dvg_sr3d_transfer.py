"""Fixed ScanRefer-to-Sr3D transfer with author object-input/filter protocol and same-forward diagnostics."""
import argparse
import datetime
import gzip
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--stage', choices=['preflight', 'formal'], required=True)
    args = parser.parse_args()
    root = args.spec.parent
    spec = json.loads(args.spec.read_text())
    assert sha(spec['checkpoint']) == spec['checkpoint_sha256']
    assert sha(__file__) == spec['evaluator_sha256']
    source = Path(spec['source'])
    for path, digest in spec['source_files'].items():
        assert sha(source / path) == digest, path
    for path, digest in spec['input_hashes'].items():
        assert sha(path) == digest, path
    if args.stage == 'formal':
        preflight = json.loads((root / 'preflight/receipt.json').read_text())
        assert preflight['status'] == 'complete' and preflight['rows'] == 8
        assert preflight['spec_sha256'] == sha(args.spec)
    output = root / args.stage
    output.mkdir()
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    sys.path.insert(1, str(source / 'pointnet2'))
    import numpy as np
    import torch
    assert torch.__version__ == spec['torch_version']
    assert torch.version.cuda == spec['cuda_version']
    from torch.utils.data import DataLoader, Subset
    from models import EG
    from src.joint_det_dataset import Joint3DDataset
    from src.grounding_evaluator import GroundingEvaluator, _iou3d_par as pairwise_iou3d
    from models.losses import _iou3d_par, box_cxcyczwhd_to_xyzxyz

    seed = spec['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = EG(num_class=256, num_obj_class=485, input_feature_dim=3,
               num_queries=256, num_decoder_layers=6,
               self_position_embedding='loc_learned', contrastive_align_loss=True,
               butd=True, pointnet_ckpt=None, data_path=spec['data_root'], self_attend=True)
    checkpoint = torch.load(spec['checkpoint'], map_location='cpu')
    prefix = spec['checkpoint_key_prefix']
    assert all(k.startswith(prefix) for k in checkpoint['model'])
    state = {k[len(prefix):]: v for k, v in checkpoint['model'].items()}
    assert all(torch.isfinite(v).all() for v in state.values())
    model.load_state_dict(state, strict=True)
    assert all(torch.equal(model.state_dict()[k], v) for k, v in state.items())
    reference = {k: v.clone() for k, v in model.state_dict().items()}
    del checkpoint, state
    model.cuda().eval()
    dataset = Joint3DDataset(dataset_dict={'sr3d': 1, 'scannet': 10}, test_dataset='sr3d',
                            split='val', data_path=spec['data_root'], use_color=True,
                            detect_intermediate=True, butd_cls=True)
    assert len(dataset) == 17726 and not dataset.augment
    assert not dataset.butd_gt and dataset.butd_cls and not dataset.butd
    expected_rows = 8 if args.stage == 'preflight' else 17726
    loader = DataLoader(Subset(dataset, range(expected_rows)), batch_size=8,
                        shuffle=False, num_workers=0, drop_last=False)
    evaluator = GroundingEvaluator(only_root=True, thresholds=[.25, .5], topks=[1, 5, 10],
                                  prefixes=['last_'], filter_non_gt_boxes=True,
                                  logger=logging.getLogger('eg3dvg'), model='EG')
    unfiltered_evaluator = GroundingEvaluator(only_root=True, thresholds=[.25, .5], topks=[1, 5, 10],
                                             prefixes=['last_'], filter_non_gt_boxes=False,
                                             logger=logging.getLogger('eg3dvg-unfiltered'), model='EG')
    counts = {mode: {'rec_hits25': 0, 'rec_hits50': 0}
              for mode in ['bbs', 'bbf', 'bbs_unfiltered', 'bbf_unfiltered']}
    native_box_rule = '0.5 * (last regressed box + box enclosing points with predicted mask > 0.5)'
    started = time.time()
    count = 0
    candidates = np.lib.format.open_memmap(str(output / 'candidates.npy'), mode='w+',
                                          dtype=np.float32, shape=(expected_rows, 256, 17))
    with gzip.open(str(output / 'rows.jsonl.gz'), 'wt') as stream, torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.cuda()
            # Author Sr3D inputs include all scene GT instance boxes and predicted classes.
            # Target identity, target box and text-role labels are not model.forward inputs.
            inputs = {'point_clouds': batch['point_clouds'].float(), 'text': batch['utterances'],
                      'det_boxes': batch['all_detected_boxes'],
                      'det_bbox_label_mask': batch['all_detected_bbox_label_mask'],
                      'det_class_ids': batch['all_detected_class_ids'],
                      'superpoint': batch['superpoint'], 'train': False}
            end = model(inputs)
            assert torch.isfinite(end['last_center']).all()
            assert torch.isfinite(end['last_pred_size']).all()
            assert torch.isfinite(end['last_sem_cls_scores']).all()
            for k, v in batch.items():
                assert k not in end, k
                end[k] = v
            # Native evaluation clamps regressed sizes after its diagnostic loss calculation.
            for k in list(end):
                if 'pred_size' in k:
                    end[k] = end[k].clamp(min=1e-6)
            evaluator.evaluate_bbox_by_pos_align(end, 'last_')
            evaluator.evaluate_bbox_by_sem_align(end, 'last_')
            unfiltered_evaluator.evaluate_bbox_by_pos_align(end, 'last_')
            unfiltered_evaluator.evaluate_bbox_by_sem_align(end, 'last_')
            maps = evaluator._parse_gt(end)
            main_map, modifier, pronoun, other, auxiliary, relation, gt = maps
            mask = torch.stack([(m > .5).int()[:, end['superpoints'][i].long()]
                                for i, m in enumerate(end['last_pred_masks'])])
            mask_boxes = evaluator.get_boxes_from_masks(mask, end['point_clouds'][..., :3])
            raw_boxes = torch.cat([end['last_center'], end['last_pred_size']], dim=-1)
            boxes = (raw_boxes + mask_boxes) / 2
            semantic = (torch.matmul(end['last_proj_queries'], end['proj_tokens'].transpose(-1, -2)) / .07).softmax(-1)
            semantic_padded = semantic.new_zeros(semantic.shape[0], semantic.shape[1], 256)
            semantic_padded[..., :semantic.shape[-1]] = semantic
            probabilities = {'bbs': end['last_sem_cls_scores'].softmax(-1), 'bbf': semantic_padded}
            supported = []
            for i in range(boxes.shape[0]):
                object_boxes = end['all_detected_boxes'][i][end['all_detected_bbox_label_mask'][i].bool()]
                overlaps, _ = pairwise_iou3d(box_cxcyczwhd_to_xyzxyz(object_boxes),
                                             box_cxcyczwhd_to_xyzxyz(boxes[i]))
                supported.append((overlaps.max(0)[0] > .25).float())
            supported = torch.stack(supported)
            selected = {}
            for mode, probs in probabilities.items():
                assert probs.shape[-1] == main_map.shape[-1] == 256
                # Preserve upstream arithmetic and sorting, including role weights and ties.
                parts = [(probs * p[:, :1]).sum(-1) for p in [main_map, modifier, pronoun, relation, other]]
                scores = parts[0] + parts[1] + parts[2] + parts[3] - parts[4]
                selected[mode + '_unfiltered'] = (scores.argsort(1, descending=True)[:, 0], scores)
                # Exact author multiplication, including zero scores for unsupported queries.
                filtered = scores * supported
                selected[mode] = (filtered.argsort(1, descending=True)[:, 0], filtered)
            batch_slice = slice(count, count + boxes.shape[0])
            candidates[batch_slice, :, :6] = raw_boxes.cpu().numpy()
            candidates[batch_slice, :, 6:12] = boxes.cpu().numpy()
            candidates[batch_slice, :, 12] = selected['bbs'][1].cpu().numpy()
            candidates[batch_slice, :, 13] = selected['bbf'][1].cpu().numpy()
            candidates[batch_slice, :, 14] = selected['bbs_unfiltered'][1].cpu().numpy()
            candidates[batch_slice, :, 15] = selected['bbf_unfiltered'][1].cpu().numpy()
            candidates[batch_slice, :, 16] = supported.cpu().numpy()
            for i in range(boxes.shape[0]):
                row = {'row_id': count, 'scan_id': batch['scan_ids'][i],
                       'target_id': int(batch['target_id'][i]), 'utterance': batch['utterances'][i],
                       'point_sha256': hashlib.sha256(batch['point_clouds'][i].cpu().numpy().tobytes()).hexdigest(),
                       'gt_box': gt[i, 0].cpu().tolist(),
                       'object_boxes': end['all_detected_boxes'][i][end['all_detected_bbox_label_mask'][i].bool()].cpu().tolist()}
                for mode, (indices, scores) in selected.items():
                    q = int(indices[i])
                    box = boxes[i, q]
                    overlap = _iou3d_par(box_cxcyczwhd_to_xyzxyz(gt[i, :1]),
                                         box_cxcyczwhd_to_xyzxyz(box[None]))[0].item()
                    assert np.isfinite(overlap)
                    counts[mode]['rec_hits25'] += int(overlap > .25)
                    counts[mode]['rec_hits50'] += int(overlap > .5)
                    row[mode] = {'query': q, 'score': float(scores[i, q]), 'iou': overlap,
                                 'box': box.cpu().tolist(), 'raw_box': raw_boxes[i, q].cpu().tolist(),
                                 'mask_box': mask_boxes[i, q].cpu().tolist()}
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                count += 1
            for mode in counts:
                for threshold, name in [(.25, 'rec_hits25'), (.5, 'rec_hits50')]:
                    check = unfiltered_evaluator if mode.endswith('_unfiltered') else evaluator
                    native_mode = mode.split('_')[0]
                    assert counts[mode][name] == check.dets[('last_', threshold, 1, native_mode)]
                    assert count == check.gts[('last_', threshold, 1, native_mode)]
            if count == 8 or count % 512 == 0 or count == expected_rows:
                print('EG_EVAL_PROGRESS ' + json.dumps({'rows': count, 'total': expected_rows,
                      'elapsed_seconds': time.time() - started, 'metrics': counts}), flush=True)
    assert count == expected_rows
    candidates.flush()
    del candidates
    assert all(torch.equal(value.cpu(), reference[k]) for k, value in model.state_dict().items())
    receipt = {'status': 'complete', 'time_cst': now(), 'stage': args.stage, 'rows': count,
               'elapsed_seconds': time.time() - started, 'metrics': counts, 'primary_mode': 'bbs',
               'native_box_rule': native_box_rule, 'checkpoint_sha256': spec['checkpoint_sha256'],
               'spec_sha256': sha(args.spec), 'rows_sha256': sha(output / 'rows.jsonl.gz'),
               'candidates_sha256': sha(output / 'candidates.npy'),
               'candidate_columns': ['raw_cx', 'raw_cy', 'raw_cz', 'raw_dx', 'raw_dy', 'raw_dz',
                                     'box_cx', 'box_cy', 'box_cz', 'box_dx', 'box_dy', 'box_dz',
                                     'bbs_filtered_score', 'bbf_filtered_score',
                                     'bbs_unfiltered_score', 'bbf_unfiltered_score', 'object_support_gt25'],
               'all_model_states_unchanged': True, 'training_steps': 0, 'optimizer_created': False,
               'upstream_commit': spec['upstream_commit'], 'mask_metric_gate': False,
               'dataset': 'sr3d', 'initialization_dataset': 'scanrefer', 'experiment_type': 'cross_dataset_transfer',
               'object_input': 'all scene GT instance boxes plus author predicted class ids',
               'primary_filter': 'native mean box overlaps any input object by IoU > .25; multiply score by 0/1',
               'unfiltered_modes_are_diagnostic': True}
    write(output / 'receipt.json', receipt)
    print('EG_EVAL_COMPLETE ' + json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
