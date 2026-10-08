"""Read-only official-parent evaluation at the two recorded batch settings."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--phase', choices=('preflight', 'formal'), required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_bytes())
    root = args.spec.parent
    assert spec['seed'] == 2027 and spec['batch_sizes'] == [8, 24]
    assert not (root / (args.phase + '_receipt.json')).exists()
    assert sha(root / 'evaluate_parent_batches.py') == spec['runner_sha256']
    runtime = Path(spec['runtime'])
    env = json.loads((runtime / 'env_spec.json').read_bytes())
    assert hashlib.sha256(json.dumps(env, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == spec['env_spec_sha256']
    manifest = json.loads(Path(spec['input_manifest']).read_bytes())
    assert sha(spec['input_manifest']) == spec['input_manifest_sha256']
    source = Path(manifest['model_source'])
    assert sha(source / 'src/joint_det_dataset.py') == spec['dataset_source_sha256']
    assert sha(source / 'src/visual_data_handlers.py') == spec['scan_helper_sha256']
    model_source = Path(spec['model_source'])
    port = json.loads(Path(spec['source_port']).read_bytes())
    assert sha(spec['source_port']) == spec['source_port_sha256']
    for name, digest in port['files'].items():
        assert sha(model_source / name) == digest, name
    checkpoint = env['weight_dirs']['scanrefer']
    assert sha(checkpoint['path']) == checkpoint['sha256'] == spec['checkpoint_sha256']

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset

    def reset_rng():
        random.seed(2027)
        np.random.seed(2027)
        torch.manual_seed(2027)
        torch.cuda.manual_seed_all(2027)

    reset_rng()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    from src.joint_det_dataset import Joint3DDataset
    from scripts.scanrefer_data_contract import verify_scanrefer_superpoints
    assert Path(sys.modules['src.joint_det_dataset'].__file__).resolve() == source / 'src/joint_det_dataset.py'
    assert 'models' not in sys.modules
    os.chdir(str(model_source))
    sys.path.insert(0, str(model_source))
    from models.pv_ground import PVGround
    from main_utils import BaseTrainTester
    from prepare_data import DataProcessor
    from pcdet.config import cfg, cfg_from_yaml_file
    path = runtime / 'PV-Ground/src/grounding_evaluator.py'
    assert sha(path) == spec['evaluator_sha256']
    module_spec = importlib.util.spec_from_file_location('official_parent_evaluator', str(path))
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    imports = {name: dict(path=str(Path(sys.modules[name].__file__).resolve()), sha256=sha(sys.modules[name].__file__))
               for name in ('src.joint_det_dataset', 'src.visual_data_handlers', 'sng_parser', 'models.pv_ground', 'main_utils', 'prepare_data')}
    write_json(root / (args.phase + '_imports.json'), imports)

    payload = torch.load(checkpoint['path'], map_location='cpu')
    config = payload['config']
    parent = {name[7:]: value for name, value in payload['model'].items()}
    assert len(parent) == 1234 and all(name.startswith('module.') for name in payload['model'])
    assert config.butd and not config.butd_gt and not config.butd_cls
    cfg_from_yaml_file(str(runtime / 'PV-Ground/wandb_config.yaml'), cfg)
    model = PVGround(cfg, num_class=256, num_queries=256, num_decoder_layers=6,
                     self_position_embedding=config.self_position_embedding, contrastive_align_loss=True,
                     butd=True, pointnet_ckpt=None, data_path=manifest['data_root'], self_attend=config.self_attend)
    assert set(model.state_dict()) - set(parent) == {'text_encoder.embeddings.position_ids'}
    assert torch.equal(model.text_encoder.embeddings.position_ids,
                       torch.arange(model.text_encoder.config.max_position_embeddings).expand((1, -1)))
    model.text_encoder.embeddings.register_buffer('position_ids', model.text_encoder.embeddings.position_ids, persistent=False)
    model.load_state_dict(parent, strict=True)
    assert set(model.state_dict()) == set(parent)
    assert all(not layer.task_read for layer in model.decoder)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in parent.items())
    model.requires_grad_(False).cuda().eval()
    criterion, set_criterion = BaseTrainTester.get_criterion(config)
    processor = DataProcessor(cfg.DATA_PROCESSOR, np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE), False, 6)
    data_check = verify_scanrefer_superpoints(manifest['data_root'], 'val', manifest['superpoint_files']['val'])

    class FormalDataset(Joint3DDataset):
        def _scene_graph_parse(self, annos):
            assert len(annos) == 9508
            for index, row in enumerate(annos):
                row['_formal_row_id'] = index
            super()._scene_graph_parse(annos[:24] if args.phase == 'preflight' else annos)

        def __getitem__(self, index):
            item = super().__getitem__(index)
            item['formal_row_id'] = index
            assert np.isin(item['gt_masks'], [0, 1]).all()
            item['gt_masks'] = item['gt_masks'].astype(np.bool_)
            return item

    os.chdir(str(source))
    dataset = FormalDataset(dataset_dict={'scanrefer': 1}, test_dataset='scanrefer', split='val',
                            data_path=manifest['data_root'], use_color=True, use_height=False, use_multiview=False,
                            detect_intermediate=True, butd=True, butd_gt=False, butd_cls=False,
                            augment_det=False, skip_missing_superpoints=True)
    assert len(dataset) == 9508 and not dataset.augment and not dataset.augment_det
    os.chdir(str(model_source))
    count = 24 if args.phase == 'preflight' else 9508
    view = Subset(dataset, list(range(count)))
    receipts = []
    for batch_size in ([24] if args.phase == 'preflight' else spec['batch_sizes']):
        reset_rng()
        torch.cuda.reset_peak_memory_stats()
        loader = DataLoader(view, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
                            drop_last=False, generator=torch.Generator().manual_seed(2027))
        evaluator = module.GroundingEvaluator(only_root=True, thresholds=[.25, .5], topks=[1, 5, 10],
                                             prefixes=['last_'], filter_non_gt_boxes=False, model='PVGround')
        directory = root / args.phase / ('batch' + str(batch_size))
        directory.mkdir(parents=True)
        all_boxes = np.lib.format.open_memmap(str(directory / 'boxes.npy'), mode='w+', dtype=np.float32, shape=(count, 256, 6))
        all_scores = np.lib.format.open_memmap(str(directory / 'scores.npy'), mode='w+', dtype=np.float32, shape=(count, 2, 256))
        rows = []
        started = time.monotonic()
        with (directory / 'rows.jsonl').open('x') as stream, torch.no_grad():
            for raw in loader:
                voxel = processor.collate_batch([processor.forward({'points': p.numpy().copy(), 'use_lead_xyz': True}) for p in raw['point_clouds']])
                actual = len(raw['utterances'])
                assert np.array_equal(voxel['points'][:, 1:].reshape(actual, 50000, 6), raw['point_clouds'].numpy())
                batch = {name: value.cuda(non_blocking=True) if torch.is_tensor(value) else value for name, value in raw.items()}
                inputs = {name: torch.from_numpy(voxel[name]).float().cuda() for name in ('points', 'voxels', 'voxel_coords', 'voxel_num_points')}
                inputs.update(batch_size=actual, text=raw['utterances'], det_boxes=batch['all_detected_boxes'],
                              det_bbox_label_mask=batch['all_detected_bbox_label_mask'], det_class_ids=batch['all_detected_class_ids'],
                              superpoint=batch['superpoint'], train=False)
                output = model(inputs)
                raw_boxes = torch.cat([output['last_center'], output['last_pred_size']], -1)
                assert not set(output).intersection(batch)
                output.update(batch)
                loss, output = criterion(output, 6, set_criterion, query_points_obj_topk=config.query_points_obj_topk)
                assert torch.isfinite(loss)
                for key in output:
                    if 'pred_size' in key:
                        output[key] = output[key].clamp(min=1e-6)
                evaluator.evaluate(output, 'last_')
                semantic = output['last_sem_cls_scores'].softmax(-1)
                projected = (torch.matmul(output['last_proj_queries'], output['proj_tokens'].transpose(-1, -2)) / .07).softmax(-1)
                contrastive = torch.zeros_like(semantic)
                contrastive[:, :, :projected.shape[-1]] = projected
                boxes = torch.cat([output['last_center'], output['last_pred_size']], -1)
                gt = torch.cat([batch['center_label'][:, 0, :3], batch['size_gts'][:, 0]], -1)
                for bid in range(actual):
                    index = len(rows)
                    assert int(raw['formal_row_id'][bid]) == index
                    record = dict(row_id=index, scan_id=raw['scan_ids'][bid], target_id=int(raw['target_id'][bid]),
                                  root_box=gt[bid].cpu().tolist(), utterance=raw['utterances'][bid],
                                  point_sha256=hashlib.sha256(raw['point_clouds'][bid].numpy().tobytes()).hexdigest(),
                                  detector_sha256=hashlib.sha256(raw['all_detected_boxes'][bid].numpy().tobytes()).hexdigest(),
                                  native_maps_sha256=hashlib.sha256(b''.join(raw[name][bid].numpy().tobytes() for name in
                                      ('positive_map', 'modify_positive_map', 'pron_positive_map', 'rel_positive_map', 'other_entity_map'))).hexdigest())
                    all_boxes[index] = raw_boxes[bid].cpu().numpy()
                    low = torch.maximum(boxes[bid, :, :3] - boxes[bid, :, 3:] / 2, gt[bid, :3] - gt[bid, 3:] / 2)
                    high = torch.minimum(boxes[bid, :, :3] + boxes[bid, :, 3:] / 2, gt[bid, :3] + gt[bid, 3:] / 2)
                    intersection = (high - low).clamp(min=0).prod(-1)
                    ious = intersection / (boxes[bid, :, 3:].prod(-1) + gt[bid, 3:].prod() - intersection)
                    assert torch.isfinite(ious).all()
                    for mode_index, (mode, probability) in enumerate((('bbs', semantic), ('bbf', contrastive))):
                        scores = (probability[bid] * (batch['positive_map'][bid, 0] > 0)).sum(-1)
                        for key in ('modify_positive_map', 'pron_positive_map', 'rel_positive_map'):
                            scores = scores + (probability[bid] * batch[key][bid, 0]).sum(-1)
                        scores = scores - (probability[bid] * batch['other_entity_map'][bid, 0]).sum(-1)
                        all_scores[index, mode_index] = scores.cpu().numpy()
                        query = int(scores.argsort(descending=True)[0])
                        alpha = output['adaptive_weights'][bid]
                        mask = ((alpha * output['last_pred_masks'][bid][0, query] + (1-alpha) * output['sp_last_pred_masks'][bid][query]).sigmoid() > .5)[output['superpoints'][bid]]
                        truth = batch['gt_masks'][bid, 0].bool()
                        mask_iou = float((mask & truth).sum().float() / (mask | truth).sum())
                        record[mode] = dict(query=query, box=boxes[bid, query].cpu().tolist(), iou=float(ious[query]), mask_iou=mask_iou)
                    rows.append(record)
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
                if len(rows) % 512 < batch_size:
                    stream.flush()
                    print(json.dumps(dict(event='PROGRESS', phase=args.phase, batch=batch_size, rows=len(rows), total=count,
                                          seconds=time.monotonic()-started)), flush=True)
                del output, inputs, batch, raw
        assert len(rows) == count
        all_boxes.flush()
        all_scores.flush()
        metrics = {}
        for mode in ('bbs', 'bbf'):
            values = [row[mode] for row in rows]
            metrics[mode] = dict(rec_hits25=sum(r['iou'] > .25 for r in values), rec_hits50=sum(r['iou'] > .5 for r in values),
                                 mask_hits25=sum(r['mask_iou'] > .25 for r in values), mask_hits50=sum(r['mask_iou'] > .5 for r in values))
            assert evaluator.gts[('last_', .25, 1, mode)] == count
            assert metrics[mode]['rec_hits25'] == evaluator.dets[('last_', .25, 1, mode)]
            assert metrics[mode]['rec_hits50'] == evaluator.dets[('last_', .5, 1, mode)]
        assert all(torch.equal(value.detach().cpu(), parent[name]) for name, value in model.state_dict().items())
        receipt = dict(status='ACTUAL_PARENT_BATCH_EVALUATION_COMPLETED', phase=args.phase, rows=count, batch=batch_size,
                       metrics=metrics, seconds=time.monotonic()-started, seed=2027,
                       allocation_peak=torch.cuda.max_memory_allocated(), reservation_peak=torch.cuda.max_memory_reserved(),
                       weight_state_unchanged=True, model_state_tensors=1234, observation_reader_installed=False,
                       optimizer_updates=0, new_weights=0, all256_candidates=True,
                       modes_reported_separately=True, spec_sha256=sha(args.spec),
                       files={name: dict(bytes=(directory/name).stat().st_size, sha256=sha(directory/name))
                              for name in ('boxes.npy', 'scores.npy', 'rows.jsonl')})
        write_json(directory / 'receipt.json', receipt)
        receipts.append(receipt)
        print(json.dumps(receipt), flush=True)
    write_json(root / (args.phase + '_receipt.json'), dict(status='COMPLETED', phase=args.phase,
               receipts=receipts, formal_accuracy_result=args.phase == 'formal', native_best_changed=False,
               training_runs=0, new_weights=0, data_protocol=data_check))


if __name__ == '__main__':
    main()
