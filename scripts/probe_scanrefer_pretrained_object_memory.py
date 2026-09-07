"""Real ScanRefer native-path probe of frozen appearance injection; no saved weights."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import sys
import time


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    manifest_path = parser.parse_args().manifest.resolve()
    config = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    source = root / 'model_source'
    source_receipt = json.loads((source / 'appearance_source_manifest.json').read_text())
    for name, digest in source_receipt['files'].items():
        assert sha(source / name) == digest, name
    cache = Path(config['cache_root'])
    assert (cache / 'cache.exit').read_text().strip() == '0'
    cache_receipt = json.loads((cache / 'receipt.json').read_text())
    assert cache_receipt['status'] == 'complete' and cache_receipt['scene_count'] == 562
    assert sha(cache / 'scenes.jsonl') == cache_receipt['scenes_sha256']
    cached_scenes = {v['scene_id']: v for v in map(json.loads, (cache / 'scenes.jsonl').read_text().splitlines())}
    for artifact in config['artifacts'].values():
        assert sha(artifact['path']) == artifact['sha256']
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from main_utils import parse_option
    from train_dist_mod import TrainTester
    from src.joint_det_dataset import Joint3DDataset
    from models.pretrained_object_appearance import PretrainedObjectAppearance
    from scripts.run_frozen_v99_pareto_contextual_official import build_authoritative_command
    from scripts.scanrefer_data_contract import set_scanrefer_data_root, verify_scanrefer_superpoints
    from scripts.scanrefer_joint_readout import JointRecReadout

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    command = set_scanrefer_data_root(build_authoritative_command(root / 'unused_output'), config['data_root'])
    verify_scanrefer_superpoints(config['data_root'], 'train', config['train_superpoint_files'])
    sys.argv = [sys.argv[0]] + command[command.index('train_dist_mod.py') + 1:]
    args = parse_option()
    assert args.dataset == ['scanrefer'] and args.butd and not args.butd_cls and not args.butd_gt
    assert args.checkpoint_path == config['artifacts']['backbone']['path']
    initial = {key[7:]: value for key, value in torch.load(args.checkpoint_path, map_location='cpu')['model'].items()}
    model = TrainTester.get_model(args).cuda().eval().requires_grad_(False)
    model.load_state_dict(initial, strict=True)
    assert len(initial) == 1144 and model.object_appearance is None and model.decoder[-1].local_visual is None
    fusion = PretrainedObjectAppearance().cuda()
    criterion, set_criterion = TrainTester.get_criterion(args)
    artifacts = {k: torch.load(v['path'], map_location='cpu') for k, v in config['artifacts'].items() if k != 'backbone'}
    readout = JointRecReadout(artifacts).cuda().eval().requires_grad_(False)
    selected = config['selected_row_ids']
    assert len(selected) == 16

    class ProbeDataset(Joint3DDataset):
        def _scene_graph_parse(self, annos):
            assert len(annos) == 36665
            for index, item in enumerate(annos):
                item['_appearance_id'] = index
            scenes = {annos[i]['scan_id'] for i in selected}
            annos[:] = [v for v in annos if v['scan_id'] in scenes]
            super()._scene_graph_parse(annos)

    dataset = ProbeDataset(dataset_dict={'scanrefer': 1}, test_dataset='scanrefer', split='train',
                           data_path=args.data_root, use_color=args.use_color, use_height=args.use_height,
                           use_multiview=args.use_multiview, detect_intermediate=args.detect_intermediate,
                           butd=args.butd, butd_gt=False, butd_cls=False, augment_det=False,
                           skip_missing_superpoints=args.skip_missing_superpoints)
    by_id = {v['_appearance_id']: v for v in dataset.annos}
    dataset.annos = [by_id[i] for i in selected]
    dataset.augment = False
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    observations = []
    started = time.time()
    keys = ['last_center', 'last_pred_size', 'last_sem_cls_scores', 'last_proj_queries',
            'sp_last_pred_masks', 'last_pred_masks', 'adaptive_weights']
    for batch_index, raw in enumerate(loader):
        batch = TrainTester._to_gpu(raw)
        inputs = TrainTester._get_inputs(batch)
        inputs['train'] = False
        shape = inputs['det_bbox_label_mask'].shape
        features = torch.zeros(*shape, 1280, device='cuda')
        available = torch.zeros(*shape, dtype=torch.bool, device='cuda')
        for i, scene_id in enumerate(raw['scan_ids']):
            info = cached_scenes[scene_id]
            assert hashlib.sha256(inputs['point_clouds'][i].cpu().numpy().tobytes()).hexdigest() == info['native_point_sha256']
            path = cache / info['file']
            assert sha(path) == info['file_sha256']
            data = np.load(str(path))
            count = len(data['boxes'])
            assert int(inputs['det_bbox_label_mask'][i].sum()) == count
            assert torch.equal(inputs['det_boxes'][i, :count], torch.from_numpy(data['boxes']).cuda())
            features[i, :count] = torch.from_numpy(data['features']).cuda()
            available[i, :count] = torch.from_numpy(data['available']).cuda()
        inputs['det_visual_features'] = features
        inputs['det_visual_available'] = available
        model.object_appearance = None
        with torch.no_grad():
            reference = model(inputs)
            reference_runtime = readout(reference, inputs)['runtime']
        model.object_appearance = fusion
        model.zero_grad(set_to_none=True)
        candidate = model(inputs)
        for key in keys:
            if torch.is_tensor(reference[key]):
                assert torch.equal(reference[key], candidate[key]), key
            else:
                assert all(torch.equal(a, b) for a, b in zip(reference[key], candidate[key])), key
        with torch.no_grad():
            runtime = readout(candidate, inputs)['runtime']
            for key, value in reference_runtime.items():
                assert torch.equal(value, runtime[key]) if torch.is_tensor(value) else value == runtime[key], key
        candidate.update(batch)
        loss, candidate = TrainTester._compute_loss(candidate, criterion, set_criterion, args)
        assert torch.isfinite(loss)
        loss.backward()
        gradient = float(fusion.projection.weight.grad.norm())
        assert gradient > 0 and torch.isfinite(fusion.projection.weight.grad).all()
        observations.append(dict(batch=batch_index, rows=len(raw['scan_ids']), available=int(available.sum()),
                                 native_and_v99_initial_parity=True, native_loss=float(loss), appearance_gradient_norm=gradient))
        if batch_index == 0:
            update_inputs, update_batch = inputs, batch
        print('APPEARANCE_NATIVE_BATCH', json.dumps(observations[-1]), flush=True)
        del reference, reference_runtime, candidate, runtime, loss
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=1e-4, weight_decay=.0005)
    steps = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(update_inputs)
        outputs.update(update_batch)
        loss, outputs = TrainTester._compute_loss(outputs, criterion, set_criterion, args)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(fusion.projection.weight.grad).all()
        norm = torch.nn.utils.clip_grad_norm_(fusion.parameters(), .1)
        optimizer.step()
        steps.append(dict(step=step + 1, native_loss=float(loss), gradient_norm=float(norm)))
        del outputs, loss
    assert fusion.projection.weight.detach().abs().sum() > 0
    state = model.state_dict()
    assert all(torch.equal(state[name].cpu(), value) for name, value in initial.items())
    assert set(state) - set(initial) == {'object_appearance.projection.weight'}
    result = dict(status='pass', rows=16, observations=observations, disposable_steps=steps,
                  original_state_tensors_preserved=1144, appearance_parameters=204800,
                  source_manifest_sha256=sha(source / 'appearance_source_manifest.json'),
                  cache_receipt_sha256=sha(cache / 'receipt.json'), manifest_sha256=sha(manifest_path),
                  elapsed_seconds=time.time() - started, gpu_peak_bytes=torch.cuda.max_memory_allocated(),
                  formal_rows=0, checkpoint_writes=0, new_rec_metrics=False,
                  scope='Engineering execution of native upstream appearance path, not a trained quality result.')
    with (root / 'receipt.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print('APPEARANCE_NATIVE_PASS', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
