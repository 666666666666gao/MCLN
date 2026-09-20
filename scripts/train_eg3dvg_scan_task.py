"""ScanRefer paired continuation from author EG weights; fixed complete epochs and native losses."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--stage', choices=['preflight', 'fit'], required=True)
    parser.add_argument('--epoch', type=int, default=1)
    args = parser.parse_args()
    root = args.spec.parent
    spec = json.loads(args.spec.read_text())
    assert sha(__file__) == spec['trainer_sha256']
    assert sha(spec['checkpoint']) == spec['checkpoint_sha256']
    for path, digest in spec['source_files'].items():
        assert sha(Path(spec['source']) / path) == digest, path
    for path, digest in spec['input_hashes'].items():
        assert sha(path) == digest, path
    if args.stage == 'fit':
        probe = json.loads((root / 'preflight/receipt.json').read_text())
        assert probe['status'] == 'complete' and probe['optimizer_steps'] == 2
        assert probe['spec_sha256'] == sha(args.spec)
    assert 1 <= args.epoch <= spec['max_epochs']
    output = root / ('preflight' if args.stage == 'preflight' else 'epoch_%02d' % args.epoch)
    output.mkdir()
    os.chdir(spec['source'])
    sys.path.insert(0, spec['source'])
    sys.path.insert(1, str(Path(spec['source']) / 'pointnet2'))
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from models import EG
    from main_utils import BaseTrainTester
    from src.joint_det_dataset import Joint3DDataset

    assert torch.__version__ == '1.12.0+cu116'
    random.seed(spec['seed'])
    np.random.seed(spec['seed'])
    torch.manual_seed(spec['seed'])
    torch.cuda.manual_seed_all(spec['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = EG(num_class=256, num_obj_class=485, input_feature_dim=3, num_queries=256,
               num_decoder_layers=6, self_position_embedding='loc_learned',
               contrastive_align_loss=True, butd=True, pointnet_ckpt=None,
               data_path=spec['data_root'], self_attend=True)
    checkpoint = torch.load(spec['checkpoint'], map_location='cpu')
    assert all(k.startswith('module.') for k in checkpoint['model'])
    state = {k[7:]: v for k, v in checkpoint['model'].items()}
    model.load_state_dict(state, strict=True)
    assert all(torch.equal(model.state_dict()[k], v) for k, v in state.items())
    if spec['task_read']:
        from models.eg3dvg_task_read import install_task_read
        install_task_read(model)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    del checkpoint, state
    model.cuda().train()
    native_args = SimpleNamespace(num_decoder_layers=6, query_points_obj_topk=4,
                                  use_contrastive_align=True, use_soft_token_loss=True,
                                  frozen=False, small_lr=False, lr=spec['lr'],
                                  lr_backbone=spec['lr_backbone'],
                                  text_encoder_lr=spec['text_encoder_lr'], weight_decay=.0005)
    criterion, set_criterion = BaseTrainTester.get_criterion(native_args)
    set_criterion.cuda()
    optimizer = BaseTrainTester.get_optimizer(native_args, model)
    dataset = Joint3DDataset(dataset_dict={'scanrefer': 1, 'scannet': 10}, test_dataset='scanrefer',
                            split='train', data_path=spec['data_root'], use_color=True,
                            detect_intermediate=True, butd=True, augment_det=True)
    assert len(dataset) == spec['fit_rows'] and dataset.augment
    assert dataset.butd and not dataset.butd_cls and not dataset.butd_gt
    if args.stage == 'preflight':
        order = spec['preflight_indices']
    else:
        order = np.load(spec['order_paths'][args.epoch - 1]).tolist()
        assert len(order) == len(dataset) and sorted(order) == list(range(len(dataset)))
    loader = DataLoader(Subset(dataset, order), batch_size=8, shuffle=False,
                        num_workers=0, drop_last=False)
    if args.stage == 'fit' and args.epoch > 1:
        previous = torch.load(Path(spec['state_root']) / 'latest.pth', map_location='cpu')
        assert previous['epoch'] == args.epoch - 1
        assert previous['spec_sha256'] == sha(args.spec)
        model.load_state_dict({k[7:]: v for k, v in previous['model'].items()}, strict=True)
        optimizer.load_state_dict(previous['optimizer'])
        random.setstate(previous['random_state'])
        np.random.set_state(previous['numpy_state'])
        torch.set_rng_state(previous['torch_rng_state'])
        torch.cuda.set_rng_state_all(previous['cuda_rng_states'])
        del previous
    steps = 0
    rows = 0
    start = time.time()
    torch.cuda.reset_peak_memory_stats()
    log = (output / 'updates.jsonl').open('x')
    for batch in loader:
        batch = BaseTrainTester._to_gpu(batch)
        inputs = {'point_clouds': batch['point_clouds'].float(), 'text': batch['utterances'],
                  'det_boxes': batch['all_detected_boxes'],
                  'det_bbox_label_mask': batch['all_detected_bbox_label_mask'],
                  'det_class_ids': batch['all_detected_class_ids'], 'superpoint': batch['superpoint']}
        negative_inputs = dict(inputs)
        negative_inputs['text'] = batch['negative_utterances']
        end = model(inputs)
        negative_end = model(negative_inputs)
        for key, value in batch.items():
            assert key not in end, key
            end[key] = value
        native_loss, end = BaseTrainTester._compute_loss(end, criterion, set_criterion, native_args)
        native_value = float(native_loss.detach())
        loss, end = BaseTrainTester._compute_negative_loss(native_loss, end, negative_end)
        assert torch.isfinite(loss)
        optimizer.zero_grad()
        loss.backward()
        task_norms = []
        if spec['task_read']:
            task_gradient = model.decoder[-1].task_queries.grad
            assert torch.isfinite(task_gradient).all()
            task_norms = [float(value.norm()) for value in task_gradient]
            assert all(value > 0 for value in task_norms)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
        assert torch.isfinite(grad_norm)
        optimizer.step()
        steps += 1
        rows += len(batch['utterances'])
        record = {'epoch': args.epoch, 'task_gradient_norms': task_norms,
                  'point_sha256': hashlib.sha256(batch['point_clouds'].detach().cpu().numpy().tobytes()).hexdigest(),
                  'step': steps, 'rows': rows, 'loss': float(loss.detach()),
                  'native_loss': native_value, 'grad_norm_before_clip': float(grad_norm),
                  'elapsed_seconds': time.time() - start,
                  'scan_ids': list(batch['scan_ids']),
                  'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        log.write(json.dumps(record, allow_nan=False) + '\n')
        log.flush()
        if steps <= 2 or steps % 64 == 0 or steps == len(loader):
            print('EG_ADAPT_PROGRESS ' + json.dumps(record), flush=True)
        if args.stage == 'fit' and (steps % 512 == 0 or steps == len(loader)):
            saved = {'model': {'module.' + k: v.cpu() for k, v in model.state_dict().items()},
                     'optimizer': optimizer.state_dict(), 'epoch': args.epoch, 'step': steps,
                     'rows': rows, 'spec_sha256': sha(args.spec),
                     'parent_checkpoint_sha256': spec['checkpoint_sha256'],
                     'random_state': random.getstate(), 'numpy_state': np.random.get_state(),
                     'torch_rng_state': torch.get_rng_state(),
                     'cuda_rng_states': torch.cuda.get_rng_state_all()}
            destination = Path(spec['state_root']) / 'latest.pth'
            temporary = destination.with_suffix('.writing')
            torch.save(saved, str(temporary))
            os.replace(str(temporary), str(destination))
            del saved
        del inputs, negative_inputs, end, negative_end, loss, native_loss, batch
    log.close()
    assert steps == (2 if args.stage == 'preflight' else spec['fit_updates'])
    assert rows == (16 if args.stage == 'preflight' else spec['fit_rows'])
    if spec['task_read']:
        assert all(torch.count_nonzero(v) > 0 for v in model.decoder[-1].task_queries)
    params = dict(model.named_parameters())
    assert all(torch.equal(params[n].detach().cpu(), value) for n, value in frozen.items())
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    receipt = {'status': 'complete', 'stage': args.stage, 'time_cst': now(),
               'epoch': args.epoch, 'rows': rows, 'optimizer_steps': steps,
               'total_fit_steps': args.epoch * steps if args.stage == 'fit' else 0, 'elapsed_seconds': time.time() - start,
               'spec_sha256': sha(args.spec), 'parent_checkpoint_sha256': spec['checkpoint_sha256'],
               'frozen_parameter_tensors': len(frozen), 'frozen_parameters_unchanged': True,
               'trainable_parameter_tensors': len(trainable), 'validation_rows': 0,
               'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
               'preflight_weight_discarded': args.stage == 'preflight', 'checkpoint_selection': False}
    if args.stage == 'fit':
        receipt['checkpoint_path'] = str(Path(spec['state_root']) / 'latest.pth')
        receipt['checkpoint_sha256'] = sha(receipt['checkpoint_path'])
    with (output / 'receipt.json').open('x') as f:
        json.dump(receipt, f, indent=2, allow_nan=False)
    print('EG_ADAPT_COMPLETE ' + json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()

