"""Fixed one-pass EG Sr3D adaptation; author model/losses, new optimizer, no validation selection."""
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
    output = root / args.stage
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
    dataset = Joint3DDataset(dataset_dict={'sr3d': 1, 'scannet': 10}, test_dataset='sr3d',
                            split='train', data_path=spec['data_root'], use_color=True,
                            detect_intermediate=True, butd_cls=True)
    assert len(dataset) == spec['fit_rows'] == 77836 and dataset.augment
    if args.stage == 'preflight':
        order = spec['preflight_indices']
    else:
        order = np.load(spec['order_path']).tolist()
        assert len(order) == len(dataset) and sorted(order) == list(range(len(dataset)))
    loader = DataLoader(Subset(dataset, order), batch_size=8, shuffle=False,
                        num_workers=0, drop_last=False)
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
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
        assert torch.isfinite(grad_norm)
        optimizer.step()
        steps += 1
        rows += len(batch['utterances'])
        record = {'step': steps, 'rows': rows, 'loss': float(loss.detach()),
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
                     'optimizer': optimizer.state_dict(), 'epoch': 1, 'step': steps,
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
    assert steps == (2 if args.stage == 'preflight' else 9730)
    assert rows == (16 if args.stage == 'preflight' else 77836)
    params = dict(model.named_parameters())
    assert all(torch.equal(params[n].detach().cpu(), value) for n, value in frozen.items())
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    receipt = {'status': 'complete', 'stage': args.stage, 'time_cst': now(),
               'rows': rows, 'optimizer_steps': steps, 'elapsed_seconds': time.time() - start,
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

