"""PV-Ground author initialization plus CS modules, without inference sidechains."""

import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch


SEED = 2027
EPOCHS = 21
CS_PREFIXES = ('cs_structure.', 'cs_context_reader.', 'cs_box_refiner.',
               'cs_geometry_readback.')


def set_seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(str(temporary), str(path))


def load_parent(model, source):
    target = model.state_dict()
    position = model.text_encoder.embeddings.position_ids
    expected = torch.arange(model.text_encoder.config.max_position_embeddings).expand((1, -1))
    assert torch.equal(position, expected)
    assert 'text_encoder.embeddings.position_ids' not in source
    model.text_encoder.embeddings.register_buffer('position_ids', position, persistent=False)
    target = model.state_dict()
    additions = {name for name in target if name.startswith(CS_PREFIXES)}
    assert len(source) == 1234
    assert set(target) - set(source) == additions
    assert not set(source) - set(target)
    for name, value in source.items():
        assert target[name].shape == value.shape and target[name].dtype == value.dtype, name
    # Exact merged state: every learned parent tensor and every declared new tensor.
    merged = dict(source)
    merged.update({name: target[name] for name in additions})
    model.load_state_dict(merged, strict=True)
    return {'core_tensors': len(source), 'new_tensors': len(additions)}


def groups_and_rates(model, batch_size):
    groups = {'new': [], 'core': [], 'backbone': []}
    rates = {name: rate * math.sqrt(batch_size / 16.0)
             for name, rate in (('new', 1e-4), ('core', 2e-5), ('backbone', 2e-6))}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group = 'new' if name.startswith(CS_PREFIXES) else (
                'backbone' if name.startswith('backbone_net.') else 'core')
            groups[group].append(parameter)
    assert groups['core'] and groups['backbone']
    return [dict(params=values, lr=rates[name], name=name)
            for name, values in groups.items() if values], rates


def delta_names(model):
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    names.update(name for name, _ in model.named_buffers() if not name.startswith('text_encoder.'))
    return names


def save_delta(path, model, optimizer, epoch, metrics, args, initial_load):
    state = model.state_dict()
    payload = {'epoch': epoch, 'arm': args.arm, 'batch_size': args.batch_size,
               'seed': SEED, 'parent_checkpoint': str(args.checkpoint),
               'initial_load': initial_load, 'metrics': metrics,
               'state_delta': {name: state[name].detach().cpu() for name in delta_names(model)}}
    if optimizer is not None:
        payload['optimizer'] = optimizer.state_dict()
    temporary = path.with_suffix('.pth.tmp')
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(path))
    return path.stat().st_size


def restore_delta(model, optimizer, path, args, initial_load):
    payload = torch.load(str(path), map_location='cpu')
    assert payload['arm'] == args.arm and payload['batch_size'] == args.batch_size
    assert payload['seed'] == SEED and payload['parent_checkpoint'] == str(args.checkpoint)
    assert payload['initial_load'] == initial_load
    delta = payload['state_delta']
    assert set(delta) == delta_names(model)
    state = model.state_dict()
    for name, value in delta.items():
        assert state[name].shape == value.shape and state[name].dtype == value.dtype, name
    state.update(delta)
    model.load_state_dict(state, strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload['optimizer'])
    return payload['epoch'], payload['metrics']


def best_rank(metrics):
    return (min(metrics['hits025'] / 5572.0, metrics['hits050'] / 4797.0),
            metrics['hits025'] + metrics['hits050'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('native', 'cs', 'cs_readback'), required=True)
    parser.add_argument('--mode', choices=('cpu', 'preflight', 'train'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset-source', type=Path, required=True)
    parser.add_argument('--model-source', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Reuse the corrected project dataset, but load the actual PV-Ground model/loss.
    os.chdir(str(args.dataset_source))
    sys.path.insert(0, str(args.dataset_source))
    from src.joint_det_dataset import Joint3DDataset
    assert 'models' not in sys.modules
    os.chdir(str(args.model_source))
    sys.path.insert(0, str(args.model_source))
    from models.pv_ground import PVGround
    from main_utils import BaseTrainTester, parse_option
    from prepare_data import DataProcessor
    from pcdet.config import cfg, cfg_from_yaml_file
    assert Path(sys.modules['models.pv_ground'].__file__).resolve() == (args.model_source / 'models/pv_ground.py').resolve()
    assert Path(sys.modules['src.joint_det_dataset'].__file__).resolve() == (args.dataset_source / 'src/joint_det_dataset.py').resolve()
    evaluator_path = args.runtime / 'PV-Ground/src/grounding_evaluator.py'
    spec = importlib.util.spec_from_file_location('official_pvground_evaluator', str(evaluator_path))
    evaluator_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator_module)
    GroundingEvaluator = evaluator_module.GroundingEvaluator

    parent = torch.load(str(args.checkpoint), map_location='cpu')
    assert all(name.startswith('module.') for name in parent['model'])
    source = {name[7:]: value for name, value in parent['model'].items()}
    sys.argv = [sys.argv[0]]
    config = parse_option()
    vars(config).update(vars(parent['config']))
    config.model = 'PVGround'
    config.data_root = str(args.data_root).rstrip('/') + '/'
    config.eval = False
    config.eval_train = False
    config.debug = False
    config.dataset = ['scanrefer']
    config.test_dataset = 'scanrefer'
    config.joint_det = True
    config.detect_intermediate = True
    config.augment_det = True
    assert config.butd and not config.butd_cls and not config.butd_gt
    assert config.use_color and not config.use_height and not config.use_multiview
    assert config.use_soft_token_loss and config.use_contrastive_align
    assert config.num_target == 256 and config.num_decoder_layers == 6
    cfg_from_yaml_file(str(args.model_source / 'wandb_config.yaml'), cfg)

    def build_model(arm):
        previous_directory = Path.cwd()
        os.chdir(str(args.model_source))
        result = PVGround(copy.deepcopy(cfg), num_class=256, num_queries=256, num_decoder_layers=6,
                        self_position_embedding=config.self_position_embedding,
                        contrastive_align_loss=True, butd=True, pointnet_ckpt=None,
                        data_path=config.data_root, self_attend=config.self_attend,
                        use_cs=arm != 'native', use_readback=arm == 'cs_readback')
        os.chdir(str(previous_directory))
        return result

    model = build_model(args.arm)
    initial_load = load_parent(model, source)
    assert bool(initial_load['new_tensors']) == (args.arm != 'native')
    assert (model.cs_geometry_readback is not None) == (args.arm == 'cs_readback')
    del parent, source
    config_record = {'arm': args.arm, 'mode': args.mode, 'seed': SEED,
                     'checkpoint': str(args.checkpoint), 'initial_load': initial_load,
                     'model': 'PVGround', 'batch_size': args.batch_size,
                     'G': False, 'quality': False, 'teacher': False,
                     'inference_sidechains': False}
    if args.mode == 'cpu':
        atomic_json(args.output / 'cpu_load.json', dict(config_record, status='pass',
                                                      gpu_forwards=0, optimizer_steps=0))
        print('CPU_LOAD_PASS', json.dumps(config_record), flush=True)
        return

    model.cuda()
    groups, rates = groups_and_rates(model, args.batch_size)
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0005)
    criterion, set_criterion = BaseTrainTester.get_criterion(config)
    processors = {training: DataProcessor(cfg.DATA_PROCESSOR,
                  np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE), training, 6)
                  for training in (True, False)}
    os.chdir(str(args.dataset_source))
    common = dict(dataset_dict={'scanrefer': 1, 'scannet': 10}, test_dataset='scanrefer',
                  use_color=True, use_height=False, use_multiview=False,
                  data_path=config.data_root, detect_intermediate=True,
                  butd=True, butd_gt=False, butd_cls=False)
    train = Joint3DDataset(split='train', augment_det=True, **common)
    validation = Joint3DDataset(split='val', **common)
    assert len(validation) == 9508
    config_record.update(train_samples=len(train), validation_samples=len(validation), rates=rates)
    atomic_json(args.output / 'config.json', config_record)
    print('CONFIG', json.dumps(config_record), flush=True)

    def prepare(raw, training):
        rows = [processors[training].forward({'points': pc.numpy().copy(), 'use_lead_xyz': True})
                for pc in raw['point_clouds']]
        voxels = processors[training].collate_batch(rows)
        bs = len(raw['utterances'])
        assert np.array_equal(voxels['points'][:, 1:].reshape(bs, 50000, 6), raw['point_clouds'].numpy())
        batch = {name: value.cuda(non_blocking=True) if torch.is_tensor(value) else value
                 for name, value in raw.items()}
        # Native PV mask matching uses a Boolean complement; project masks are 0/1 integers.
        assert bool(((batch['gt_masks'] == 0) | (batch['gt_masks'] == 1)).all())
        batch['gt_masks'] = batch['gt_masks'].bool()
        inputs = {name: torch.from_numpy(voxels[name]).float().cuda()
                  for name in ('points', 'voxels', 'voxel_coords', 'voxel_num_points')}
        inputs.update(batch_size=bs, text=batch['utterances'],
                      det_boxes=batch['all_detected_boxes'],
                      det_bbox_label_mask=batch['all_detected_bbox_label_mask'],
                      det_class_ids=batch['all_detected_class_ids'], superpoint=batch['superpoint'])
        return inputs, batch

    def forward_loss(raw, training):
        inputs, batch = prepare(raw, training)
        outputs = model(inputs)
        assert not set(outputs).intersection(batch)
        outputs.update(batch)
        return BaseTrainTester._compute_loss(outputs, criterion, set_criterion, config)

    def evaluate(epoch):
        model.eval()
        set_seed(SEED)
        evaluator = GroundingEvaluator(only_root=True, thresholds=[0.25, 0.5], topks=[1],
                                      prefixes=['last_'], filter_non_gt_boxes=False, model='PVGround')
        started = time.time()
        loader = torch.utils.data.DataLoader(validation, batch_size=args.batch_size,
                                            shuffle=False, num_workers=args.num_workers)
        with torch.no_grad():
            for step, raw in enumerate(loader, 1):
                inputs, batch = prepare(raw, False)
                outputs = model(inputs)
                assert not set(outputs).intersection(batch)
                outputs.update(batch)
                outputs['last_pred_size'] = outputs['last_pred_size'].clamp(min=1e-6)
                evaluator.evaluate(outputs, 'last_')
                if step % 200 == 0:
                    print('EVAL_PROGRESS', epoch, step, len(loader), time.time() - started, flush=True)
        count = int(evaluator.gts[('last_', 0.25, 1, 'bbs')])
        assert count == 9508
        hits025 = int(evaluator.dets[('last_', 0.25, 1, 'bbs')])
        hits050 = int(evaluator.dets[('last_', 0.5, 1, 'bbs')])
        return {'samples': count, 'hits025': hits025, 'hits050': hits050,
                'acc025': hits025 / count, 'acc050': hits050 / count,
                'seconds': time.time() - started}

    if args.mode == 'preflight':
        assert not args.resume and args.arm != 'native'
        native = build_model('native')
        parent = torch.load(str(args.checkpoint), map_location='cpu')
        load_parent(native, {name[7:]: value for name, value in parent['model'].items()})
        del parent
        native.cuda().eval()
        model.eval()
        raw = next(iter(torch.utils.data.DataLoader(validation, batch_size=1, num_workers=0)))
        inputs, _ = prepare(raw, False)
        with torch.no_grad():
            set_seed(SEED)
            left = native(inputs)
            set_seed(SEED)
            right = model(inputs)
        differences = {}
        for name in ('last_center', 'last_pred_size', 'last_sem_cls_scores'):
            differences[name] = float((left[name] - right[name]).abs().max())
            assert torch.allclose(left[name], right[name], rtol=1e-5, atol=1e-5), name
        for name in ('sp_last_pred_masks', 'last_pred_masks'):
            differences[name] = max(float((a - b).abs().max()) for a, b in zip(left[name], right[name]))
            assert differences[name] <= 1e-5, name
        del native, left, right, raw, inputs
        torch.cuda.empty_cache()
        model.train()
        set_seed(SEED)
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        gradients = []
        loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True,
                    num_workers=0, generator=torch.Generator().manual_seed(SEED))
        parameters = dict(model.named_parameters())
        terminal_names = ['cs_structure.seed_delta.2.weight', 'cs_structure.super_delta.weight',
                          'cs_context_reader.delta.weight', 'cs_box_refiner.delta.2.weight']
        internal_names = ['cs_structure.seed_delta.0.weight',
                          'cs_context_reader.global_attention.in_proj_weight',
                          'cs_context_reader.local_attention.in_proj_weight', 'cs_box_refiner.delta.0.weight']
        internal_names += ['cs_structure.source_encoders.%d.0.weight' % i for i in range(8)]
        if args.arm == 'cs_readback':
            terminal_names += ['cs_geometry_readback.output.weight']
            internal_names += ['cs_geometry_readback.coarse_encoder.weight',
                               'cs_geometry_readback.support_encoder.weight',
                               'cs_geometry_readback.refined_encoder.weight',
                               'cs_geometry_readback.encode.0.weight',
                               'cs_geometry_readback.set_attention.in_proj_weight']
        for step, raw in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            loss, outputs = forward_loss(raw, True)
            assert bool(torch.isfinite(loss))
            if step == 2:
                mask_gradient = torch.autograd.grad(outputs['last_center'].sum()
                    + outputs['last_pred_size'].sum(), model.x_query[0].weight, retain_graph=True)[0]
                assert bool(mask_gradient.abs().sum() > 0)
            loss.backward()
            norms = {name: float(parameters[name].grad.norm())
                     for name in terminal_names + internal_names}
            assert all(math.isfinite(value) for value in norms.values())
            assert all(norms[name] > 0 for name in terminal_names)
            if step == 2:
                assert all(norms[name] > 0 for name in internal_names)
            gradients.append(norms)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            assert bool(torch.isfinite(norm)) and float(norm) > 0
            optimizer.step()
            print('PREFLIGHT_STEP', step, float(loss), float(norm), flush=True)
            if step == 2:
                break
        torch.cuda.synchronize()
        seconds = time.time() - started
        del outputs, loss, raw
        path = args.output / 'preflight_delta.pth'
        checkpoint_bytes = save_delta(path, model, optimizer, 0, {}, args, initial_load)
        reference = {name: model.state_dict()[name].detach().cpu().clone() for name in delta_names(model)}
        probe_parameter = parameters[terminal_names[0]]
        optimizer_reference = optimizer.state[probe_parameter]['exp_avg'].cpu().clone()
        with torch.no_grad():
            probe_parameter.add_(1)
            optimizer.state[probe_parameter]['exp_avg'].add_(1)
        restore_delta(model, optimizer, path, args, initial_load)
        assert all(torch.equal(model.state_dict()[name].cpu(), value) for name, value in reference.items())
        assert torch.equal(optimizer.state[probe_parameter]['exp_avg'].cpu(), optimizer_reference)
        atomic_json(args.output / 'preflight.json', dict(config_record, status='pass',
                    optimizer_steps=2, checkpoint_bytes=checkpoint_bytes,
                    seconds_two_steps_including_checks=seconds,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    zero_update_max_abs_differences=differences, gradients=gradients,
                    delta_model_optimizer_reload=True))
        print('PREFLIGHT_PASS', flush=True)
        return

    first_epoch = 1
    best = None
    if args.resume:
        epoch, _ = restore_delta(model, optimizer, args.output / 'latest.pth', args, initial_load)
        first_epoch = epoch + 1
        assert first_epoch <= EPOCHS
        best = json.loads((args.output / 'best.json').read_text(encoding='utf-8'))
    else:
        assert not (args.output / 'latest.pth').exists()
        assert not (args.output / 'best.pth').exists()
        baseline = evaluate(0)
        atomic_json(args.output / 'epoch_0.json', baseline)
        print('EVAL', 0, json.dumps(baseline), flush=True)
    for epoch in range(first_epoch, EPOCHS + 1):
        set_seed(SEED + epoch)
        model.train()
        loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, generator=torch.Generator().manual_seed(SEED + epoch))
        factor = 0.1 if epoch == 1 else 0.5 * (1 + math.cos(math.pi * (epoch - 2) / (EPOCHS - 1)))
        for group in optimizer.param_groups:
            group['lr'] = rates[group['name']] * (1.0 if group['name'] == 'new' and epoch == 1 else factor)
        started = time.time()
        for step, raw in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = forward_loss(raw, True)
            assert bool(torch.isfinite(loss))
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            assert bool(torch.isfinite(norm))
            optimizer.step()
            if step == 1 or step % 500 == 0:
                print('TRAIN', args.arm, epoch, step, len(loader), float(loss), time.time() - started, flush=True)
        result = {'epoch': epoch, 'train_seconds': time.time() - started,
                  'steps': len(loader), 'samples': len(train), 'validation': evaluate(epoch)}
        atomic_json(args.output / ('epoch_%d.json' % epoch), result)
        if best is None or best_rank(result['validation']) > best_rank(best['metrics']):
            best = {'epoch': epoch, 'metrics': result['validation']}
            save_delta(args.output / 'best.pth', model, None, epoch, best['metrics'], args, initial_load)
            atomic_json(args.output / 'best.json', best)
            print('BEST', json.dumps(best), flush=True)
        saved = save_delta(args.output / 'latest.pth', model, optimizer, epoch,
                           result['validation'], args, initial_load)
        print('EPOCH', json.dumps(result), 'CHECKPOINT_BYTES', saved, flush=True)
    atomic_json(args.output / 'complete.json', {'epochs': EPOCHS, 'best': best})


if __name__ == '__main__':
    main()
