"""Paired native/CS-MCLN ScanRefer training from the protected E71 core."""

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from main_utils import parse_option
from train_dist_mod import TrainTester
from src.grounding_evaluator import GroundingEvaluator


SEED = 2027
EPOCHS = 21
CS_PREFIXES = ('cs_structure.', 'cs_context_reader.', 'cs_box_refiner.')


def set_seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    os.replace(str(temporary), str(path))


def experiment_args(config, arm, data_root):
    sys.argv = [sys.argv[0]]
    args = parse_option()
    vars(args).update(vars(config))
    args.data_root = str(data_root).rstrip('/') + '/'
    args.use_cs_mcln = arm == 'cs'
    args.use_source_choice_selector = False
    args.eval_use_selector_choice_scores = False
    args.use_source_moe = False
    args.pp_checkpoint = None
    args.mask_loss_scale = 1.0 if args.mask_loss_scale is None else args.mask_loss_scale
    args.consistency_loss_scale = (
        1.0 if args.consistency_loss_scale is None else args.consistency_loss_scale
    )
    assert args.model == 'MCLN'
    assert args.dataset == ['scanrefer'] and args.test_dataset == 'scanrefer'
    assert args.butd and not args.butd_cls and not args.butd_gt
    assert args.num_decoder_layers == 6 and args.num_target == 256
    assert args.joint_det and args.detect_intermediate
    return args


def load_exact_e71(model, checkpoint_state, arm):
    assert all(name.startswith('module.') for name in checkpoint_state)
    source = {name[7:]: value for name, value in checkpoint_state.items()}
    selector = [name for name in source if name.startswith('source_choice_selector.')]
    assert len(selector) == 9
    for name in selector:
        del source[name]
    target = model.state_dict()
    additions = {name for name in target if name.startswith(CS_PREFIXES)}
    assert (arm == 'cs') == bool(additions)
    assert set(target) - set(source) == additions
    assert set(source) - set(target) == set()
    for name, value in source.items():
        assert target[name].shape == value.shape, name
        assert target[name].dtype == value.dtype, name
    result = model.load_state_dict(source, strict=arm == 'native')
    assert set(result.missing_keys) == additions
    assert not result.unexpected_keys
    return {'core_tensors': len(source), 'removed_selector_tensors': len(selector),
            'new_tensors': len(additions)}


def parameter_groups(model, batch_size):
    scale = math.sqrt(batch_size / 16.0)
    grouped = {'new': [], 'core': [], 'backbone': []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(CS_PREFIXES):
            grouped['new'].append(parameter)
        elif name.startswith('backbone_net.'):
            grouped['backbone'].append(parameter)
        else:
            grouped['core'].append(parameter)
    assert grouped['core'] and grouped['backbone']
    rates = {'new': 1e-4 * scale, 'core': 2e-5 * scale,
             'backbone': 2e-6 * scale}
    return [dict(params=grouped[name], lr=rates[name], name=name)
            for name in ('new', 'core', 'backbone') if grouped[name]], rates


def batch_loss(model, raw, criterion, set_criterion, args, training):
    batch = TrainTester._to_gpu(raw)
    inputs = TrainTester._get_inputs(batch)
    inputs['train'] = training
    outputs = model(inputs)
    for key, value in batch.items():
        assert key not in outputs, key
        outputs[key] = value
    loss, outputs = TrainTester._compute_loss(
        outputs, criterion, set_criterion, args
    )
    return loss, outputs


def evaluate(model, dataset, criterion, set_criterion, args, batch_size):
    model.eval()
    evaluator = GroundingEvaluator(
        only_root=True, thresholds=[0.25, 0.5], topks=[1],
        prefixes=['last_'], filter_non_gt_boxes=False, model='MCLN',
        eval_use_selector_choice_scores=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    with torch.no_grad():
        for raw in loader:
            _, outputs = batch_loss(
                model, raw, criterion, set_criterion, args, False
            )
            outputs['last_pred_size'] = outputs['last_pred_size'].clamp(min=1e-6)
            evaluator.evaluate(outputs, 'last_')
    hits025 = int(evaluator.dets[('last_', 0.25, 1, 'bbs')])
    hits050 = int(evaluator.dets[('last_', 0.5, 1, 'bbs')])
    count = int(evaluator.gts[('last_', 0.25, 1, 'bbs')])
    assert count == 9508
    return {'samples': count, 'hits025': hits025, 'hits050': hits050,
            'acc025': hits025 / count, 'acc050': hits050 / count}


def checkpoint(path, model, optimizer, epoch, args, initial_load):
    payload = {'epoch': epoch, 'model': model.state_dict(),
               'optimizer': optimizer.state_dict(), 'arm': args.arm,
               'batch_size': args.batch_size, 'seed': SEED,
               'initial_load': initial_load}
    temporary = path.with_suffix('.tmp')
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(path))
    return path.stat().st_size


def best_rank(metrics):
    return (min(metrics['hits025'] / 5572.0,
                metrics['hits050'] / 4797.0),
            metrics['hits025'] + metrics['hits050'])


def save_best(path, model, epoch, metrics, args, initial_load):
    payload = {'epoch': epoch, 'model': model.state_dict(),
               'metrics': metrics, 'arm': args.arm,
               'batch_size': args.batch_size, 'seed': SEED,
               'initial_load': initial_load}
    temporary = path.with_suffix('.tmp')
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(path))
    return path.stat().st_size


def zero_update_check(cs_model, validation, parent_path, config):
    parent = torch.load(str(parent_path), map_location='cpu')
    native_args = experiment_args(parent['config'], 'native', config.data_root)
    native = TrainTester.get_model(native_args)
    load_exact_e71(native, parent['model'], 'native')
    del parent
    native = native.cuda().eval()
    cs_model.eval()
    raw = next(iter(torch.utils.data.DataLoader(
        validation, batch_size=1, shuffle=False, num_workers=0
    )))
    batch = TrainTester._to_gpu(raw)
    inputs = TrainTester._get_inputs(batch)
    inputs['train'] = False
    with torch.no_grad():
        native_output = native(inputs)
        cs_output = cs_model(inputs)
    differences = {}
    for name in ('last_center', 'last_pred_size', 'last_sem_cls_scores'):
        left, right = native_output[name], cs_output[name]
        differences[name] = float((left - right).abs().max())
        assert torch.allclose(left, right, rtol=1e-5, atol=1e-5), name
    for name in ('sp_last_pred_masks', 'last_pred_masks'):
        differences[name] = max(float((left - right).abs().max())
                                for left, right in zip(native_output[name],
                                                       cs_output[name]))
        assert differences[name] <= 1e-5, name
    del native, native_output, cs_output, inputs, batch, raw
    torch.cuda.empty_cache()
    cs_model.train()
    return differences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('native', 'cs'), required=True)
    parser.add_argument('--mode', choices=('preflight', 'train'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--best-dir', type=Path)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    assert args.batch_size in (4, 8, 12, 16)
    args.output.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    payload = torch.load(str(args.checkpoint), map_location='cpu')
    config = experiment_args(payload['config'], args.arm, args.data_root)
    model = TrainTester.get_model(config)
    initial_load = load_exact_e71(model, payload['model'], args.arm)
    del payload
    model = model.cuda()
    groups, rates = parameter_groups(model, args.batch_size)
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0005)
    criterion, set_criterion = TrainTester.get_criterion(config)
    train, validation = TrainTester.get_datasets(config)
    assert len(validation) == 9508
    print('CONFIG', json.dumps({'arm': args.arm, 'mode': args.mode,
          'train_samples': len(train), 'val_samples': len(validation),
          'batch_size': args.batch_size, 'rates': rates,
          'initial_load': initial_load}), flush=True)

    if args.mode == 'preflight':
        assert not args.resume
        zero_differences = (
            zero_update_check(model, validation, args.checkpoint, config)
            if args.arm == 'cs' else None
        )
        loader = torch.utils.data.DataLoader(
            train, batch_size=args.batch_size, shuffle=True, num_workers=0,
            generator=torch.Generator().manual_seed(SEED),
        )
        model.train()
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        for step, raw in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            loss, outputs = batch_loss(model, raw, criterion, set_criterion,
                                       config, True)
            assert bool(torch.isfinite(loss))
            if args.arm == 'cs' and step == 2:
                mask_to_box = torch.autograd.grad(
                    outputs['last_center'].sum()
                    + outputs['last_pred_size'].sum(),
                    model.x_query[0].weight,
                    retain_graph=True,
                )[0]
                assert bool(mask_to_box.abs().sum() > 0)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            assert bool(torch.isfinite(norm)) and float(norm) > 0
            if args.arm == 'cs':
                names = ('cs_structure.seed_delta.2.weight',
                         'cs_structure.super_delta.weight',
                         'cs_context_reader.delta.weight',
                         'cs_box_refiner.delta.2.weight')
                parameters = dict(model.named_parameters())
                if step == 1:
                    assert all(parameters[name].grad is not None and
                               bool(parameters[name].grad.abs().sum() > 0)
                               for name in names)
                if step == 2:
                    internal = (
                        'cs_structure.seed_delta.0.weight',
                        'cs_structure.source_encoders.0.0.weight',
                        'cs_structure.source_encoders.1.0.weight',
                        'cs_structure.source_encoders.2.0.weight',
                        'cs_structure.source_encoders.3.0.weight',
                        'cs_context_reader.global_attention.in_proj_weight',
                        'cs_context_reader.local_attention.in_proj_weight',
                        'cs_box_refiner.delta.0.weight',
                    )
                    assert all(parameters[name].grad is not None and
                               bool(parameters[name].grad.abs().sum() > 0)
                               for name in internal)
            optimizer.step()
            print('PREFLIGHT_STEP', step, float(loss), float(norm), flush=True)
            if step == 2:
                break
        torch.cuda.synchronize()
        two_step_seconds = time.time() - started
        checkpoint_bytes = checkpoint(args.output / 'preflight.pth', model,
                                      optimizer, 0, args, initial_load)
        os.remove(str(args.output / 'preflight.pth'))
        model_bytes = save_best(args.output / 'preflight_model.pth', model, 0,
                                {'hits025': 0, 'hits050': 0}, args,
                                initial_load)
        os.remove(str(args.output / 'preflight_model.pth'))
        atomic_json(args.output / 'preflight.json', {
            'arm': args.arm, 'batch_size': args.batch_size,
            'seconds_two_steps_including_diagnostics': two_step_seconds,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'checkpoint_bytes': checkpoint_bytes,
            'model_only_bytes': model_bytes,
            'initial_load': initial_load,
            'zero_update_max_abs_differences': zero_differences,
        })
        return

    assert args.best_dir is not None
    args.best_dir.mkdir(parents=True, exist_ok=True)
    first_epoch = 1
    if args.resume:
        if (args.output / 'latest.pth').exists():
            saved = torch.load(str(args.output / 'latest.pth'), map_location='cpu')
            assert saved['arm'] == args.arm and saved['batch_size'] == args.batch_size
            assert saved['seed'] == SEED and saved['initial_load'] == initial_load
            model.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            first_epoch = saved['epoch'] + 1
            assert first_epoch <= EPOCHS
            del saved
        else:
            assert (args.best_dir / 'best.pth').exists()
            baseline = evaluate(model, validation, criterion, set_criterion,
                                config, args.batch_size)
            atomic_json(args.output / 'epoch_0.json', baseline)
        best_payload = torch.load(str(args.best_dir / 'best.pth'),
                                  map_location='cpu')
        assert best_payload['arm'] == args.arm
        best = {'arm': best_payload['arm'], 'epoch': best_payload['epoch'],
                'metrics': best_payload['metrics']}
        del best_payload
        atomic_json(args.best_dir / 'best.json', best)
    else:
        assert not (args.output / 'latest.pth').exists()
        assert not (args.best_dir / 'best.pth').exists()
        baseline = evaluate(model, validation, criterion, set_criterion,
                            config, args.batch_size)
        atomic_json(args.output / 'epoch_0.json', baseline)
        print('EVAL', 0, json.dumps(baseline), flush=True)
        best = None
    for epoch in range(first_epoch, EPOCHS + 1):
        set_seed(SEED + epoch)
        model.train()
        loader = torch.utils.data.DataLoader(
            train, batch_size=args.batch_size, shuffle=True, num_workers=4,
            generator=torch.Generator().manual_seed(SEED + epoch),
        )
        progress = (epoch - 2) / (EPOCHS - 1)
        factor = 0.1 if epoch == 1 else 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group['lr'] = rates[group['name']] * (1.0 if group['name'] == 'new'
                                                and epoch == 1 else factor)
        started = time.time()
        for step, raw in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = batch_loss(model, raw, criterion, set_criterion,
                                 config, True)
            assert bool(torch.isfinite(loss))
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            assert bool(torch.isfinite(norm))
            optimizer.step()
            if step % 500 == 0:
                print('TRAIN', args.arm, epoch, step, len(loader),
                      float(loss), time.time() - started, flush=True)
        result = {'epoch': epoch, 'seconds': time.time() - started,
                  'steps': len(loader), 'samples': len(train)}
        result['validation'] = evaluate(
            model, validation, criterion, set_criterion,
            config, args.batch_size,
        )
        atomic_json(args.output / ('epoch_%d.json' % epoch), result)
        if best is None or best_rank(result['validation']) > best_rank(best['metrics']):
            best = {'arm': args.arm, 'epoch': epoch,
                    'metrics': result['validation']}
            best_bytes = save_best(args.best_dir / 'best.pth', model, epoch,
                                   result['validation'], args, initial_load)
            atomic_json(args.best_dir / 'best.json', best)
            print('BEST', args.arm, epoch, best_bytes,
                  json.dumps(result['validation']), flush=True)
        saved = checkpoint(args.output / 'latest.pth', model, optimizer,
                           epoch, args, initial_load)
        print('EPOCH', args.arm, json.dumps(result), 'CHECKPOINT_BYTES', saved,
              flush=True)


if __name__ == '__main__':
    main()
