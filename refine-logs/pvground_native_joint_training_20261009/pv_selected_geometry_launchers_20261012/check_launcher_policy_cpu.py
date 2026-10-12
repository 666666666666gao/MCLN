"""Pure parser and extracted train-loop diagnostics fixtures; no neural execution."""
import argparse
import ast
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

root = Path(__file__).resolve().parent
path = root / 'source_overlay_v3/main_utils.py'
tree = ast.parse(path.read_bytes().decode('utf-8'))
parse = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'parse_option')
scope = dict(argparse=argparse)
exec(compile(ast.fix_missing_locations(ast.Module(body=[parse])), str(path), 'exec'), scope)
arguments = []
for arm in ('native_control', 'selected_geometry'):
    protocol = json.loads((root / (arm + '_protocol.json')).read_bytes())
    sys.argv = ['train_dist_mod.py'] + protocol['common_arguments'] + [
        '--native_init_spec', protocol['native_init_spec'], '--log_dir', '/isolated/logs', '--exp', arm]
    args = scope['parse_option']()
    assert args.model == 'PVGround' and args.dataset == ['scanrefer'] and args.test_dataset == 'scanrefer'
    assert args.num_target == 256 and args.num_decoder_layers == 6 and args.num_encoder_layers == 3
    assert args.max_epoch == 8 and args.start_epoch == 1 and args.val_freq == 1
    assert args.batch_size == 8 and args.rng_seed == 2027
    assert args.lr == args.lr_backbone == 1e-6 and args.lr_modules == 1e-5
    assert args.weight_decay == .0005 and args.clip_norm == .1
    assert args.butd and not args.butd_gt and not args.butd_cls
    assert not args.joint_det and not args.detect_intermediate and not args.augment_det
    assert args.use_soft_token_loss and args.use_contrastive_align and not args.checkpoint_path
    arguments.append(dict(arm=arm, source_path=protocol['model_source'], parser_pass=True))
base = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'BaseTrainTester')
methods = [node for node in base.body if isinstance(node, ast.FunctionDef)
           and node.name in ('train_one_epoch', '_accumulate_stats')]
assert len(methods) == 2
base.body = methods
scope = dict(os=os, json=json, tqdm=lambda value, **unused: value,
             dist=SimpleNamespace(get_rank=lambda: 0))
exec(compile(ast.fix_missing_locations(ast.Module(body=[base])), str(path), 'exec'), scope)

class Scalar:
    def item(self):
        return 1.0
    def backward(self):
        pass  # Engineering fixture, not autograd.

class Model:
    def __init__(self, enabled):
        self.module = SimpleNamespace(use_selected_geometry_supervision=enabled)
    def train(self):
        pass
    def __call__(self, batch):
        return {}

def compute_loss(points, criterion, set_criterion, args):
    scalar = Scalar()
    points['loss'] = scalar
    points['selected_geometry_extra_record'] = points['fixture_record']
    return scalar, points

rows = [dict(role=role) for role in ('matched_root', 'unmatched', 'unmatched', 'matched_other')]
record = dict(rows=rows, extra_rows=1, loss=.125,
              selected_roles=dict(matched_root=1, unmatched=2, matched_other=1))
fixture = [{'fixture_record': record}, {'fixture_record': record}]
results = []
for enabled in (False, True):
    directory = root / ('diagnostic_fixture_strategy' if enabled else 'diagnostic_fixture_control')
    assert not directory.exists()
    directory.mkdir()
    tester = scope['BaseTrainTester'].__new__(scope['BaseTrainTester'])
    tester._to_gpu = lambda value: dict(value)
    tester._get_inputs = lambda value, training: value
    tester._compute_loss = compute_loss
    tester.logger = SimpleNamespace(info=lambda value: None)
    tester.tensorboard = SimpleNamespace(item=dict(train_loss=dict(loss=0.0), train_lr={}),
                                         dump_tensorboard=lambda *unused: None)
    optimizer = SimpleNamespace(zero_grad=lambda: None, step=lambda: None,
        param_groups=[dict(lr=1e-6), dict(lr=1e-6)])
    scheduler = SimpleNamespace(step=lambda: None)
    args = SimpleNamespace(clip_norm=0, print_freq=128, log_dir=str(directory))
    tester.train_one_epoch(1, fixture, Model(enabled), None, None, optimizer, scheduler, args)
    output = directory / 'selected_geometry_responsibility.jsonl'
    assert output.exists() == enabled
    if enabled:
        lines = output.read_text().splitlines()
        assert len(lines) == 1
        value = json.loads(lines[0])
        assert value == dict(epoch=1, batches=2, rows=8, extra_rows=2, extra_loss_sum=.25,
                            matched_root=2, matched_other=2, unmatched=4)
    results.append(dict(strategy_enabled=enabled, diagnostic_created=output.exists()))
result = dict(status='SOURCE_ARGUMENT_PARSER_AND_DIAGNOSTIC_STUB_PASS', argument_checks=arguments,
    diagnostic_fixtures=results, neural_calls=0, GPU_calls=0, optimizer_updates=0,
    actual_autograd=False, actual_dataset=False, actual_training_loop=False,
    scope='extracted original parser plus actual diagnostic branch under explicit stubs')
(root / 'LAUNCHER_ARGUMENT_AND_DIAGNOSTIC_CPU_CHECK.json').write_bytes((json.dumps(result, indent=2) + '\n').encode())
print(json.dumps(dict(status=result['status'], neural_calls=0, actual_training_loop=False)))
