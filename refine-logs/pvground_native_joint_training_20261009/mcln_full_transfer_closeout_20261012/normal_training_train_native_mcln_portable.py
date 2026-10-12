"""Full native MCLN epochs with the integrated portable A/B forward path."""
import argparse
import datetime
import json
import os
from pathlib import Path
import random
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True)
parser.add_argument('--spec', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--max-epochs', type=int, required=True)
options = parser.parse_args()
assert options.max_epochs == 8
source = Path(options.source)
output = Path(options.output)
assert not output.exists()
output.mkdir(parents=True)
os.chdir(str(source))
sys.path.insert(0, str(source))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from main_utils import parse_option, _atomic_torch_save
from train_dist_mod import TrainTester
from utils import get_scheduler
from mcln_transfer_initialization import checked_payload, load_mcln_core, install_portable_states
from native_decline_policy import decline_decision

spec = json.loads(Path(options.spec).read_bytes())
assert spec['seed'] == 2027 and spec['dataset'] == 'scanrefer'
random.seed(2027)
np.random.seed(2027)
torch.manual_seed(2027)
torch.cuda.manual_seed_all(2027)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
core = checked_payload(spec['mcln_checkpoint'], spec['mcln_checkpoint_sha256'])
sys.argv = [str(source / 'train_dist_mod.py')]
args = parse_option()
vars(args).update(vars(core['config']))
args.data_root = spec['data_root']
args.eval = args.debug = args.eval_train = False
args.batch_size = 8
args.num_workers = 4
args.rng_seed = 2027
args.use_cs_mcln = False
args.use_source_choice_selector = False
args.source_choice_selector_train_only = False
args.eval_use_selector_choice_scores = False
args.use_source_moe = False
args.source_moe_train_only = args.source_moe_gate_train_only = False
args.pp_checkpoint = args.checkpoint_path = None
args.frozen = args.small_lr = False
args.lr = args.lr_backbone = args.text_encoder_lr = 1e-6
args.weight_decay = 5e-4
args.clip_norm = .1
args.lr_scheduler = 'step'
args.lr_decay_epochs = [280, 340]
args.lr_decay_rate = .1
args.warmup_epoch = -1
args.start_epoch = 1
args.max_epoch = options.max_epochs
args.max_train_batches = 0
args.gradient_accumulation_steps = 1
args.drop_incomplete_accumulation_group = False
args.val_freq = 1
args.print_freq = 200
args.log_dir = str(output / 'native_logs')
args.exp = 'mcln_portable_full_transfer'
args.local_rank = 0
args.mask_loss_scale = 1.0 if args.mask_loss_scale is None else args.mask_loss_scale
args.consistency_loss_scale = 1.0 if args.consistency_loss_scale is None else args.consistency_loss_scale
assert args.model == 'MCLN' and args.dataset == ['scanrefer'] and args.test_dataset == 'scanrefer'
assert args.butd and not args.butd_cls and not args.butd_gt

torch.cuda.set_device(0)
dist.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=5400))


class PortableTrainTester(TrainTester):
    def _build_grounding_evaluator(self, current_args, prefixes):
        evaluator = super()._build_grounding_evaluator(current_args, prefixes)
        self.portable_evaluator = evaluator
        return evaluator

    def evaluate_one_epoch(self, epoch, test_loader, model, criterion, set_criterion, current_args):
        super().evaluate_one_epoch(epoch, test_loader, model, criterion, set_criterion, current_args)
        evaluator = self.portable_evaluator
        row = dict(epoch=epoch, rows=9508, primary_score='last/bbs')
        for threshold, suffix in ((.25, '025'), (.5, '050')):
            key = ('last_', threshold, 1, 'bbs')
            assert int(evaluator.gts[key]) == 9508
            hits = int(evaluator.dets[key])
            row['hits' + suffix] = hits
            row['acc' + suffix + '_percent'] = 100 * hits / 9508
        row['completed_cst'] = datetime.datetime.now().astimezone().isoformat()
        return row


tester = PortableTrainTester(args)
train_loader, test_loader = tester.get_loaders(args)
train_loader.generator.manual_seed(2027)
test_loader.generator.manual_seed(2027)
train_loader.sampler.seed = 2027
assert len(test_loader.dataset) == 9508 and len(train_loader) > 0
model = TrainTester.get_model(args)
load_mcln_core(model, core)
del core
install_portable_states(model, spec)
model.portable_geometry_arm = 'full_transfer'
model = model.cuda()
assert not any(parameter.requires_grad for parameter in model.text_encoder.parameters())
optimizer = TrainTester.get_optimizer(args, model)
new_parameters = [parameter for name, parameter in model.named_parameters()
                  if name.startswith(('candidate_support_corrector.', 'candidate_span_mixer.'))]
new_ids = {id(parameter) for parameter in new_parameters}
assert sum(parameter.numel() for parameter in new_parameters) == 57634
for group in optimizer.param_groups:
    group['params'] = [parameter for parameter in group['params'] if id(parameter) not in new_ids]
optimizer.add_param_group(dict(params=new_parameters, lr=1e-5))
optimizer_ids = [id(parameter) for group in optimizer.param_groups for parameter in group['params']]
assert len(set(optimizer_ids)) == len(optimizer_ids)
assert set(optimizer_ids) == {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
scheduler = get_scheduler(optimizer, len(train_loader), args)
criterion, set_criterion = TrainTester.get_criterion(args)
model = DistributedDataParallel(model, device_ids=[0], broadcast_buffers=False, find_unused_parameters=True)
history = []
best = None
total_updates = 0
progress = dict(status='running', max_epochs=8, actual_training_dataset_rows=len(train_loader.dataset),
                training_batches_per_epoch=len(train_loader), training_batch_size=8,
                loader_generator_seed=2027, train_sampler_seed=train_loader.sampler.seed,
                actual_training_flags={name: getattr(args, name) for name in ('joint_det', 'detect_intermediate', 'augment_det', 'butd')},
                checkpoint_history='E71 core plus PV-trained A/B; fresh optimizer; no engineering-check state',
                PV_best_unchanged=True, full_goal_complete=False)


def save_completed_state(epoch, metrics):
    global best
    state = dict(model=model.module.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                 config=args, spec=spec, epoch=epoch, optimizer_updates=total_updates, history=history,
                 rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch_cpu=torch.get_rng_state(),
                          torch_cuda=torch.cuda.get_rng_state_all(), train_loader=train_loader.generator.get_state(),
                          test_loader=test_loader.generator.get_state()))
    _atomic_torch_save(state, str(output / 'latest.pth'))
    rank = (metrics['hits050'], metrics['hits025'])
    if best is None or rank > (best['hits050'], best['hits025']):
        best = metrics
        _atomic_torch_save(dict(model=model.module.state_dict(), config=args, spec=spec,
                                epoch=epoch, metrics=metrics, optimizer_updates=total_updates), str(output / 'best.pth'))
    decision = decline_decision(history)
    progress.update(epoch=epoch, metrics=metrics, best=best, early_stop=decision,
                    optimizer_updates=total_updates, updated_cst=datetime.datetime.now().astimezone().isoformat())
    (output / 'PROGRESS.json').write_text(json.dumps(progress, indent=2) + '\n')
    return decision


metrics = tester.evaluate_one_epoch(0, test_loader, model, criterion, set_criterion, args)
history.append(metrics)
with (output / 'native_metrics.jsonl').open('a') as stream:
    stream.write(json.dumps(metrics) + '\n')
save_completed_state(0, metrics)
for epoch in range(1, options.max_epochs + 1):
    train_loader.sampler.set_epoch(epoch)
    started = time.monotonic()
    receipt = tester.train_one_epoch(epoch, train_loader, model, criterion, set_criterion, optimizer, scheduler, args)
    assert receipt['batch_count'] == len(train_loader)
    total_updates += len(train_loader)
    _atomic_torch_save(dict(model=model.module.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                           config=args, spec=spec, epoch=epoch, optimizer_updates=total_updates,
                           validation_pending=True, rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                           torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state_all(),
                           train_loader=train_loader.generator.get_state(), test_loader=test_loader.generator.get_state())),
                       str(output / 'latest.pth'))
    (output / ('train_epoch_' + str(epoch) + '.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    metrics = tester.evaluate_one_epoch(epoch, test_loader, model, criterion, set_criterion, args)
    metrics['epoch_training_and_validation_seconds'] = time.monotonic() - started
    history.append(metrics)
    with (output / 'native_metrics.jsonl').open('a') as stream:
        stream.write(json.dumps(metrics) + '\n')
    decision = save_completed_state(epoch, metrics)
    print(json.dumps(dict(metrics=metrics, early_stop=decision, best=best)), flush=True)
    if decision['stop']:
        break
progress.update(status='complete', stop_reason='two_consecutive_strict_declines' if decision['stop'] else 'maximum_epoch_budget',
                formal_training_completed=True, full_goal_complete=False,
                completed_cst=datetime.datetime.now().astimezone().isoformat())
(output / 'RESULT.json').write_text(json.dumps(progress, indent=2) + '\n')
(output / 'PROGRESS.json').write_text(json.dumps(progress, indent=2) + '\n')
dist.destroy_process_group()
