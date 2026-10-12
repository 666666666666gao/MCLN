"""Two real native MCLN updates and saved same-process full-state recovery."""
import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import random
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True)
parser.add_argument('--spec', required=True)
parser.add_argument('--output', required=True)
options = parser.parse_args()
source = Path(options.source)
output = Path(options.output)
assert not output.exists()
output.mkdir(parents=True)
os.chdir(str(source))
sys.path.insert(0, str(source))

import numpy as np
import torch
from main_utils import parse_option
from train_dist_mod import TrainTester
from utils import get_scheduler
from mcln_transfer_initialization import checked_payload, load_mcln_core, install_portable_states

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
args.eval = False
args.debug = False
args.eval_train = False
args.batch_size = 8
args.num_workers = 0
args.rng_seed = 2027
args.use_cs_mcln = False
args.use_source_choice_selector = False
args.source_choice_selector_train_only = False
args.eval_use_selector_choice_scores = False
args.use_source_moe = False
args.source_moe_train_only = False
args.source_moe_gate_train_only = False
args.pp_checkpoint = None
args.frozen = False
args.small_lr = False
args.lr = args.lr_backbone = 1e-6
args.text_encoder_lr = 1e-6
args.weight_decay = 5e-4
args.clip_norm = .1
args.lr_scheduler = 'step'
args.lr_decay_epochs = [280, 340]
args.lr_decay_rate = .1
args.warmup_epoch = -1
args.max_epoch = 8
args.mask_loss_scale = 1.0 if args.mask_loss_scale is None else args.mask_loss_scale
args.consistency_loss_scale = 1.0 if args.consistency_loss_scale is None else args.consistency_loss_scale
assert args.model == 'MCLN' and args.dataset == ['scanrefer'] and args.test_dataset == 'scanrefer'
assert args.butd and not args.butd_cls and not args.butd_gt
model = TrainTester.get_model(args)
load_mcln_core(model, core)
del core
install_portable_states(model, spec)
model.portable_geometry_arm = 'full_transfer'
model = model.cuda().train()
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
criterion, set_criterion = TrainTester.get_criterion(args)
train, _ = TrainTester.get_datasets(args)
assert train is not None and len(train) >= 16
loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train, list(range(16))),
                                    batch_size=8, shuffle=False, num_workers=0, drop_last=True)
scheduler = get_scheduler(optimizer, len(loader), args)
before = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
          if name.startswith(('candidate_support_corrector.', 'candidate_span_mixer.'))}

def gradient_role(prefix):
    rows = [(name, parameter.grad) for name, parameter in model.named_parameters()
            if name.startswith(prefix) and parameter.grad is not None]
    return dict(total_l1=sum(float(gradient.detach().abs().sum()) for _, gradient in rows),
                output_l1=sum(float(gradient.detach().abs().sum()) for name, gradient in rows if '.output.' in name),
                internal_l1=sum(float(gradient.detach().abs().sum()) for name, gradient in rows if '.output.' not in name),
                gradient_tensors=len(rows))

updates = []
for step, raw_batch in enumerate(loader, 1):
    batch = TrainTester._to_gpu(raw_batch)
    inputs = TrainTester._get_inputs(batch)
    inputs['train'] = True
    predictions = model(inputs)
    for name, value in batch.items():
        assert name not in predictions
        predictions[name] = value
    loss, predictions = TrainTester._compute_loss(predictions, criterion, set_criterion, args)
    assert loss.ndim == 0 and torch.isfinite(loss)
    optimizer.zero_grad()
    loss.backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
    support = gradient_role('candidate_support_corrector.')
    span = gradient_role('candidate_span_mixer.')
    core_gradient = sum(float(parameter.grad.detach().abs().sum()) for name, parameter in model.named_parameters()
                        if parameter.grad is not None and not name.startswith(('candidate_support_corrector.', 'candidate_span_mixer.')))
    raw_gate = predictions['span_raw_axis_gate'].detach()
    record = dict(step=step, rows=len(batch['point_clouds']), loss=float(loss.detach()),
                  support_gradient=support, span_gradient=span, original_core_gradient_l1=core_gradient,
                  raw_span_gate_min=float(raw_gate.min()), raw_span_gate_max=float(raw_gate.max()),
                  span_gate_clamped_low_fraction=float((raw_gate < 0).float().mean()),
                  span_gate_clamped_high_fraction=float((raw_gate > 1).float().mean()),
                  invalid_mask_references=int((~predictions['mask_reference_valid']).sum()))
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
    optimizer.step()
    scheduler.step()
    updates.append(record)
    print(json.dumps(record), flush=True)
    del inputs, predictions, loss, batch, raw_batch
assert len(updates) == 2
changed = {prefix: sum(int((parameter.detach().cpu() != before[name]).sum())
                      for name, parameter in model.named_parameters() if name.startswith(prefix))
           for prefix in ('candidate_support_corrector.', 'candidate_span_mixer.')}
assert all(value > 0 for value in changed.values())
assert all(any(row[role]['output_l1'] > 0 and row[role]['internal_l1'] > 0 for row in updates)
           for role in ('support_gradient', 'span_gradient'))
assert all(row['original_core_gradient_l1'] > 0 for row in updates)

def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state_all())

def exact(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(exact(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
    return left == right

checkpoint = output / 'engineering_full_state.pth'
torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                rng=capture_rng(), optimizer_updates=2, spec=spec, engineering_only=True), str(checkpoint))
saved = torch.load(str(checkpoint), map_location='cpu')
with torch.no_grad():
    model.candidate_span_mixer.output.bias.add_(1)
torch.rand(1, device='cuda')
model.load_state_dict(saved['model'], strict=True)
optimizer.load_state_dict(saved['optimizer'])
scheduler.load_state_dict(copy.deepcopy(saved['scheduler']))
random.setstate(saved['rng']['python'])
np.random.set_state(saved['rng']['numpy'])
torch.set_rng_state(saved['rng']['torch_cpu'])
torch.cuda.set_rng_state_all(saved['rng']['torch_cuda'])
assert exact(model.state_dict(), saved['model'])
assert exact(optimizer.state_dict(), saved['optimizer'])
assert exact(scheduler.state_dict(), saved['scheduler'])
assert exact(capture_rng(), saved['rng'])
result = dict(status='ACTUAL_NATIVE_MCLN_TWO_UPDATES_AND_SAME_PROCESS_FULL_RESTORE_PASS',
              actual_optimizer_updates=2, actual_training_rows=16, actual_training_dataset_rows=len(train),
              actual_training_flags={name: getattr(args, name) for name in ('joint_det', 'detect_intermediate', 'augment_det', 'butd')},
              updates=updates, changed_module_elements=changed, original_core_trainable=True,
              roberta_parameters_frozen=True, full_model_states=len(model.state_dict()),
              optimizer_state_count=len(optimizer.state), same_process_saved_model_optimizer_scheduler_rng_exact=True,
              cold_restart_verified=False, formal_training_completed=False, formal_accuracy=None,
              engineering_checkpoint=str(checkpoint), engineering_checkpoint_bytes=checkpoint.stat().st_size,
              peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
              completed_cst=datetime.datetime.now().astimezone().isoformat(), full_goal_complete=False)
(output / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
