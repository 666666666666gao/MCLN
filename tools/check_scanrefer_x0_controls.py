"""Run isolated CPU checks on X0 controls; never construct MCLN or load data."""

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch


root = Path(__file__).resolve().parents[1]
paths = [
    root / 'main_utils.py', root / 'train_dist_mod.py',
    root / 'models/candidate_local_visual_training.py',
]
trees = {}
hashes = {}
for path in paths:
    data = path.read_bytes()
    trees[path.name] = ast.parse(data, filename=str(path))
    hashes[str(path.relative_to(root))] = hashlib.sha256(data).hexdigest()

# Execute only the actual training-mode methods, without application imports.
base = next(node for node in trees['main_utils.py'].body
            if isinstance(node, ast.ClassDef) and node.name == 'BaseTrainTester')
methods = [node for node in base.body if isinstance(node, ast.FunctionDef)
           and node.name in ('_set_source_moe_module_train_mode',
                             '_set_source_moe_train_mode')]
assert len(methods) == 2
namespace = {'torch': torch}
isolated = ast.Module(body=[ast.ClassDef(
    name='BaseTrainTester', bases=[], keywords=[], body=methods,
    decorator_list=[])], type_ignores=[])
exec(compile(ast.fix_missing_locations(isolated), '<actual-mode-methods>', 'exec'),
     namespace)
mode = namespace['BaseTrainTester']._set_source_moe_train_mode
model = torch.nn.Sequential(torch.nn.BatchNorm1d(4), torch.nn.Linear(4, 2))
mode(model, SimpleNamespace())
assert model[0].training and model[1].training
before = {name: value.clone() for name, value in model.named_buffers()}
mode(model, SimpleNamespace(freeze_batchnorm=True))
assert not model[0].training and model[1].training
model(torch.arange(16, dtype=torch.float32).reshape(4, 4)).sum().backward()
assert model[0].weight.grad is not None and model[1].weight.grad is not None
assert all(torch.equal(before[name], value)
           for name, value in model.named_buffers())
model.pvground_support_corrector = torch.nn.Sequential(torch.nn.BatchNorm1d(4))
model.pvground_span_mixer = torch.nn.Linear(4, 4)
mode(model, SimpleNamespace(pvground_train_only=True,
                           disable_pvground_reader=True, freeze_batchnorm=True))
assert model.pvground_span_mixer.training
assert not model.pvground_support_corrector[0].training

# Exercise the actual Reader/Support/Span attachment block with small stand-ins.
trainer = next(node for node in trees['train_dist_mod.py'].body
               if isinstance(node, ast.ClassDef) and node.name == 'TrainTester')
builder = next(node for node in trainer.body
               if isinstance(node, ast.FunctionDef) and node.name == 'get_model')
start = next(index for index, node in enumerate(builder.body)
             if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == 'local_variant'
                     for target in node.targets))
block = ast.Module(body=builder.body[start:-1], type_ignores=[])
code = compile(ast.fix_missing_locations(block), '<actual-module-attachment>', 'exec')
for disabled in (False, True):
    built = SimpleNamespace(decoder=[SimpleNamespace() for _ in range(6)])
    args = SimpleNamespace(use_pvground_innovations=True,
                           disable_pvground_reader=disabled, num_decoder_layers=6)
    exec(code, dict(args=args, model=built, CandidateLocalVisual=object,
                    CandidateMaskSupportCorrector=object, ExtremalSpanMixer=object))
    assert hasattr(built.decoder[5], 'local_visual') is (not disabled)
    assert hasattr(built, 'pvground_support_corrector')
    assert hasattr(built, 'pvground_span_mixer')

state_namespace = {}
exec(compile(trees['candidate_local_visual_training.py'], '<actual-validator>', 'exec'),
     state_namespace)
validate = state_namespace['validate_local_visual_checkpoint']
tensor = torch.zeros(1)
official = {'native': tensor}
new = dict(official, **{'pvground_support_corrector.weight': tensor,
                       'pvground_span_mixer.weight': tensor})
validate(new, official, True, require_local_visual=False,
         additional_new_prefixes=('pvground_support_corrector.',
                                  'pvground_span_mixer.'))
validate(new, new, False, require_local_visual=False)
with_reader = dict(new, **{'decoder.5.local_visual.weight': tensor})
rejected = False
try:
    validate(new, with_reader, True, require_local_visual=False)
except ValueError:
    rejected = True
assert rejected
validate(with_reader, with_reader, False)
print(json.dumps({'passed': True, 'cuda_available_to_process': False,
                  'scope': 'Isolated source compilation and extracted controls on CPU; '
                           'no MCLN construction, data, checkpoint, training run or GPU use.',
                  'checks': ['legacy Reader attachment', 'Reader-free Support/Span',
                             'joint BN frozen buffers and affine gradients',
                             'Stage A without Reader and with frozen BN',
                             'official initialization and exact resume validation',
                             'reject unrequested Reader checkpoint keys'],
                  'sha256': hashes}))
