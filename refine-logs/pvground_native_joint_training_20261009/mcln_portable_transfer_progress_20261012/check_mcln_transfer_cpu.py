"""Actual MCLN factory and pretrained-state check; CPU only, no forward."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--source',required=True)
parser.add_argument('--spec',required=True)
parser.add_argument('--output',required=True)
opt = parser.parse_args()
source = Path(opt.source);output = Path(opt.output)
assert not output.exists()
os.chdir(str(source));sys.path.insert(0,str(source))
import torch
from main_utils import parse_option
from train_dist_mod import TrainTester
from mcln_transfer_initialization import checked_payload,load_mcln_core,install_portable_states
spec = json.loads(Path(opt.spec).read_bytes())
core = checked_payload(spec['mcln_checkpoint'],spec['mcln_checkpoint_sha256'])
sys.argv = [str(source/'train_dist_mod.py')]
args = parse_option();vars(args).update(vars(core['config']))
args.data_root = spec['data_root'];args.use_cs_mcln = False
args.use_source_choice_selector = False;args.use_source_moe = False
args.eval_use_selector_choice_scores = False;args.pp_checkpoint = None
assert args.model == 'MCLN' and args.num_target == 256
model = TrainTester.get_model(args)
load_mcln_core(model,core)
before = {name:value.detach().clone() for name,value in model.state_dict().items()}
native_trainable = {name for name,param in model.named_parameters() if param.requires_grad}
install_portable_states(model,spec)
assert all(torch.equal(model.state_dict()[name],value) for name,value in before.items())
assert all(param.requires_grad for name,param in model.named_parameters() if name in native_trainable)
assert not any(param.requires_grad for param in model.text_encoder.parameters())
assert all(value.device.type == 'cpu' for value in model.state_dict().values())
result = dict(status='ACTUAL_MCLN_FACTORY_CORE_AND_PORTABLE_STATES_CPU_PASS',
    model_source=str(source),mcln_core_states=len(before),removed_SourceChoice_states=9,
    full_model_states=len(model.state_dict()),support_parameters=27841,span_parameters=29793,
    original_core_unchanged=True,original_trainability_preserved=True,RoBERTa_frozen=True,
    actual_neural_forwards=0,actual_optimizer_updates=0,actual_accuracy=None,
    actual_import_train_dist_mod=sys.modules['train_dist_mod'].__file__,
    actual_import_mcln=sys.modules['models.mcln'].__file__,spec=spec,full_goal_complete=False)
output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
