"""Recover the original failed-run saved full state; no new optimizer updates."""
import argparse, datetime, hashlib, json, os, random, sys
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--protocol',required=True)
parser.add_argument('--checkpoint-identity',required=True)
parser.add_argument('--output',required=True)
options=parser.parse_args()
protocol=json.loads(Path(options.protocol).read_bytes())
identity=json.loads(Path(options.checkpoint_identity).read_bytes())
output=Path(options.output);assert not output.exists();output.mkdir(parents=True)
source=Path(protocol['model_source']);os.chdir(str(source));sys.path.insert(0,str(source))
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from main_utils import parse_option,load_checkpoint
from train_dist_mod import TrainTester
from pcdet.config import cfg_from_yaml_file,cfg as model_cfg
from utils import get_scheduler
sys.argv=[str(source/'train_dist_mod.py')]+protocol['common_arguments']+[
    '--native_init_spec',str(source/'init_manifests/whole_support.json'),
    '--log_dir',str(output),'--exp','recovery_only_previous_whole']
args=parse_option()
torch.cuda.set_device(0)
dist.init_process_group(backend='nccl',init_method='env://',timeout=datetime.timedelta(seconds=5400))
cfg_from_yaml_file('wandb_config.yaml',model_cfg)
tester=TrainTester(args,model_cfg)
args.checkpoint_path=identity['path']
checkpoint=Path(args.checkpoint_path)
h=hashlib.sha256()
with checkpoint.open('rb') as f:
    for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
assert checkpoint.stat().st_size==identity['bytes'] and h.hexdigest()==identity['sha256']
saved=torch.load(str(checkpoint),map_location='cpu')
assert len(saved['model'])==1295 and saved['retained_metrics'] is None
assert all(int(state['step'])==2 for state in saved['optimizer']['state'].values())
restored_model=tester.get_model(args)
restored_optimizer=tester.get_optimizer(args,restored_model)
restored_scheduler=get_scheduler(restored_optimizer,4583,args)
restored_model=DistributedDataParallel(restored_model.cuda(),device_ids=[0],
    broadcast_buffers=False,find_unused_parameters=True)
load_checkpoint(args,restored_model,restored_optimizer,restored_scheduler)
restored_state = restored_model.state_dict()
assert set(restored_state) == set(saved['model'])
assert all(torch.equal(value.cpu(), saved['model'][name]) for name, value in restored_state.items())
restored_adam = restored_optimizer.state_dict()
assert restored_adam['param_groups'] == saved['optimizer']['param_groups']
assert set(restored_adam['state']) == set(saved['optimizer']['state'])
for index, state in restored_adam['state'].items():
    expected = saved['optimizer']['state'][index]
    assert set(state) == set(expected)
    for name, value in state.items():
        if torch.is_tensor(value):
            assert torch.equal(value.cpu(), expected[name])
        else:
            assert value == expected[name]
assert restored_scheduler.state_dict() == saved['scheduler'] and args.start_epoch == 1
assert random.getstate() == saved['rng']['python']
numpy_rng = np.random.get_state()
assert numpy_rng[0] == saved['rng']['numpy'][0] and numpy_rng[2:] == saved['rng']['numpy'][2:]
assert np.array_equal(numpy_rng[1], saved['rng']['numpy'][1])
assert torch.equal(torch.get_rng_state(), saved['rng']['torch'])
cuda_rng = torch.cuda.get_rng_state_all()
assert len(cuda_rng) == len(saved['rng']['cuda'])
assert all(torch.equal(value, expected) for value, expected in zip(cuda_rng, saved['rng']['cuda']))
receipt=dict(status='PREVIOUS_TWO_UPDATE_FULL_CHECKPOINT_COLD_RECOVERY_COMPLETE',
    checkpoint_identity=identity,full_recovery_exact=True,full_state_tensors=1295,
    new_optimizer_steps=0,new_training_batches=0,new_neural_forwards=0,
    original_training_attempt=2,original_controller_exit=1,original_fail_preserved=True,
    original_updates=2,restored_optimizer_states=len(restored_adam['state']),
    model_source=str(source),formal_accuracy=None,normal_training_started=False)
(output/'RECOVERY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
dist.destroy_process_group()
