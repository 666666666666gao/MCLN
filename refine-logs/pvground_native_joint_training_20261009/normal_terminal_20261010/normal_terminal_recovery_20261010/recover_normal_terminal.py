"""Cold-rebuild both completed native checkpoints; no forward or update."""
import argparse
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--protocol', required=True)
parser.add_argument('--terminal', required=True)
parser.add_argument('--output', required=True)
options = parser.parse_args()
protocol = json.loads(Path(options.protocol).read_bytes())
terminal = json.loads(Path(options.terminal).read_bytes())
assert terminal['status'] == 'complete' and terminal['exit_code'] == 0
assert [r['epoch'] for r in terminal['metrics']] == [0, 1, 2, 3]
output = Path(options.output)
assert not output.exists()
output.mkdir(parents=True)
source = Path(protocol['model_source'])
os.chdir(str(source))
sys.path.insert(0, str(source))
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from main_utils import parse_option, load_checkpoint, native_metric_order
from train_dist_mod import TrainTester
from pcdet.config import cfg_from_yaml_file, cfg as model_cfg
from utils import get_scheduler

torch.cuda.set_device(0)
dist.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=5400))
cfg_from_yaml_file('wandb_config.yaml', model_cfg)
retained = max(terminal['metrics'], key=native_metric_order)
assert retained == terminal['metrics'][0]
receipts = []
for identity in terminal['weights']:
    checkpoint = Path(identity['path'])
    name = checkpoint.stem
    assert name in ('best', 'latest') and checkpoint.is_file()
    digest = hashlib.sha256()
    with checkpoint.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    assert checkpoint.stat().st_size == identity['bytes'] and digest.hexdigest() == identity['sha256']
    saved = torch.load(str(checkpoint), map_location='cpu')
    assert len(saved['model']) == 1295
    assert saved['epoch'] == (0 if name == 'best' else 3)
    assert saved['retained_metrics'] == retained
    sys.argv = [str(source / 'train_dist_mod.py')] + protocol['common_arguments'] + [
        '--native_init_spec', str(source / 'init_manifests/extremal_support.json'),
        '--log_dir', str(output / name), '--exp', 'terminal_recovery_only']
    args = parse_option()
    assert not args.checkpoint_path and not args.eval
    tester = TrainTester(args, model_cfg)
    # Set the actual checkpoint after creating separate recovery logs.
    args.checkpoint_path = str(checkpoint)
    model = tester.get_model(args)
    optimizer = tester.get_optimizer(args, model)
    scheduler = get_scheduler(optimizer, 4583, args)
    model = DistributedDataParallel(model.cuda(), device_ids=[0], broadcast_buffers=False, find_unused_parameters=True)
    load_checkpoint(args, model, optimizer, scheduler)
    restored = model.state_dict()
    assert set(restored) == set(saved['model'])
    assert all(torch.equal(value.cpu(), saved['model'][key]) for key, value in restored.items())
    adam = optimizer.state_dict()
    assert adam['param_groups'] == saved['optimizer']['param_groups']
    assert set(adam['state']) == set(saved['optimizer']['state'])
    for index, state in adam['state'].items():
        expected = saved['optimizer']['state'][index]
        assert set(state) == set(expected)
        for key, value in state.items():
            assert torch.equal(value.cpu(), expected[key]) if torch.is_tensor(value) else value == expected[key]
    assert scheduler.state_dict() == saved['scheduler']
    assert args.start_epoch == saved['epoch'] + 1
    assert random.getstate() == saved['rng']['python']
    nrng = np.random.get_state()
    assert nrng[0] == saved['rng']['numpy'][0] and nrng[2:] == saved['rng']['numpy'][2:]
    assert np.array_equal(nrng[1], saved['rng']['numpy'][1])
    assert torch.equal(torch.get_rng_state(), saved['rng']['torch'])
    crng = torch.cuda.get_rng_state_all()
    assert len(crng) == len(saved['rng']['cuda'])
    assert all(torch.equal(value, expected) for value, expected in zip(crng, saved['rng']['cuda']))
    receipts.append(dict(name=name, identity=identity, saved_epoch=saved['epoch'], full_state_tensors=len(restored),
        optimizer_states=len(adam['state']), optimizer_step_values=sorted({int(s['step']) for s in adam['state'].values()}),
        restored_start_epoch=args.start_epoch, full_recovery_exact=True, retained_metrics=saved['retained_metrics']))
    del saved, restored, adam, model, optimizer, scheduler, tester
    gc.collect()
    torch.cuda.empty_cache()

result = dict(status='NORMAL_BEST_AND_FIXED_E3_FULL_COLD_RECOVERY_COMPLETE', time_cst=datetime.datetime.now().astimezone().isoformat(),
    checkpoints=receipts, actual_native_factory=True, actual_native_checkpoint_loader=True, scheduler_epoch_updates=4583,
    new_neural_forwards=0, new_optimizer_steps=0, new_training_batches=0, formal_accuracy_not_rerun=True,
    source=str(source), actual_imports={n: str(sys.modules[n].__file__) for n in ('train_dist_mod','main_utils','native_model_initialization','models.pv_ground')},
    full_goal_complete=False)
(output / 'RECOVERY_RECEIPT.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
dist.destroy_process_group()
