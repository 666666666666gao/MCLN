import json
import os
from pathlib import Path
import sys
import torch


SOURCE = '/root/autodl-tmp/cs_mcln_readback_source_20260930_v1'
BASE = '/root/autodl-tmp/cs_mcln_source_20260923_v1'
sys.path.insert(0, BASE)
sys.path.insert(0, SOURCE)
from scripts.train_cs_mcln_scanrefer import experiment_args, parameter_groups, SEED
from train_dist_mod import TrainTester


root = Path('/root/cs_mcln_scanrefer_readback_20260930')
saved = torch.load(str(root / 'latest.pth'), map_location='cpu')
assert saved['arm'] == 'cs_readback' and saved['seed'] == SEED
assert saved['batch_size'] == 12 and saved['epoch'] >= 1
assert saved['root_quality_loss_weight'] == 0.0
assert saved['initial_load'] == {
    'core_tensors': 1135, 'removed_selector_tensors': 9, 'new_tensors': 54,
}
os.chdir(BASE)
initial = torch.load('/root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_pair_sweep/mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_0.57993.pth', map_location='cpu')
config = experiment_args(initial['config'], 'cs_readback', '/root/autodl-tmp/DATA_ROOT')
del initial
model = TrainTester.get_model(config)
assert model.cs_geometry_readback is not None
loaded = model.load_state_dict(saved['model'], strict=True)
assert not loaded.missing_keys and not loaded.unexpected_keys
restored = model.state_dict()
assert len(restored) == 1189
assert all(torch.equal(restored[name], value) for name, value in saved['model'].items())
groups, _ = parameter_groups(model, 12)
optimizer = torch.optim.AdamW(groups, weight_decay=0.0005)
optimizer.load_state_dict(saved['optimizer'])
state = optimizer.state_dict()
assert len(state['param_groups']) == len(saved['optimizer']['param_groups']) == 3
assert set(state['state']) == set(saved['optimizer']['state'])
for index, expected in saved['optimizer']['state'].items():
    assert set(state['state'][index]) == set(expected)
    for key, value in expected.items():
        if torch.is_tensor(value):
            assert torch.equal(state['state'][index][key], value)
        else:
            assert state['state'][index][key] == value
receipt = {
    'checkpoint_epoch': saved['epoch'], 'arm': saved['arm'],
    'model_tensor_states_strict_restored': len(restored),
    'readback_states': len([name for name in restored if name.startswith('cs_geometry_readback.')]),
    'optimizer_parameter_groups': len(state['param_groups']),
    'optimizer_parameter_states_exact_restored': len(state['state']),
    'all_model_tensors_exact': True, 'all_optimizer_states_exact': True,
    'cpu_only': True, 'optimizer_steps_performed': 0,
    'checkpoint_bytes': (root / 'latest.pth').stat().st_size,
}
assert receipt['readback_states'] == 14
print(json.dumps(receipt), flush=True)
