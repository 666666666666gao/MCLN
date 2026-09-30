import hashlib
import json
import os
from pathlib import Path
import sys
import torch


SOURCE = Path('/root/autodl-tmp/cs_mcln_readback_source_20260930_v1')
BASE = Path('/root/autodl-tmp/cs_mcln_source_20260923_v1')
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(SOURCE))
os.chdir(BASE)
import train_dist_mod
import models.cs_mcln_modules
import models.mcln
from scripts.train_cs_mcln_scanrefer import experiment_args, load_exact_e71, parameter_groups, set_seed, SEED


set_seed(SEED)
payload = torch.load('/root/autodl-tmp/DATA_ROOT/output/preserved_best/mcln_pair_sweep/mcln_pair_default_rankblend010_2ep_best_acc025_epoch71_0.57993.pth', map_location='cpu')
config = experiment_args(payload['config'], 'cs', '/root/autodl-tmp/DATA_ROOT')
assert config.use_cs_mcln and not config.use_cs_geometry_readback
assert config.cs_native_root_quality_weight == 0
model = train_dist_mod.TrainTester.get_model(config)
assert model.cs_geometry_readback is None
assert model.cs_structure is not None
assert model.cs_context_reader is not None
assert model.cs_box_refiner is not None
assert all(parameter.device.type == 'cpu' for parameter in model.parameters())
loaded = load_exact_e71(model, payload['model'], 'cs')
assert loaded == {'core_tensors': 1135, 'removed_selector_tensors': 9, 'new_tensors': 40}
groups, rates = parameter_groups(model, 12)
assert [group['name'] for group in groups] == ['new', 'core', 'backbone']
paths = {
    'train_dist_mod.py': Path(train_dist_mod.__file__),
    'models/cs_mcln_modules.py': Path(models.cs_mcln_modules.__file__),
    'models/mcln.py': Path(models.mcln.__file__),
}
for name, path in paths.items():
    assert path.resolve() == (SOURCE / name).resolve()
receipt = {
    'arm': 'cs', 'batch_size': 12, 'seed': SEED,
    'initial_load': loaded,
    'three_cs_modules_instantiated': True,
    'R_instantiated': False,
    'all_parameters_on_cpu': True,
    'optimizer_parameter_groups': [group['name'] for group in groups],
    'rates': rates,
    'source_files': {name: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                     for name, path in paths.items()},
    'gpu_forward_performed': False, 'optimizer_steps_performed': 0,
    'dataset_loading_performed': False,
    'limit': 'CPU construction and exact E71 load only; not a GPU input/gradient/capacity or accuracy result',
}
print(json.dumps(receipt), flush=True)
