import hashlib
import json
import runpy
import torch


factory = runpy.run_path('/root/check_cs_numeric_control_factory_20260930.py')
control_model = factory['model']
control_state = control_model.state_dict()
set_seed = factory['set_seed']
SEED = factory['SEED']
set_seed(SEED)
config = factory['experiment_args'](
    factory['payload']['config'], 'cs_readback', '/root/autodl-tmp/DATA_ROOT',
)
readback_model = factory['train_dist_mod'].TrainTester.get_model(config)
loaded = factory['load_exact_e71'](readback_model, factory['payload']['model'], 'cs_readback')
assert loaded == {'core_tensors': 1135, 'removed_selector_tensors': 9, 'new_tensors': 54}
assert readback_model.cs_geometry_readback is not None
assert all(parameter.device.type == 'cpu' for parameter in readback_model.parameters())
readback_state = readback_model.state_dict()
assert len(control_state) == 1175 and len(readback_state) == 1189
extra = set(readback_state) - set(control_state)
assert len(extra) == 14 and all(name.startswith('cs_geometry_readback.') for name in extra)
assert all(torch.equal(value, readback_state[name]) for name, value in control_state.items())
prefixes = ('cs_structure.', 'cs_context_reader.', 'cs_box_refiner.')
shared_new = {name: value for name, value in control_state.items() if name.startswith(prefixes)}
assert len(shared_new) == 40
assert not bool(readback_state['cs_geometry_readback.output.weight'].count_nonzero())
assert not bool(readback_state['cs_geometry_readback.output.bias'].count_nonzero())
receipt = {
    'seed': SEED,
    'CS_initial_load': factory['loaded'], 'R_initial_load': loaded,
    'shared_state_count': len(control_state),
    'shared_core_state_count': 1135, 'shared_new_CS_state_count': len(shared_new),
    'all_shared_states_bitwise_equal': True,
    'new_readback_state_count': len(extra), 'readback_output_zero_initialized': True,
    'shared_CS_initial_state_sha256': {
        name: hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()
        for name, value in shared_new.items()
    },
    'gpu_forward_performed': False, 'optimizer_steps_performed': 0,
    'dataset_loading_performed': False,
    'limit': 'CPU initialization equality only; real input, training gradients, BN/dropout stream and performance remain separate checks',
}
print(json.dumps(receipt), flush=True)
