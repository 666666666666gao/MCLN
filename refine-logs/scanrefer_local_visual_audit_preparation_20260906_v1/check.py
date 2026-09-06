import datetime, gc, hashlib, json, os
from pathlib import Path
import pytest
import torch
from scripts.audit_scanrefer_local_visual_pair import reconstruct_initial_states, check_readouts

directory = Path('/root/autodl-tmp/mcln_scanrefer_local_visual_audit_preparation_20260906_v1')
manifest = json.loads(Path('/root/autodl-tmp/mcln_scanrefer_local_visual_pair_20260906_v1/input_manifest.json').read_text())
assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and not torch.cuda.is_available()
initial, first, core = reconstruct_initial_states(manifest)
base_shapes = {key: tuple(value.shape) for key, value in initial.items()}
del initial
gc.collect()
initial, second, second_core = reconstruct_initial_states(manifest)
assert core == second_core
assert base_shapes == {key: tuple(value.shape) for key, value in initial.items()}
assert all(torch.equal(first[key], second[key]) for key in first)
del initial
gc.collect()

import scripts
scripts.__path__.insert(0, str(directory / 'scripts'))
from scripts.scanrefer_joint_readout import JointRecReadout
original = {name: torch.load(item['path'], map_location='cpu')
            for name, item in manifest['artifacts'].items() if name != 'backbone'}
readout = JointRecReadout(original).eval()
saved = readout.export_artifacts()
check_readouts(saved, original)
for name in original:
    changed = readout.export_artifacts()
    parameter = next(iter(changed[name]['model_state_dict']))
    changed[name]['model_state_dict'][parameter].reshape(-1)[0] += .1
    with pytest.raises(AssertionError):
        check_readouts(changed, original)
fingerprints = {name: hashlib.sha256(value.numpy().tobytes()).hexdigest() for name, value in first.items()}
receipt = {'status': 'preparation_pass',
    'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'cpu_torch': torch.__version__, 'gpu_forwards': 0, 'optimizer_updates': 0,
    'formal_rows': 0, 'trained_endpoint_checked': False,
    'reconstructed_initial_state_keys': len(base_shapes), 'core_trainable_tensors': len(core),
    'local_parameter_tensors': len(first), 'local_parameters': sum(value.numel() for value in first.values()),
    'repeated_cpu_factory_initialization_equal': True,
    'local_initial_fingerprints': fingerprints,
    'fingerprint_role': 'reconstructed seed0 CPU reference; not an independently saved trainer-start tensor dump',
    'real_readout_exports_equal': True, 'changed_readout_parameter_detected': list(original)}
with (directory / 'receipt.json').open('x') as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True)
print(json.dumps(receipt), flush=True)
