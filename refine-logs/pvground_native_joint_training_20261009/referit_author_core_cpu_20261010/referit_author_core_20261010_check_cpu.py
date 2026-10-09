"""Actual native factory author-core loading only; no forward, optimizer or data."""
import copy
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

root = Path(__file__).resolve().parent
source = root / 'PV-Ground'
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
os.chdir(str(source))
sys.path.insert(0, str(source))
import torch
torch.set_num_threads(1)
assert not torch.cuda.is_initialized()
from main_utils import parse_option
from train_dist_mod import TrainTester
from pcdet.config import cfg, cfg_from_yaml_file
assert Path(inspect.getfile(TrainTester)).resolve() == source/'train_dist_mod.py'
from native_model_initialization import configure_native_model
assert Path(inspect.getfile(configure_native_model)).resolve() == source/'native_model_initialization.py'
cfg_from_yaml_file('wandb_config.yaml', cfg)
protocol = json.loads((root/'NORMAL_NATIVE_RUN_PROTOCOL.json').read_bytes())
cases = {}
for dataset in ('nr3d', 'sr3d'):
    started = time.monotonic()
    arguments = list(protocol['common_arguments'])
    arguments[arguments.index('--dataset')+1] = dataset
    arguments[arguments.index('--test_dataset')+1] = dataset
    spec_path = root/dataset/'init.json'
    sys.argv = [str(source/'train_dist_mod.py')] + arguments + ['--native_init_spec', str(spec_path)]
    args = parse_option()
    assert args.dataset == [dataset] and args.test_dataset == dataset
    assert args.rng_seed == 2027 and args.checkpoint_path is None
    torch.manual_seed(2027)
    tester = TrainTester.__new__(TrainTester)
    tester.model_cfg = copy.deepcopy(cfg)
    model = tester.get_model(args)
    spec = json.loads(spec_path.read_bytes())
    payload = torch.load(spec['official_checkpoint'], map_location='cpu')
    expected = {name[7:]:value for name,value in payload['model'].items()}
    assert len(expected) == 1235
    positions = expected.pop('text_encoder.embeddings.position_ids')
    assert torch.equal(positions, model.text_encoder.embeddings.position_ids)
    actual = model.state_dict()
    assert len(actual) == 1295 and len(expected) == 1234
    assert set(expected) <= set(actual)
    assert all(actual[name].shape == value.shape and actual[name].dtype == value.dtype
        and actual[name].device.type == 'cpu' and torch.equal(actual[name], value)
        for name, value in expected.items())
    assert 'text_encoder.embeddings.position_ids' not in actual
    assert not any(parameter.requires_grad for parameter in model.text_encoder.parameters())
    assert all(torch.count_nonzero(value) == 0
        for module in (model.candidate_support_corrector, model.candidate_span_mixer)
        for value in module.output.state_dict().values())
    assert model.use_g_supervision and model.use_selected_mask_supervision
    assert model.native_training_architecture['scanrefer_core_or_module_state_loaded'] is False
    assert not torch.cuda.is_initialized()
    author_config = vars(payload['config'])
    keys = ('use_color','use_height','use_multiview','butd','butd_gt','butd_cls','joint_det','detect_intermediate','augment_det')
    cases[dataset] = dict(author_checkpoint_sha256=spec['official_checkpoint_sha256'],
        native_factory=str(inspect.getfile(TrainTester)), author_core_tensors=1235,
        exact_loaded_core_tensors=1234, full_model_state_tensors=1295,
        deterministic_position_buffer_verified=True, core_key_shape_dtype_and_values_exact=True,
        new_G_A_B_initialized=True, A_and_B_output_zero_verified=True,
        model_architecture=model.native_training_architecture,
        author_training_flags={name:author_config.get(name) for name in keys},
        CPU_factory_flags={name:getattr(args,name) for name in keys},
        CPU_only=True, CUDA_initialized=False, forwards=0, optimizer_steps=0,
        real_loader_rows=0, elapsed_seconds=time.monotonic()-started)
    del expected, actual, payload, model, tester
    gc.collect()
result = dict(status='ACTUAL_AUTHOR_CORE_NATIVE_FACTORY_CPU_LOAD_PASS', cases=cases,
    torch_version=torch.__version__, CUDA_initialized=torch.cuda.is_initialized(),
    current_training_queries=0, GPU_calls=0, training_launched=False,
    scope='Two native model constructors and exact author-core loading; no forward/criterion/loader/optimizer/recovery or REC accuracy')
(root/'AUTHOR_CORE_CPU_RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
