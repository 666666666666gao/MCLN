"""Bind official Nr/Sr parents to D on CPU; no forward, update or saved weights."""
import datetime
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


training = Path('/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260917_task_observation_v1')
formal = Path('/root/autodl-tmp/mcln_pvground_scanrefer_formal_20260917_task_observation_v1')
spec = json.loads((training / 'spec.json').read_bytes())
runtime = Path(spec['runtime'])
env = json.loads((runtime / 'env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(env, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == spec['env_spec_sha256']
os.environ.update(env['env'])
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
source = Path(spec['model_source'])
assert sha(spec['source_port']) == spec['source_port_sha256']
for name, digest in json.loads(Path(spec['source_port']).read_bytes())['files'].items():
    assert sha(source / name) == digest, name
for name, field in [('pvground_source_query.py', 'source_query_module_sha256'),
                    ('pvground_observation_query.py', 'observation_module_sha256'),
                    ('pvground_task_observation_query.py', 'task_module_sha256')]:
    assert sha(formal / name) == spec[field], name
os.chdir(str(source))
sys.path[:0] = [str(formal), str(source)]
import torch
from models.pv_ground import PVGround
from pcdet.config import cfg, cfg_from_yaml_file
from pvground_task_observation_query import install_task_observation_query_read

assert Path(sys.modules['models.pv_ground'].__file__).resolve() == source / 'models/pv_ground.py'
torch.set_num_threads(1)
cfg_from_yaml_file(str(runtime / 'PV-Ground/wandb_config.yaml'), cfg)
original_config = copy.deepcopy(cfg)
results = {}
for dataset, filename, digest in [
    ('nr', 'PV-Ground_NR3D.pth', 'd2d9afaf9c293c54977f3555a46c7bb2a72d9f80163a3f602dd8426032d7fa5d'),
    ('sr', 'PV-Ground_SR3D.pth', 'a4a14b0090947177a648703ad6de094246891d89174fffa56fe454f730dbe3dc')]:
    started = time.time()
    parent = Path('/root/autodl-tmp/mcln_pvground_' + dataset + '_checkpoint_inspection_20260908_v1') / filename
    assert sha(parent) == digest
    payload = torch.load(str(parent), map_location='cpu')
    config = payload['config']
    assert not config.butd and config.butd_cls and not config.butd_gt
    assert config.num_target == 256 and config.num_decoder_layers == 6
    assert config.use_soft_token_loss and config.use_contrastive_align
    assert all(name.startswith('module.') for name in payload['model'])
    state = {name[7:]: value for name, value in payload['model'].items()}
    torch.manual_seed(2027)
    model_config = copy.deepcopy(original_config)
    model = PVGround(model_config, num_class=256, num_queries=256, num_decoder_layers=6,
        self_position_embedding=config.self_position_embedding, contrastive_align_loss=True,
        butd=config.butd or config.butd_cls or config.butd_gt, pointnet_ckpt=None,
        data_path='/root/autodl-tmp/DATA_ROOT_mcln_meshsp/', self_attend=config.self_attend)
    assert len(state) == 1235 and set(model.state_dict()) == set(state), dict(
        dataset=dataset, parent_count=len(state), model_count=len(model.state_dict()),
        missing=sorted(set(state) - set(model.state_dict())),
        extra=sorted(set(model.state_dict()) - set(state)))
    position_name = 'text_encoder.embeddings.position_ids'
    assert torch.equal(state[position_name], torch.arange(model.text_encoder.config.max_position_embeddings).expand((1, -1)))
    model.load_state_dict(state, strict=True)
    install_task_observation_query_read(model)
    expanded = model.state_dict()
    added = set(expanded) - set(state)
    assert len(expanded) == 1272 and len(added) == 37
    assert all(name.startswith('decoder.5.source_query_read.') for name in added)
    assert all(torch.equal(expanded[name], value) for name, value in state.items())
    assert sum(p.numel() for p in model.decoder[-1].source_query_read.parameters()) == 923616
    assert torch.count_nonzero(model.decoder[-1].source_query_read.task_queries) == 0
    assert sum(p.requires_grad for p in model.parameters()) == 820
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 28883227
    assert sum(not p.requires_grad for p in model.parameters()) == 199
    model.load_state_dict(expanded, strict=True)
    assert all(torch.equal(value, expanded[name]) for name, value in model.state_dict().items())
    assert not torch.cuda.is_initialized()
    assert cfg == original_config
    results[dataset] = dict(checkpoint_sha256=digest, parent_states=1235, expanded_states=1272,
        added_states=len(added), parent_values_unchanged=True, position_ids_persistent=True,
        strict_load=True, task_queries_zero=True, trainable_tensors=820,
        independent_config=True, model_constructor_mutated_its_config=model_config != original_config,
        trainable_parameters=28883227, elapsed_seconds=time.time() - started)
    del model, expanded, state, payload
    gc.collect()
record = dict(status='pass', time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    datasets=results, source_port_sha256=spec['source_port_sha256'], task_module_sha256=spec['task_module_sha256'],
    env_spec_sha256=spec['env_spec_sha256'], cpu_only=True, torch_cuda_initialized=False,
    model_forwards=0, optimizer_steps=0, formal_rows=0, new_checkpoints=0,
    scope='actual Nr/Sr parent CPU strict load plus D installation; no data forward, capacity or quality claim')
print('PVG_REFERIT_TASK_CPU_PASS ' + json.dumps(record), flush=True)
