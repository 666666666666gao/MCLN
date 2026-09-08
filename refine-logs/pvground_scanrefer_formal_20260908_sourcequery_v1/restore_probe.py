"""CPU restore check for the exact formal model; no scene inference or optimizer."""
import argparse
from collections import Counter
import io
import json
import os
from pathlib import Path
import sys
import time

from evaluate import expanded_parent_state, terminal_state, sha, write_json, now

parser = argparse.ArgumentParser()
parser.add_argument('--spec', type=Path, required=True)
parser.add_argument('--terminal', action='store_true')
args = parser.parse_args()
root = args.spec.parent
spec = json.loads(args.spec.read_bytes())
for name,digest in spec['files'].items():
    assert sha(root/name) == digest, name
training = Path(spec['training_root'])
assert sha(training/'spec.json') == spec['training_spec_sha256']
train_spec = json.loads((training/'spec.json').read_bytes())
runtime = Path(train_spec['runtime'])
environment = json.loads((runtime/'env_spec.json').read_bytes())
source = Path(train_spec['model_source'])
assert sha(train_spec['source_port']) == train_spec['source_port_sha256']
port = json.loads(Path(train_spec['source_port']).read_bytes())
for name,digest in port['files'].items():
    assert sha(source/name) == digest, name
assert train_spec['source_query_read'] is True
assert sha(root/'pvground_source_query.py') == train_spec['source_query_module_sha256']
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
import torch
torch.set_num_threads(1)
torch.manual_seed(2027)
assert not torch.cuda.is_initialized()
os.chdir(str(source));sys.path.insert(0,str(source))
from models.pv_ground import PVGround
from pcdet.config import cfg, cfg_from_yaml_file
assert Path(sys.modules['models.pv_ground'].__file__).resolve() == source/'models/pv_ground.py'
checkpoint = environment['weight_dirs']['scanrefer']
assert sha(checkpoint['path']) == checkpoint['sha256'] == train_spec['checkpoint_sha256']
begin = time.time()
payload = torch.load(checkpoint['path'], map_location='cpu')
config = payload['config']
assert all(name.startswith('module.') for name in payload['model'])
parent = {name[7:]:value for name,value in payload['model'].items()}
cfg_from_yaml_file(str(runtime/'PV-Ground/wandb_config.yaml'),cfg)
model = PVGround(cfg,num_class=256,num_queries=256,num_decoder_layers=6,
    self_position_embedding=config.self_position_embedding,contrastive_align_loss=True,butd=True,
    pointnet_ckpt=None,data_path='/root/autodl-tmp/DATA_ROOT_mcln_meshsp/',self_attend=config.self_attend)
initial = expanded_parent_state(model,parent)
added = set(initial)-set(parent)
if args.terminal:
    receipt = json.loads((training/'receipt.json').read_bytes())
    assert receipt['status'] == 'complete'
    assert sha(training/'terminal.pth') == receipt['terminal_sha256']
    payload = torch.load(str(training/'terminal.pth'),map_location='cpu')
    partitions = json.loads(Path(json.loads(Path(train_spec['input_manifest']).read_bytes())['split_protocol']).read_bytes())['row_ids']
    assert payload['step'] == 3723 and Counter(payload['row_ids']) == Counter(partitions['fit'])
    assert payload['parent_checkpoint_sha256'] == checkpoint['sha256']
    assert payload['spec_sha256'] == sha(training/'spec.json')
    for key in ['source_query_read','source_query_module_sha256','source_port_sha256']:
        assert payload[key] == receipt[key] == train_spec[key], key
    delta = payload['state_delta']
    scope = 'actual fixed terminal delta'
else:
    selected = {name for name,p in model.named_parameters() if p.requires_grad}
    selected |= {name for name,_ in model.named_buffers() if name in initial}
    delta = {name:initial[name] for name in selected}
    for name in added:
        delta[name] = delta[name].clone()
        delta[name].view(-1)[0] += .125
    stream = io.BytesIO()
    torch.save(delta,stream);stream.seek(0)
    delta = torch.load(stream,map_location='cpu')
    scope = 'in-memory serialization fixture changing all 24 reader tensors; not trained weights'
terminal = terminal_state(model,initial,delta)
model.load_state_dict(terminal,strict=True)
assert all(torch.equal(value,terminal[name]) for name,value in model.state_dict().items())
assert all(torch.equal(model.state_dict()[name],delta[name]) for name in added)
model.decoder[-1].source_query_read.enabled = True
changed_added = [name for name in added if not torch.equal(terminal[name],initial[name])]
if not args.terminal:
    assert len(changed_added) == 24
    missing = dict(delta);del missing[sorted(added)[0]]
    rejected = False
    try:
        terminal_state(model,initial,missing)
    except AssertionError:
        rejected = True
    assert rejected, 'missing new state must be rejected rather than silently initialized'
model.load_state_dict(initial,strict=True)
model.decoder[-1].source_query_read.enabled = False
assert all(torch.equal(value,initial[name]) for name,value in model.state_dict().items())
assert all(torch.equal(model.state_dict()[name],parent[name]) for name in parent)
assert not torch.cuda.is_initialized()
result = dict(status='pass',time_cst=now(),scope=scope,parent_tensors=len(parent),
    added_tensors=len(added),delta_tensors=len(delta),expanded_tensors=len(initial),
    changed_added_tensors=len(changed_added),strict_terminal_and_parent_restore=True,
    missing_added_state_rejected=True if not args.terminal else None,cpu_only=True,torch_cuda_initialized=False,
    model_forwards=0,optimizer_steps=0,formal_rows=0,new_checkpoint_files=0,
    checkpoint_sha256=checkpoint['sha256'],source_port_sha256=train_spec['source_port_sha256'],
    module_sha256=train_spec['source_query_module_sha256'],evaluator_sha256=sha(root/'evaluate.py'),
    terminal_sha256=receipt['terminal_sha256'] if args.terminal else None,
    training_spec_sha256=sha(training/'spec.json'),
    elapsed_seconds=time.time()-begin,script_sha256=sha(__file__))
write_json(root/('terminal_restore.json' if args.terminal else 'restore_preparation.json'),result)
print('PVG_SOURCE_QUERY_CPU_RESTORE_PASS '+json.dumps(result),flush=True)
