"""Inspect both closed terminals and their complete CPU reconstruction, no NN."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

root=Path(__file__).resolve().parents[1]
assert not (root/'postrun/checkpoint_inspection.json').exists()
summary=json.loads((root/'analysis/SUMMARY.json').read_bytes())
assert summary['status']=='ACTUAL_CLOSED_FULL_FORMAL_CPU_RECOUNT'
assert summary['formal_rows_per_arm']==9508 and summary['support_total_updates_at_terminal']==7446
audit=json.loads((root/'analysis/EXPERIMENT_AUDIT.json').read_bytes())
assert audit['execution_scope']=='ACTUAL_CLOSED_TRAINED_PAIR' and audit['fresh_context'] is True
assert audit['verdict'] in ('PASS','WARN') and not audit['blocking_findings']
required={(root/name).resolve() for name in ('analysis/SUMMARY.json','complete_fit/INTAKE.json','pair_spec.json')}
assert required.issubset({Path(row['path']).resolve() for row in audit['reviewed_files']})
for row in audit['reviewed_files']:
    assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
spec=json.loads((root/'pair_spec.json').read_bytes())
expected={arm:json.loads((root/'complete_fit'/arm/'formal_restore.json').read_bytes())['terminal_sha256']
          for arm in spec['support_modes']}
code=r'''import datetime,hashlib,json,os,sys
from pathlib import Path
b=json.load(sys.stdin);spec=b['spec'];root=Path(spec['root'])
assert root.resolve()==root
assert (root/'fit_controller.exit').read_text().strip()=='0'
assert json.loads((root/'fit_status.json').read_bytes())['status']=='complete'
assert os.environ['CUDA_VISIBLE_DEVICES']==''
runtime=Path(spec['runtime']);env=json.loads((runtime/'env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest()==spec['env_spec_sha256']
os.environ.update(env['env']);os.environ['CUDA_VISIBLE_DEVICES']=''
model_source=Path(spec['model_source']);os.chdir(str(model_source))
sys.path[:0]=[str(model_source),str(root),spec['helper_root']]
sys.path.extend(env['env']['PYTHONPATH'].split(':'))
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()
assert sha(spec['source_port'])==spec['source_port_sha256']
for name,digest in json.loads(Path(spec['source_port']).read_bytes())['files'].items():assert sha(model_source/name)==digest,name
for name,digest in spec['new_runner_files'].items():assert sha(root/name)==digest,name
for name,digest in spec['runner_files'].items():assert sha(Path(spec['helper_root'])/name)==digest,name
import torch
from pcdet.config import cfg,cfg_from_yaml_file
from mask_support_model_factory import build_support_model
cfg_from_yaml_file(str(runtime/'PV-Ground/wandb_config.yaml'),cfg)
official=env['weight_dirs']['scanrefer']
deps=[dict(path=official['path'],sha256=spec['checkpoint_sha256']),
      dict(path=spec['base_terminal'],sha256=spec['base_terminal_sha256']),
      dict(path=spec['selected_terminal'],sha256=spec['selected_terminal_sha256'])]
for row in deps:assert sha(row['path'])==row['sha256'];row['bytes']=Path(row['path']).stat().st_size
official_payload=torch.load(official['path'],map_location='cpu')
original_g=torch.load(spec['base_terminal'],map_location='cpu')
selected=torch.load(spec['selected_terminal'],map_location='cpu')
data_root=json.loads(Path(spec['input_manifest']).read_bytes())['data_root']
torch.manual_seed(spec['seed'])
parent_rng_state=torch.get_rng_state()
parent,_,receipt=build_support_model(cfg,official_payload,original_g,selected,data_root)
assert receipt['full_state_tensors']==1304
parent_state={name:value.clone() for name,value in parent.state_dict().items()}
del parent
weights=[]
for arm in spec['support_modes']:
    path=root/arm/'terminal.pth';assert sha(path)==b['expected'][arm]
    payload=torch.load(str(path),map_location='cpu')
    assert payload['arm']==arm and payload['step']==3723
    assert payload['support_prior_updates']==3723 and payload['total_support_updates']==7446
    assert payload['optimizer_reinitialized'] and payload['unused_geometry_columns_zeroed_at_initialization']
    assert payload['geometry_encoding']==('signed_log' if arm=='box_conditioned' else 'zero')
    assert payload['mask_loss_coefficients']==[5,1,10,2] and payload['spec_sha256']==sha(root/'pair_spec.json')
    for key in ('checkpoint_sha256','base_terminal_sha256','selected_terminal_sha256','source_port_sha256','warm_support_terminal_sha256'):
        assert payload[key]==spec[key]
    assert len(payload['state_delta'])==10 and len(payload['row_ids'])==len(set(payload['row_ids']))==29778
    assert all(torch.isfinite(value).all().item() for value in payload['state_delta'].values())
    optimizer=payload['optimizer'];assert len(optimizer['state'])==10 and len(optimizer['param_groups'])==1
    assert all(float(value['step'])==3723 and torch.isfinite(value['exp_avg']).all().item()
        and torch.isfinite(value['exp_avg_sq']).all().item() for value in optimizer['state'].values())
    torch.set_rng_state(parent_rng_state)
    model,_,load=build_support_model(cfg,official_payload,original_g,selected,data_root,payload)
    expected=dict(parent_state,**payload['state_delta']);state=model.state_dict()
    assert set(state)==set(expected) and len(state)==load['full_state_tensors']==1314
    assert all(torch.equal(value,state[name]) for name,value in expected.items())
    assert load['deployed_support_heads']==1
    assert torch.count_nonzero(model.candidate_box_refiner.output.weight).item()==0
    assert torch.count_nonzero(model.candidate_box_refiner.output.bias).item()==0
    assert all(value.device.type=='cpu' for value in state.values()) and not torch.cuda.is_initialized()
    weights.append(dict(arm=arm,path=str(path),bytes=path.stat().st_size,sha256=b['expected'][arm],
        step=3723,total_support_updates=7446,geometry_encoding=payload['geometry_encoding'],
        state_tensors=10,optimizer_states=10,full_cpu_state_tensors=1314,full_state_exact=True))
    del model,state,expected,payload
print(json.dumps(dict(status='PASS_CLOSED_CPU_BOTH_TERMINALS_STRICT_RECONSTRUCTION',
    time_cst=datetime.datetime.now().astimezone().isoformat(),weights=weights,required_dependencies=deps,
    full_cpu_state_tensors=1314,cpu_parent_rng_seed=spec['seed'],cpu_parent_rng_replayed_per_build=True,
    uncheckpointed_zero_R_hidden_states_matched_between_cpu_builds_only=True,
    original_formal_gpu_all_state_identity_verified=False,cuda_initialized=False,neural_forwards=0,optimizer_updates=0,
    weights_copied=0,deletions=0,full_gpu_cold_reconstruction=False)))
'''
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
command=shlex.join(['env','CUDA_VISIBLE_DEVICES=',spec['runtime']+'/venv/bin/python','-B','-c',code])
stdin,stdout,stderr=client.exec_command(command,timeout=180)
stdin.write(json.dumps(dict(spec=spec,expected=expected)));stdin.channel.shutdown_write()
raw=stdout.read();error=stderr.read().decode();assert stdout.channel.recv_exit_status()==0,error
client.close()
(root/'postrun/checkpoint_inspection.stdout.txt').write_bytes(raw)
(root/'postrun/checkpoint_inspection.stderr.txt').write_text(error,encoding='utf-8')
record=json.loads(raw.decode().splitlines()[-1])
assert record['status']=='PASS_CLOSED_CPU_BOTH_TERMINALS_STRICT_RECONSTRUCTION'
(root/'postrun/checkpoint_inspection.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=record['status'],terminals=2,neural_forwards=0,weights_copied=0,deletions=0)),flush=True)
