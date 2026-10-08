"""Read closed checkpoint states, strictly rebuild on CPU, archive only metric best."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

root = Path(__file__).resolve().parent.parent
receipt_path = root / 'postrun/checkpoint_inspection.json'
assert not receipt_path.exists()
summary = json.loads((root / 'analysis/SUMMARY.json').read_bytes())
assert summary['metric_best_candidate']['arm'] == 'content'
assert summary['metric_best_candidate']['rec_hits25'] == 5598
assert summary['metric_best_candidate']['rec_hits50'] == 4856
spec = json.loads((root / 'pair_spec.json').read_bytes())
expected = {arm: json.loads((root / 'complete_fit' / arm / 'formal_restore.json').read_bytes())['terminal_sha256'] for arm in spec['support_modes']}
code = r'''import datetime,hashlib,json,os,sys
from pathlib import Path
b=json.load(sys.stdin);spec=b['spec'];root=Path(spec['root'])
assert root.resolve()==root
assert (root/'fit_controller.exit').read_text().strip()=='0'
assert json.loads((root/'fit_status.json').read_bytes())['status']=='complete'
assert os.environ['CUDA_VISIBLE_DEVICES']==''
runtime=Path(spec['runtime']);env=json.loads((runtime/'env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest()==spec['env_spec_sha256']
os.environ.update(env['env'])
os.environ['CUDA_VISIBLE_DEVICES']=''
model_source=Path(spec['model_source']);os.chdir(str(model_source))
sys.path[:0]=[str(model_source),str(root),spec['helper_root']]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
assert sha(spec['source_port'])==spec['source_port_sha256']
for name,digest in json.loads(Path(spec['source_port']).read_bytes())['files'].items():assert sha(model_source/name)==digest,name
for name,digest in spec['new_runner_files'].items():assert sha(root/name)==digest,name
for name,digest in spec['runner_files'].items():assert sha(Path(spec['helper_root'])/name)==digest,name
import torch
from pcdet.config import cfg,cfg_from_yaml_file
from mask_support_model_factory import build_support_model
cfg_from_yaml_file(str(runtime/'PV-Ground/wandb_config.yaml'),cfg)
payloads={};weights=[]
for arm in spec['support_modes']:
    path=root/arm/'terminal.pth';assert sha(path)==b['expected'][arm]
    payload=torch.load(str(path),map_location='cpu');payloads[arm]=payload
    assert payload['arm']==arm and payload['step']==3723
    assert payload['mask_loss_coefficients']==[5,1,10,2]
    assert payload['spec_sha256']==sha(root/'pair_spec.json')
    assert payload['selected_terminal_sha256']==spec['selected_terminal_sha256']
    assert payload['checkpoint_sha256']==spec['checkpoint_sha256']
    assert payload['base_terminal_sha256']==spec['base_terminal_sha256']
    assert len(payload['state_delta'])==10 and len(payload['row_ids'])==29778
    assert all(torch.isfinite(v).all().item() for v in payload['state_delta'].values())
    optimizer=payload['optimizer'];assert len(optimizer['state'])==10 and len(optimizer['param_groups'])==1
    assert all(float(v['step'])==3723 and torch.isfinite(v['exp_avg']).all().item() and torch.isfinite(v['exp_avg_sq']).all().item() for v in optimizer['state'].values())
    weights.append(dict(arm=arm,path=str(path),bytes=path.stat().st_size,sha256=b['expected'][arm],step=3723,state_tensors=10,optimizer_states=10))
official=env['weight_dirs']['scanrefer'];deps=[dict(path=official['path'],sha256=spec['checkpoint_sha256']),dict(path=spec['base_terminal'],sha256=spec['base_terminal_sha256']),dict(path=spec['selected_terminal'],sha256=spec['selected_terminal_sha256'])]
for dep in deps:assert sha(dep['path'])==dep['sha256'];dep['bytes']=Path(dep['path']).stat().st_size
selected=torch.load(spec['selected_terminal'],map_location='cpu')
model,config,load=build_support_model(cfg,torch.load(official['path'],map_location='cpu'),torch.load(spec['base_terminal'],map_location='cpu'),selected,json.loads(Path(spec['input_manifest']).read_bytes())['data_root'],payloads['content'])
assert load['full_state_tensors']==1314 and load['deployed_support_heads']==1
state=model.state_dict()
assert all(torch.equal(value,state[name]) for name,value in payloads['content']['state_delta'].items())
assert torch.count_nonzero(model.candidate_box_refiner.output.weight).item()==0
assert torch.count_nonzero(model.candidate_box_refiner.output.bias).item()==0
assert all(value.device.type=='cpu' for value in state.values())
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS_CLOSED_CPU_STRICT_RECONSTRUCTION',time_cst=datetime.datetime.now().astimezone().isoformat(),weights=weights,required_dependencies=deps,load=load,full_cpu_state_tensors=len(state),head_exact=True,box_output_zero=True,cuda_initialized=False,neural_forwards=0,optimizer_updates=0,deletions=0)))
'''
client = paramiko.SSHClient(); client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
command = shlex.join(['env', 'CUDA_VISIBLE_DEVICES=', spec['runtime']+'/venv/bin/python', '-B', '-c', code])
stdin, stdout, stderr = client.exec_command(command, timeout=180)
stdin.write(json.dumps(dict(spec=spec, expected=expected))); stdin.channel.shutdown_write()
raw = stdout.read(); err = stderr.read().decode(); assert stdout.channel.recv_exit_status()==0, err
# Model constructors can print warm-cache configuration; retain the actual complete output.
(root/'postrun/checkpoint_inspection.stdout.txt').write_bytes(raw)
(root/'postrun/checkpoint_inspection.stderr.txt').write_text(err,encoding='utf-8')
record = json.loads(raw.decode().splitlines()[-1])
assert record['status']=='PASS_CLOSED_CPU_STRICT_RECONSTRUCTION'
winner = next(row for row in record['weights'] if row['arm']=='content')
archive = Path('C:/Users/gb/.codex/archives/pvg_mask_support_best_20261008')
archive.mkdir(parents=True,exist_ok=True)
dest = archive/'terminal.pth'; assert not dest.exists()
sftp=client.open_sftp(); sftp.get(winner['path'],str(dest)); sftp.close(); client.close()
assert dest.stat().st_size==winner['bytes'] and hashlib.sha256(dest.read_bytes()).hexdigest()==winner['sha256']
record.update(best_local_archive=str(dest),archive_bytes_exact=True,archive_sha256_exact=True,nonbest_weights_copied=0)
receipt_path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
(archive/'RESTORATION_MANIFEST.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=record['status'],full_cpu_state_tensors=record['full_cpu_state_tensors'],best_local_archive=str(dest),bytes=winner['bytes'],sha256=winner['sha256'],neural_forwards=0,weights_deleted=0)),flush=True)
