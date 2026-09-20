"""Run CPU-only analysis of completed G and collect its fixed result evidence."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

repo = Path(__file__).resolve().parents[1]
name = 'pvground_semantic_assignment_comparison_20260918_v1'
root = '/root/autodl-tmp/mcln_' + name
local = repo/'refine-logs'/name
candidate = '/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260918_semantic_assignment_v1'
control = '/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260917_task_observation_v1'
formal = '/root/autodl-tmp/mcln_pvground_scanrefer_formal_20260918_semantic_assignment_v1'
audit = '/root/autodl-tmp/mcln_pvground_scanrefer_endpoint_audit_20260918_semantic_assignment_v1'
c = paramiko.SSHClient();c.load_system_host_keys()
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s = c.open_sftp()


def read(path):
    with s.open(path,'rb') as f:f.prefetch();return f.read()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


for path in [candidate, audit, formal]: assert read(path+'/controller.exit').strip() == b'0'
verified = json.loads(read(audit+'/audit.json'))
assert verified['integrity_pass'] and verified['semantic_assignment_verified']
assert verified['receipt_sha256'] == sha(read(candidate+'/receipt.json'))
configs = [read(path+'/spec.json') for path in [control, candidate]]
pair = dict(native_root=control,candidate_root=candidate,formal_root=formal,
    control_source_query=True,control_observation_state=True,
    training_spec_sha256=[sha(raw) for raw in configs],
    source_port_sha256=[sha(read(json.loads(raw)['source_port'])) for raw in configs])
files = {'spec.json': (json.dumps(pair,indent=2)+'\n').encode()}
for filename in ['analyze_pvground_semantic_assignment_results.py',
                 'compare_pvground_semantic_assignment_control.py',
                 'analyze_pvground_fixed_memory_selection.py']:
    files[filename] = (repo/'scripts'/filename).read_bytes()
assert Path(root).name not in s.listdir('/root/autodl-tmp') and not local.exists()
s.mkdir(root);local.mkdir()
for filename, raw in files.items():
    if filename.endswith('.py'):compile(raw,filename,'exec')
    (local/filename).write_bytes(raw)
    with s.open(root+'/'+filename,'wx') as f:f.write(raw)
    assert read(root+'/'+filename) == raw
command = ['/root/miniconda3/envs/bdetr/bin/python','-u',root+'/analyze_pvground_semantic_assignment_results.py',
           '--spec',root+'/spec.json','--out',root+'/summary.json']
_, out, err = c.exec_command("CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 " + ' '.join(map(shlex.quote,command)),timeout=120)
stdout=out.read();stderr=err.read();code=out.channel.recv_exit_status()
for filename, raw in [('analysis.log',stdout),('analysis.stderr',stderr),('analysis.exit',(str(code)+'\n').encode())]:
    (local/filename).write_bytes(raw)
    with s.open(root+'/'+filename,'wx') as f:f.write(raw)
assert code == 0,stderr.decode()
raw=read(root+'/summary.json');(local/'summary.json').write_bytes(raw)
result=json.loads(raw)
for where, stages in [(candidate,['initial','terminal']), (formal,['published_parent','fit_terminal'])]:
    archive=repo/'refine-logs'/Path(where).name.replace('mcln_','',1)
    for stage in stages:
        (archive/stage).mkdir(exist_ok=True)
        (archive/stage/'receipt.json').write_bytes(read(where+'/'+stage+'/receipt.json'))
    if where == formal:
        (archive/'protocol.json').write_bytes(read(where+'/protocol.json'))
    else:
        manifest=json.loads(read(where+'/spec.json'))['input_manifest']
        (archive/'input_manifest.json').write_bytes(read(manifest))
s.close();c.close()
print(json.dumps({key:result[key] for key in ['status','formal_primary_percent','protected_v99_gap_hits','pairs','promotion']}),flush=True)
