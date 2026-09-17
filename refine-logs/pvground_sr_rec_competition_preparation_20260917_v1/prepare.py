import ast,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).parent
spec=json.loads((root/'spec.json').read_bytes())
def sha(path):
 d=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):d.update(block)
 return d.hexdigest()
for name,digest in spec['files'].items():assert sha(root/name)==digest,name
source=(root/'check.py').read_text();tree=ast.parse(source)
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
torch_line=min(n.lineno for n in ast.walk(main) if isinstance(n,ast.Import) and any(i.name=='torch' for i in n.names))
gate_line=source[:source.index("assert audit['advance_to_nr3d_sr3d_rec']")].count('\n')+1
assert gate_line<torch_line
formal=Path(spec['formal_root']);assert not (formal/'controller.exit').is_file()
assert (str(formal)+'/controller.py').encode() in Path('/proc/'+str(spec['formal_pid'])+'/cmdline').read_bytes()
run=subprocess.run([sys.executable,str(root/'check.py'),'--formal-root',str(formal),'--output',str(root/'must_not_create')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
assert run.returncode!=0 and b'controller.exit' in run.stderr and b'FileNotFoundError' in run.stderr
assert not (root/'must_not_create').exists()
(root/'gate_rejection.log').write_bytes(run.stdout+run.stderr)
batch_root=Path('/root/autodl-tmp/mcln_pvground_sr_native_batch_20260917_v1')
assert (batch_root/'controller.exit').read_text().strip()=='0'
batch_receipt=json.loads((batch_root/'receipt.json').read_bytes())
assert batch_receipt['status']=='pass' and batch_receipt['model_forwards']==0
assets={'sr_parent':{'path':'/root/autodl-tmp/mcln_pvground_sr_checkpoint_inspection_20260908_v1/PV-Ground_SR3D.pth','sha256':'a4a14b0090947177a648703ad6de094246891d89174fffa56fe454f730dbe3dc'},
        'sr_batch':{'path':str(batch_root/'native_batch.pt'),'sha256':batch_receipt['input_sha256']}}
for name,item in assets.items():assert sha(item['path'])==item['sha256'],name
import torch
torch.set_num_threads(1)
batch=torch.load(assets['sr_batch']['path'],map_location='cpu')
sources=batch['native_batch']['sample_dataset']
assert sources==[row['dataset'] for row in batch['rows']]==['sr3d']*4+['scannet']*4
assert batch['native_batch']['language_dataset']==['sr3d']*8
assert not torch.cuda.is_initialized()
record=dict(status='cpu_preparation_pass',time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),assets=assets,actual_batch_sources=sources,gate_rejected_before_torch_import=True,model_forwards=0,backward_calls=0,optimizer_steps=0,formal_rows=0,new_checkpoints=0,missing_actual_sr_f_backward=True,training_jobs_launched=0,script_sha256=sha(root/'check.py'))
(root/'receipt.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
