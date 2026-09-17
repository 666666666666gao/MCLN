import ast,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).parent
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()
source=(root/'check.py').read_text()
compile(source,str(root/'check.py'),'exec')
tree=ast.parse(source)
main=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='main')
imports=[node.lineno for node in ast.walk(main) if isinstance(node,ast.Import) and any(item.name=='torch' for item in node.names)]
gate=source[:source.index("assert audit['advance_to_nr3d_sr3d_rec']")].count('\n')+1
assert gate<min(imports)
formal=Path('/root/autodl-tmp/mcln_pvground_scanrefer_formal_20260917_fixed_memory_v1')
assert not (formal/'controller.exit').exists()
assert not (formal/'decision.json').exists()
out=root/'must_not_create'
run=subprocess.run([sys.executable,str(root/'check.py'),'--formal-root',str(formal),'--output',str(out)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
assert run.returncode!=0 and b'controller.exit' in run.stderr and b'FileNotFoundError' in run.stderr
assert not out.exists()
(root/'gate_rejection.log').write_bytes(run.stdout+run.stderr)
assets={
 'nr_parent':('/root/autodl-tmp/mcln_pvground_nr_checkpoint_inspection_20260908_v1/PV-Ground_NR3D.pth','d2d9afaf9c293c54977f3555a46c7bb2a72d9f80163a3f602dd8426032d7fa5d'),
 'sr_parent':('/root/autodl-tmp/mcln_pvground_sr_checkpoint_inspection_20260908_v1/PV-Ground_SR3D.pth','a4a14b0090947177a648703ad6de094246891d89174fffa56fe454f730dbe3dc'),
 'nr_batch':('/root/autodl-tmp/mcln_pvground_nr_native_batch_20260908_v1/native_batch.pt','7974b2bd74a4c8990501d6ea2264440cc985a2145090aa04efc515394d813f98')}
verified={}
for label,(path,digest) in assets.items():
    assert sha(path)==digest,label
    verified[label]=dict(path=path,bytes=Path(path).stat().st_size,sha256=digest)
training='/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260917_fixed_memory_v1'
assert (training+'/controller.py').encode() in Path('/proc/8666/cmdline').read_bytes()
record=dict(status='prepared_syntax_and_gate_only',time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    script_sha256=sha(root/'check.py'),python=sys.version,assets=verified,gate_rejected_before_torch_import=True,
    model_forwards=0,backward_calls=0,optimizer_steps=0,new_checkpoints=0,formal_rows=0,training_controller_pid=8666,
    missing_actual_e_nr_forward_backward=True,no_nr_sr_job_queued=True)
(root/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
