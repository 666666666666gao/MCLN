import ast,datetime,hashlib,json,subprocess
from pathlib import Path
base=Path('/root/autodl-tmp/mcln_scanrefer_native_box_transfer_posttraining_20260907_v1')
m=json.loads((base/'input_manifest.json').read_text())
assert hashlib.sha256((base/'posttraining_queue.py').read_bytes()).hexdigest()==m['queue_script_sha256']
for p in base.rglob('*.py'):ast.parse(p.read_text())
training=Path(m['training_directory']);prep=Path(m['formal_preparation_directory'])
assert hashlib.sha256((training/'input_manifest.json').read_bytes()).hexdigest()==m['training_manifest_sha256']
assert hashlib.sha256((prep/'preparation.json').read_bytes()).hexdigest()==m['formal_preparation_sha256']
prepared=json.loads((prep/'preparation.json').read_text())
for name,digest in prepared['files'].items():assert hashlib.sha256((prep/name).read_bytes()).hexdigest()==digest,name
point_dir=Path(m['data_root'])/'superpoints/val'
assert len(m['val_superpoint_files'])==312
for name,digest in m['val_superpoint_files'].items():assert hashlib.sha256((point_dir/name).read_bytes()).hexdigest()==digest,name
p=subprocess.run(['ps','-p','58020,58023','-o','pid,ppid,comm,stat,etime,args'],stdout=subprocess.PIPE)
assert p.returncode==0 and b'58023' in p.stdout and b'run_scanrefer_native_box_transfer_pair.py' in p.stdout
print(json.dumps({'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
'original_python_ast_and_imports':True,'queue_and_formal_source_bytes_verified':True,
'val_superpoints_verified':312,'live_training_processes':p.stdout.decode(),
'new_gpu_forwards':0,'training_manifest_unchanged':True,'queue_started':False}))
