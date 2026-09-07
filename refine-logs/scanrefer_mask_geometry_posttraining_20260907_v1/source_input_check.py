import ast,datetime,hashlib,json,subprocess
from pathlib import Path
base=Path('/root/autodl-tmp/mcln_scanrefer_mask_geometry_posttraining_20260907_v1');m=json.loads((base/'input_manifest.json').read_text())
assert hashlib.sha256((base/'posttraining_queue.py').read_bytes()).hexdigest()==m['queue_script_sha256']
for root in [base,Path(m['formal_preparation_directory'])]:
 for p in root.rglob('*.py'):ast.parse(p.read_text())
training=Path(m['training_directory']);prep=Path(m['formal_preparation_directory'])
assert hashlib.sha256((training/'input_manifest.json').read_bytes()).hexdigest()==m['training_manifest_sha256']
t=json.loads((training/'input_manifest.json').read_text())
for name,digest in t['files'].items():assert hashlib.sha256((training/name).read_bytes()).hexdigest()==digest,name
assert hashlib.sha256((prep/'preparation.json').read_bytes()).hexdigest()==m['formal_preparation_sha256']
prepared=json.loads((prep/'preparation.json').read_text())
for name,digest in prepared['files'].items():assert hashlib.sha256((prep/name).read_bytes()).hexdigest()==digest,name
for name,digest in m['val_superpoint_files'].items():assert hashlib.sha256((Path(m['data_root'])/'superpoints/val'/name).read_bytes()).hexdigest()==digest,name
p=subprocess.run(['ps','-p','62966,62969','-o','pid,ppid,comm,stat,etime,args'],stdout=subprocess.PIPE)
assert p.returncode==0 and b'62969' in p.stdout and b'run_scanrefer_mask_geometry_pair.py' in p.stdout
print(json.dumps({'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
'source_ast_and_bytes_verified':True,'val_superpoints_verified':312,'live_training':p.stdout.decode(),
'new_gpu_forwards':0,'training_files_unchanged':True,'queue_started':False}))
