import ast
import hashlib
import json
import os
from pathlib import Path
import paramiko

client=paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
root='/root/autodl-tmp/mcln_scanrefer_object_appearance_native_20260908_v1/model_source'
with sftp.open(root+'/src/joint_det_dataset.py','rb') as f:
    raw=f.read()
tree=ast.parse(raw)
lines=raw.decode().splitlines()
functions={n.name:'\n'.join(lines[n.lineno-1:n.end_lineno]) for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in ['_augment','_get_detected_objects','__getitem__','_get_all_bboxes','_get_target_bbox']}
out=Path('C:/Users/gb/.codex/tmp/pvg_scan_augmentation_source_20260908.json')
out.write_text(json.dumps(dict(source=root+'/src/joint_det_dataset.py',sha256=hashlib.sha256(raw).hexdigest(),functions=functions),indent=2),encoding='utf-8')
Path('C:/Users/gb/.codex/tmp/pvg_actual_joint_det_dataset_20260908.py').write_bytes(raw)
print(json.dumps(dict(sha256=hashlib.sha256(raw).hexdigest(),functions={k:v for k,v in functions.items() if k in ['_augment','_get_detected_objects']})))
sftp.close()
client.close()
