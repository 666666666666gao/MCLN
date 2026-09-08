import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import textwrap
import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
archive=repo/'refine-logs/scanrefer_detection_augmentation_20260908_v1'
archive.mkdir()
root='/root/autodl-tmp/mcln_scanrefer_detection_augmentation_20260908_v1'
runtime='/root/autodl-tmp/mcln_pvground_runtime_20260908_v1'
original=Path('C:/Users/gb/.codex/tmp/pvg_actual_joint_det_dataset_20260908.py').read_text(encoding='utf-8')
rotation="            all_det_pts = rot_z(all_det_pts, augmentations['theta_z'])\n            all_det_pts = rot_x(all_det_pts, augmentations['theta_x'])\n            all_det_pts = rot_y(all_det_pts, augmentations['theta_y'])\n"
flips="            if augmentations.get('yz_flip', False):\n                all_det_pts[:, 0] = -all_det_pts[:, 0]\n            if augmentations.get('xz_flip', False):\n                all_det_pts[:, 1] = -all_det_pts[:, 1]\n"
assert original.count(rotation+flips)==1
fixed=original.replace(rotation+flips,flips+rotation)
tree=ast.parse(fixed)
node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_get_detected_objects')
method=textwrap.dedent('\n'.join(fixed.splitlines()[node.lineno-1:node.end_lineno]))+'\n'
repo_node=next(n for n in ast.walk(ast.parse((repo/'src/joint_det_dataset.py').read_bytes())) if isinstance(n,ast.FunctionDef) and n.name=='_get_detected_objects')
assert ast.dump(node,include_attributes=False)==ast.dump(repo_node,include_attributes=False)
audit=(repo/'scripts/audit_scanrefer_detection_augmentation.py').read_bytes()
compile(audit,'audit.py','exec')
spec=dict(input_manifest='/root/autodl-tmp/mcln_scanrefer_object_appearance_pair_20260908_v1/input_manifest.json',
    runtime=runtime, rows=32,seed=2027,selection='SHA256 scanrefer_det_affine_v1 plus row ID, one row per physical fit scene',
    native_dataset_sha256='4a8edacf2c59c8ded76697153d96e8eb078b70146222c187e98b0e34f64bc77e',
    fixed_dataset_sha256=hashlib.sha256(fixed.encode()).hexdigest(),script_sha256=hashlib.sha256(audit).hexdigest(),
    only_change='detected boxes flip before rotation, matching actual points; no Z or other augmentation change',formal_rows=0)
controller='''import json,os,subprocess
from pathlib import Path
root=Path(__file__).parent
runtime=Path("/root/autodl-tmp/mcln_pvground_runtime_20260908_v1")
env=os.environ.copy()
env.update(json.loads((runtime/"env_spec.json").read_bytes())["env"])
env.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
(root/"controller.pid").write_text(str(os.getpid())+"\\n")
with (root/"run.log").open("xb") as log:
    r=subprocess.run([str(runtime/"venv/bin/python"),"-u",str(root/"audit.py"),"--spec",str(root/"spec.json")],env=env,stdout=log,stderr=subprocess.STDOUT)
(root/"controller.exit").write_text(str(r.returncode)+"\\n")
raise SystemExit(r.returncode)
'''.encode()
files={'audit.py':audit,'fixed_dataset.py':fixed.encode(),'fixed_detected_objects.py':method.encode(),'controller.py':controller,'spec.json':(json.dumps(spec,indent=2)+'\n').encode()}
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
with sftp.open(runtime+'/env_spec.json','rb') as f: env=json.loads(f.read())
assert hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest()=='966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c'
code="import json,os,socket,shutil,subprocess;print(json.dumps(dict(uid=os.getuid(),hostname=socket.gethostname(),disk_free=shutil.disk_usage('/root/autodl-tmp').free,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader']).decode())))"
_,out,err=client.exec_command('/root/miniconda3/bin/python -c '+shlex.quote(code),timeout=30)
preflight=json.loads(out.read());assert out.channel.recv_exit_status()==0 and not err.read() and preflight['uid']==0
sftp.mkdir(root)
for name,raw in files.items():
    (archive/name).write_bytes(raw)
    with sftp.open(root+'/'+name,'wb') as f:f.write(raw)
argv=['screen','-dmS','mcln_scan_det_affine_cpu_v1','/root/miniconda3/bin/python',root+'/controller.py']
_,out,err=client.exec_command(' '.join(shlex.quote(s) for s in argv),timeout=30)
assert out.channel.recv_exit_status()==0 and not err.read()
record=dict(root=root,argv=argv,preflight=preflight,time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),estimated_seconds=180)
(archive/'launch.json').write_bytes((json.dumps(record,indent=2)+'\n').encode())
print(json.dumps(record))
sftp.close();client.close()
