import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
started=time.time()
assert sys.executable=='/root/miniconda3/envs/bdetr/bin/python'
spec=json.loads((root/'env_spec.json').read_bytes())
assert not (root/'build_receipt.json').exists()
assert (root/'build_repair_v2/controller.exit').read_text().strip()=='1'
source=root/'verify_pvground_runtime.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()==spec['smoke']['script_sha256']
package_code="import pkg_resources,json; print(json.dumps({x.key:x.version for x in pkg_resources.working_set},sort_keys=True))"
before=(root/'base_packages_before.json').read_bytes()
after=subprocess.check_output([sys.executable,'-c',package_code])
assert before==after==(root/'base_packages_after.json').read_bytes()
extensions=[*root.glob('OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/*.so'),*root.glob('OpenPCDet/pcdet/ops/roiaware_pool3d/*.so'),*root.glob('venv/lib/python3.7/site-packages/pointnet2/*.so')]
assert len(extensions)==3
subprocess.run([str(root/'venv/bin/python'),str(source),'--output',str(root/'kernel_receipt.json')],cwd=str(root/'PV-Ground'),env=dict(os.environ,**spec['env']),check=True)
receipt={'status':'pass','time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'spec_sha256':hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
    'kernel_receipt_sha256':hashlib.sha256((root/'kernel_receipt.json').read_bytes()).hexdigest(),
    'extensions':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in extensions},
    'base_packages_unchanged':True,'base_packages_sha256':hashlib.sha256(before).hexdigest(),
    'elapsed_seconds':time.time()-started,'full_model_forwards':0,'optimizer_steps':0,'agent_follows_doc':'pending',
    'build_history':'v1 setuptools packaging error; v2 all extensions built and reference dtype assertion; v3 same binaries with float32 CPU voxel reference',
    'cuda_minor_version_warning_retained':True}
(root/'build_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print('PVG_RUNTIME_BUILD_COMPLETE '+json.dumps(receipt),flush=True)
