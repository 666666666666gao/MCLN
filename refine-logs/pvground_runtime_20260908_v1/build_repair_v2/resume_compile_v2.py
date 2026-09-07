import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
repair = root / 'build_repair_v2'
started = time.time()
spec = json.loads((root / 'env_spec.json').read_bytes())
assert sys.executable == '/root/miniconda3/envs/bdetr/bin/python'
assert (root / 'controller.exit').read_text().strip() == '1'
assert not (root / 'build_receipt.json').exists()
python = root / 'venv/bin/python'
package_code = "import pkg_resources,json; print(json.dumps({x.key:x.version for x in pkg_resources.working_set},sort_keys=True))"
before = (root / 'base_packages_before.json').read_bytes()
assert before == subprocess.check_output([sys.executable, '-c', package_code])
environment = dict(os.environ, **spec['env'])
package = json.loads((repair / 'package_source.json').read_bytes())
wheel = repair / package['filename']
assert hashlib.sha256(wheel.read_bytes()).hexdigest() == package['digests']['sha256']
subprocess.run([str(python), '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-build-isolation', '--no-cache-dir', str(wheel)], env=environment, check=True)
check = "import setuptools;from torch.utils.cpp_extension import packaging; assert setuptools.__version__=='58.0.4';assert packaging.version.parse('11.6').major == packaging.version.parse('11.1').major == 11;print('BUILD_TOOL_VERSION_CHECK_PASS')"
subprocess.run([str(python), '-c', check], env=environment, check=True)
subprocess.run([str(python), str(root / 'setup_pvg_ops.py'), 'build_ext', '--inplace'], cwd=str(root / 'OpenPCDet'), env=environment, check=True)
subprocess.run([str(python), '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-build-isolation', '--no-cache-dir', str(root / 'PV-Ground/pointnet2')], env=environment, check=True)
after = subprocess.check_output([sys.executable, '-c', package_code])
assert before == after
(root / 'base_packages_after.json').write_bytes(after)
subprocess.run([str(python), str(root / 'verify_pvground_runtime.py'), '--output', str(root / 'kernel_receipt.json')], cwd=str(root / 'PV-Ground'), env=environment, check=True)
receipt = {'status':'pass','time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'spec_sha256':hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'kernel_receipt_sha256':hashlib.sha256((root/'kernel_receipt.json').read_bytes()).hexdigest(),'base_packages_unchanged':True,'base_packages_sha256':hashlib.sha256(before).hexdigest(),'elapsed_seconds':time.time()-started,'full_model_forwards':0,'optimizer_steps':0,'agent_follows_doc':'pending','repair':'isolated setuptools47.1.0 -> 58.0.4; upstream CUDA minor-version warning retained'}
(root/'build_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('PVG_RUNTIME_BUILD_COMPLETE '+json.dumps(receipt),flush=True)
