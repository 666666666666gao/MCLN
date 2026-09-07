import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
spec = json.loads((root / 'env_spec.json').read_bytes())
assert root == Path(spec['root'])
assert sys.version_info[:2] == (3, 7)
assert not (root / 'venv').exists()
started = time.time()
source_receipt = json.loads((root / 'source_bundle_receipt.json').read_bytes())
for directory, item in source_receipt['sources'].items():
    for name, metadata in item['files'].items():
        assert hashlib.sha256((root / directory / name).read_bytes()).hexdigest() == metadata['sha256'], name
for metadata in source_receipt['packages'].values():
    assert hashlib.sha256((root / 'wheels' / metadata['filename']).read_bytes()).hexdigest() == metadata['sha256']
package_code = "import pkg_resources,json; print(json.dumps({x.key:x.version for x in pkg_resources.working_set},sort_keys=True))"
before = subprocess.check_output([sys.executable, '-c', package_code])
(root / 'base_packages_before.json').write_bytes(before)
subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', str(root / 'venv')], check=True)
python = root / 'venv/bin/python'
site = root / 'venv/lib/python3.7/site-packages'
(site / 'mcln_sparse_parent.pth').write_text(spec['sparse_parent_site'] + '\n')
environment = dict(os.environ, **spec['env'])
for phase in spec['pip_phases']:
    subprocess.run([str(python), '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-build-isolation', '--no-cache-dir', '--find-links', str(root / 'wheels')] + phase, env=environment, check=True)

path = root / 'PV-Ground/models/pv_utils.py'
original = path.read_bytes()
unused = b'from pcdet.models.model_utils.basic_block_2d import BasicBlock2D\n'
assert original.count(unused) == 1 and original.count(b'BasicBlock2D') == 1
path.write_bytes(original.replace(unused, b''))
(root / 'source_port.json').write_text(json.dumps({'file': 'PV-Ground/models/pv_utils.py', 'removed_unused_import': unused.decode().strip(), 'before_sha256': hashlib.sha256(original).hexdigest(), 'after_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'model_parameters_changed': 0}, indent=2) + '\n')
(root / 'OpenPCDet/pcdet/version.py').write_text("__version__ = '0.6.0+233f849'\n")
subprocess.run([str(python), str(root / 'setup_pvg_ops.py'), 'build_ext', '--inplace'], cwd=str(root / 'OpenPCDet'), env=environment, check=True)
subprocess.run([str(python), '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-build-isolation', '--no-cache-dir', str(root / 'PV-Ground/pointnet2')], env=environment, check=True)
after = subprocess.check_output([sys.executable, '-c', package_code])
assert before == after
(root / 'base_packages_after.json').write_bytes(after)
subprocess.run([str(python), str(root / 'verify_pvground_runtime.py'), '--output', str(root / 'kernel_receipt.json')], cwd=str(root / 'PV-Ground'), env=environment, check=True)
receipt = {'status': 'pass', 'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
           'spec_sha256': hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
           'kernel_receipt_sha256': hashlib.sha256((root / 'kernel_receipt.json').read_bytes()).hexdigest(),
           'base_packages_unchanged': True, 'base_packages_sha256': hashlib.sha256(before).hexdigest(),
           'elapsed_seconds': time.time() - started, 'full_model_forwards': 0, 'optimizer_steps': 0,
           'agent_follows_doc': 'pending'}
(root / 'build_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('PVG_RUNTIME_BUILD_COMPLETE ' + json.dumps(receipt), flush=True)
