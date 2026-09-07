import datetime
import importlib.util
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import pkg_resources
import torch

assert os.environ.get('CUDA_VISIBLE_DEVICES', '') == ''
modules = ['spconv', 'cumm', 'pcdet', 'torch_scatter', 'transformers', 'easydict', 'yaml', 'tensorview', 'skimage']
names = ['spconv', 'cumm', 'torch', 'transformers', 'easydict', 'pyyaml', 'numpy', 'scipy', 'ninja']
packages = {d.key: d.version for d in pkg_resources.working_set if any(d.key.startswith(n) for n in names)}
gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used', '--format=csv,noheader'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
nvcc = subprocess.run(['/usr/local/cuda/bin/nvcc', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
assert gpu.returncode == 0 and nvcc.returncode == 0
root = Path('/root/autodl-tmp')
report = {
    'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'python': platform.python_version(), 'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
    'packages': packages, 'module_paths': {name: (importlib.util.find_spec(name).origin if importlib.util.find_spec(name) else None) for name in modules},
    'gpu': gpu.stdout.decode().strip(), 'nvcc': nvcc.stdout.decode().strip(),
    'free_bytes': shutil.disk_usage(str(root)).free,
    'cuda_environment': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
    'installed_packages': 0, 'gpu_forwards': 0,
}
print(json.dumps(report))
