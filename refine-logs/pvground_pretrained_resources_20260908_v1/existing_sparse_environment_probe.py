import datetime
import json
import os
from pathlib import Path
import subprocess

interpreter = Path('/root/autodl-tmp/mcln_sparse_runtime_20260906_v3/venv/bin/python')
assert interpreter.exists()
code = '''import importlib.util,json,platform,torch,spconv,cumm
names=['pcdet','torch_scatter','skimage','easydict','yaml']
print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'spconv':spconv.__version__,'cumm':cumm.__version__,'spconv_path':spconv.__file__,'module_paths':{name:(importlib.util.find_spec(name).origin if importlib.util.find_spec(name) else None) for name in names}}))
'''
environment = dict(os.environ, CUDA_VISIBLE_DEVICES='')
result = subprocess.run([str(interpreter), '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment)
assert result.returncode == 0, result.stderr.decode()
report = {'status':'pass', 'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
          'interpreter': str(interpreter), 'versions':json.loads(result.stdout), 'installed_packages':0, 'gpu_forwards':0}
print(json.dumps(report))
