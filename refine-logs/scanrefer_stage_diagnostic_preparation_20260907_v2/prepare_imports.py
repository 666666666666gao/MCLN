import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
local = repo / 'refine-logs/scanrefer_stage_diagnostic_preparation_20260907_v2'
remote = '/root/autodl-tmp/mcln_scanrefer_stage_diagnostic_preparation_20260907_v2'
source = '/root/autodl-tmp/mcln_scanrefer_local_visual_preflight_20260906_v2/model_source'
names = ['scripts/scanrefer_joint_readout.py', 'scripts/scanrefer_rec_evaluation.py',
         'scripts/scanrefer_data_contract.py']
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
files = {name: (repo / name).read_bytes() for name in names}
check = '''import datetime, hashlib, importlib, json, os, sys
from pathlib import Path
root = Path(ROOT)
source = Path(SOURCE)
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
os.chdir(str(source))
sys.path.insert(0, str(source))
import scripts
scripts.__path__ = [str(root / 'scripts'), str(source / 'scripts')]
names = ['main_utils', 'train_dist_mod', 'models.rec_reranker', 'models.candidate_local_visual',
         'scripts.run_frozen_v99_pareto_contextual_official', 'scripts.scanrefer_joint_readout',
         'scripts.scanrefer_rec_evaluation', 'scripts.scanrefer_data_contract',
         'scripts.trace_scanrefer_readout_stages', 'scripts.scanrefer_stage_diagnostics']
loaded = {}
for name in names:
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    expected = root if name in names[5:] else source
    assert path == expected / (name.replace('.', '/') + '.py'), (name, str(path))
    loaded[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
import torch
assert not torch.cuda.is_available()
result = {'status': 'runtime_imports_pass', 'modules': loaded, 'torch': torch.__version__,
          'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
          'gpu_forwards': 0, 'data_rows_loaded': 0, 'optimizer_updates': 0, 'checkpoint_writes': 0}
with (root / 'import_receipt.json').open('x') as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
print(json.dumps(result))
'''.replace('ROOT', repr(remote)).replace('SOURCE', repr(source))
files['check_imports.py'] = check.encode()
files['prepare_imports.py'] = Path(__file__).read_bytes()
for name, raw in files.items():
    (local / name).write_bytes(raw)
    with sftp.open(remote + '/' + name, 'wx') as stream:
        stream.write(raw)
command = ('cd ' + shlex.quote(remote)
           + " && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"
           + ' PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/bdetr/bin/python check_imports.py'
           + ' > import_check.log 2>&1')
_, output, error = client.exec_command(command, timeout=60)
output.read()
error_text = error.read().decode()
status = output.channel.recv_exit_status()
with sftp.open(remote + '/import_check.log', 'rb') as stream:
    log = stream.read()
(local / 'import_check.log').write_bytes(log)
assert status == 0, log.decode() + error_text
with sftp.open(remote + '/import_receipt.json', 'rb') as stream:
    receipt_raw = stream.read()
(local / 'import_receipt.json').write_bytes(receipt_raw)
binding = {'existing_manifest_sha256': hashlib.sha256((local / 'manifest.json').read_bytes()).hexdigest(),
           'added_files': {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()},
           'import_receipt_sha256': hashlib.sha256(receipt_raw).hexdigest(),
           'actual_data_or_model_forward_executed': False}
raw = (json.dumps(binding, indent=2, sort_keys=True) + '\n').encode()
(local / 'import_binding.json').write_bytes(raw)
with sftp.open(remote + '/import_binding.json', 'wx') as stream:
    stream.write(raw)
sftp.close()
client.close()
print(json.dumps({'status': 'runtime_imports_pass', 'modules': 10,
                  'import_receipt_sha256': binding['import_receipt_sha256']}))
