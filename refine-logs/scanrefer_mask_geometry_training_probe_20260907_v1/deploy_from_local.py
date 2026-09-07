import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
local = repo / 'refine-logs/scanrefer_mask_geometry_training_probe_20260907_v1'
remote = '/root/autodl-tmp/mcln_scanrefer_mask_geometry_training_probe_20260907_v1'
local.mkdir(exist_ok=False)
(local / 'scripts').mkdir()
(local / 'tests').mkdir()
base = json.loads((repo / 'refine-logs/scanrefer_frozen_readout_probe_20260907_v1/input_manifest.json').read_bytes())
files = {}
for name in ['scripts/probe_scanrefer_mask_geometry_training.py', 'scripts/scanrefer_data_contract.py', 'scripts/scanrefer_joint_readout.py', 'scripts/prototype_probability_geometry.py', 'tests/test_mask_geometry_gradient_contract.py', 'tests/test_probability_geometry_prototype.py', 'scripts/native_mask_geometry_supervision.py', 'scripts/native_teacher_box_transfer.py', 'tests/test_native_mask_geometry_supervision.py']:
    raw = (repo / name).read_bytes()
    (local / name).write_bytes(raw)
    files[name] = hashlib.sha256(raw).hexdigest()
(local / 'scripts/__init__.py').write_bytes(b'')
files['scripts/__init__.py'] = hashlib.sha256(b'').hexdigest()
manifest = {k: base[k] for k in ['model_source', 'source_manifest_sha256', 'artifacts', 'data_root', 'train_superpoint_files', 'split_salt', 'split_protocol', 'split_protocol_sha256', 'environment_reuse']}
manifest.update(schema='mcln-mask-geometry-training-probe-v1', files=files, real_train_rows=16, batch_size=4,
    disposable_optimizer_steps_per_arm=2, checkpoint_writes=0, formal_rows=0, new_network_modules=0,
    purpose='Actual current-root Hungarian geometry supervision; fixed16 fit gradients and two disposable updates per arm; no saved weights.')
(local / 'input_manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n', encoding='utf-8')
controller = '''#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
cd {remote}
flock -n /root/autodl-tmp/mcln_v99_backbone_gpu0.lock /root/miniconda3/envs/bdetr/bin/python -u scripts/probe_scanrefer_mask_geometry_training.py --manifest {remote}/input_manifest.json
status=$?
printf '%s\\n' "$status" > controller.exit
exit "$status"
'''.format(remote=remote)
(local / 'controller.sh').write_bytes(controller.encode())
c = paramiko.SSHClient()
c.load_system_host_keys()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
s = c.open_sftp()
master = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
remote_master = '/home/gb/new butd/butd_detr-main/MCLN-main/' + master
assert hashlib.sha256(s.open(remote_master, 'rb').read()).hexdigest() == hashlib.sha256((repo / master).read_bytes()).hexdigest()
s.mkdir(remote)
s.mkdir(remote + '/scripts')
s.mkdir(remote + '/tests')
for path in local.rglob('*'):
    if path.is_file():
        s.put(str(path), remote + '/' + path.relative_to(local).as_posix())
syntax = "import ast,pathlib; p=pathlib.Path(" + repr(remote + '/scripts/probe_scanrefer_mask_geometry_training.py') + "); ast.parse(p.read_text()); print('Python3.7 syntax PASS')"
_, out, err = c.exec_command('/root/miniconda3/envs/bdetr/bin/python -c ' + shlex.quote(syntax))
text = out.read().decode() + err.read().decode()
assert out.channel.recv_exit_status() == 0, text
(local / 'syntax_check.txt').write_text(text, encoding='utf-8')
test_command = 'cd ' + shlex.quote(remote) + ' && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=' + shlex.quote(remote + ':' + manifest['model_source']) + ' /root/miniconda3/envs/bdetr/bin/python -m pytest -q tests'
_, out, err = c.exec_command(test_command)
test_output = out.read().decode() + err.read().decode()
(local / 'cpu_tests.txt').write_text(test_output, encoding='utf-8')
print(test_output, flush=True)
assert out.channel.recv_exit_status() == 0, test_output
_, out, err = c.exec_command('nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits')
gpu = int(out.read().decode().strip())
assert gpu < 500
command = 'screen -L -Logfile ' + shlex.quote(remote + '/run.log') + ' -dmS mcln_mask_geometry_training_probe_v1 bash ' + shlex.quote(remote + '/controller.sh')
_, out, err = c.exec_command(command)
assert out.channel.recv_exit_status() == 0, err.read().decode()
_, out, err = c.exec_command("ps -eo pid,ppid,stat,etime,args | grep '[p]robe_scanrefer_mask_geometry_training'\nscreen -ls")
live = out.read().decode()
assert 'scripts/probe_scanrefer_mask_geometry_training.py' in live
proof = {'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'screen': 'mcln_mask_geometry_training_probe_v1', 'remote_directory': remote, 'live': live,
    'estimated_runtime_seconds': 180, 'prelaunch_gpu_mib': gpu, 'disposable_optimizer_steps_per_arm': 2, 'formal_rows': 0,
    'input_manifest_sha256': hashlib.sha256((local / 'input_manifest.json').read_bytes()).hexdigest()}
(local / 'launch.json').write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8')
(local / 'deploy_from_local.py').write_bytes(Path(__file__).read_bytes())
s.close()
c.close()
print(json.dumps(proof), flush=True)
