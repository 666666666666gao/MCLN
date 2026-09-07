import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
archive = repo / 'refine-logs/native_score_fit_20260907_v2'
remote = '/root/autodl-tmp/mcln_native_score_fit_20260907_v2'
source_parent = '/root/autodl-tmp/mcln_native_mask_geometry_source_preparation_20260907_v1'
historical_dir = '/root/autodl-tmp/mcln_scanrefer_mesh_teacher_transfer_20260907_v1'
old = json.loads((repo / 'refine-logs/scanrefer_mesh_teacher_transfer_20260907_v1/input_manifest.json').read_text())
source_manifest = repo / 'refine-logs/native_mask_geometry_source_preparation_20260907_v1/native_source_manifest.json'
files = ['run_scanrefer_native_score_fit_audit.py', 'scanrefer_data_contract.py', 'scanrefer_rec_evaluation.py']
archive.mkdir()
c = paramiko.SSHClient()
c.load_system_host_keys()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
s = c.open_sftp()
probe = "import os,socket,shutil,json,subprocess;print(json.dumps(dict(uid=os.getuid(),host=socket.gethostname(),cwd=os.getcwd(),disk=shutil.disk_usage('/root/autodl-tmp')._asdict(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits']).decode(),processes=subprocess.check_output(['ps','-eo','pid,ppid,args']).decode())))"
_, out, err = c.exec_command('/root/miniconda3/envs/bdetr/bin/python -c ' + shlex.quote(probe), timeout=40)
raw = out.read()
assert out.channel.recv_exit_status() == 0, err.read().decode()
preflight = json.loads(raw)
assert preflight['uid'] == 0 and preflight['host'] == 'autodl-container-c7cb4299a4-24929f53'
assert preflight['disk']['free'] > 2 * 1024**3
assert int(preflight['gpu'].split(',')[1]) < 500
preflight['time_cst'] = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
(archive / 'preflight.json').write_bytes(json.dumps(preflight, indent=2).encode())
for relative in ['docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md']:
    with s.open('/home/gb/new butd/butd_detr-main/MCLN-main/' + relative, 'rb') as stream:
        assert stream.read() == (repo / relative).read_bytes()
with s.open(source_parent + '/model_source/native_source_manifest.json', 'rb') as stream:
    assert stream.read() == source_manifest.read_bytes()
manifest = {k: old[k] for k in ['data_root', 'rows', 'selected_row_ids', 'split_salt', 'train_superpoint_files']}
manifest.update(schema='mcln-native-score-fit-audit-v1', model_source=source_parent + '/model_source',
                source_manifest=source_parent + '/model_source/native_source_manifest.json',
                source_manifest_sha256=hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
                artifacts=old['artifacts'], historical_rows=historical_dir + '/rows.json',
                historical_rows_sha256='69f377364985444bc4c45cd86da7d8db0fef6d1dfdec31ec8e38fd48df0d958b',
                environment_reuse=old['environment_reuse'], optimizer_steps=0, checkpoint_writes=0, formal_rows=0,
                files={name: hashlib.sha256((repo / 'scripts' / name).read_bytes()).hexdigest() for name in files})
(archive / 'input_manifest.json').write_bytes(json.dumps(manifest, indent=2, sort_keys=True).encode() + b'\n')
s.mkdir(remote)
for name in files:
    s.put(str(repo / 'scripts' / name), remote + '/' + name)
s.put(str(archive / 'input_manifest.json'), remote + '/input_manifest.json')
s.put(str(repo / 'docs/SCANREFER_NATIVE_SCORE_FIT_AUDIT_PLAN_2026-09-07.md'), remote + '/plan.md')
compile_code = "from pathlib import Path;import sys;[compile(Path(p).read_bytes(),p,'exec') for p in sys.argv[1:]];print('SYNTAX_OK')"
_, out, err = c.exec_command('/root/miniconda3/envs/bdetr/bin/python -c ' + shlex.quote(compile_code) + ' ' + ' '.join(shlex.quote(remote + '/' + n) for n in files), timeout=30)
compile_output = out.read().decode()
assert out.channel.recv_exit_status() == 0, err.read().decode()
controller = ('#!/bin/bash\n'
              'export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false\n'
              'cd ' + shlex.quote(remote) + '\n'
              'flock -n /root/autodl-tmp/mcln_v99_backbone_gpu0.lock /root/miniconda3/envs/bdetr/bin/python -u run_scanrefer_native_score_fit_audit.py --manifest input_manifest.json\n'
              'status=$?\nprintf "%s\\n" "$status" > controller.exit\nexit "$status"\n')
(archive / 'controller.sh').write_bytes(controller.encode())
s.put(str(archive / 'controller.sh'), remote + '/controller.sh')
command = 'screen -L -Logfile ' + shlex.quote(remote + '/run.log') + ' -dmS mcln_native_score_fit_20260907_v2 bash ' + shlex.quote(remote + '/controller.sh')
_, out, err = c.exec_command(command, timeout=30)
out.read()
assert out.channel.recv_exit_status() == 0, err.read().decode()
_, out, err = c.exec_command('ps -eo pid,ppid,args', timeout=30)
processes = out.read().decode()
assert out.channel.recv_exit_status() == 0
lines = [line for line in processes.splitlines() if remote in line or 'python -u run_scanrefer_native_score_fit_audit.py' in line]
assert any('python -u run_scanrefer_native_score_fit_audit.py' in line for line in lines)
launch = {'remote': remote, 'screen': 'mcln_native_score_fit_20260907_v2', 'command': command,
          'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
          'processes': lines, 'compile': compile_output, 'manifest_sha256': hashlib.sha256((archive / 'input_manifest.json').read_bytes()).hexdigest()}
(archive / 'launch.json').write_bytes(json.dumps(launch, indent=2).encode() + b'\n')
(archive / 'launch_from_local.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({'launch': launch, 'free_bytes': preflight['disk']['free'], 'gpu': preflight['gpu']}), flush=True)
s.close()
c.close()
