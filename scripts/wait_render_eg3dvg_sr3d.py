"""Wait for audited real Sr cases, then download and render them locally.

No model inference or changes to the remote GPU queue. The caller supplies the
already authorized SSH credential through the process environment.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import paramiko


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


repo = Path(__file__).resolve().parents[1]
temporary = Path('D:/Program Files/UserCache/gb/codex/tmp/eg3dvg_acceptance_20260920')
destination = Path('C:/Users/gb/Desktop/document/EG3DVG_Sr3D_failure_visualizations_20260920')
remote_run = '/root/autodl-tmp/mcln_eg3dvg_sr3d_transfer_20260920_v1'
remote_cases = '/root/autodl-tmp/mcln_eg3dvg_sr3d_failure_visuals_20260920_v1'
alignment = Path('C:/Users/gb/Desktop/document/EG3DVG_3D_failure_visualizations_20260920/paper_style_dense/scans_axis_alignment_matrices.json')
alignment_sha = '39029c41a9baf3851e82d42f0b2093dac16b28c1e6f08bfa7456d5e4d54e4dee'
assert hashlib.sha256(alignment.read_bytes()).hexdigest() == alignment_sha
spec = json.loads((temporary / 'sr_render_queue_spec.json').read_text())
for name, digest in spec['scripts'].items():
    assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == digest, name
while True:
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect('region-9.autodl.pro', port=33476, username='root',
                   password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
    sftp = client.open_sftp()
    names = sftp.listdir(remote_run)
    ready = 'failure_visual_export.exit' in names
    if ready:
        with sftp.open(remote_run + '/failure_visual_export.exit', 'rb') as f:
            assert f.read().decode().strip() == '0'
        with sftp.open(remote_run + '/formal/audit.json', 'rb') as f:
            assert json.loads(f.read())['integrity_pass']
    sftp.close()
    client.close()
    (temporary / 'sr_render_queue_status.json').write_text(json.dumps(
        {'time_cst': now(), 'pid': os.getpid(), 'stage': 'export_ready' if ready else 'waiting_export',
         'new_model_forwards': 0, 'manual_visual_inspection': 'pending'}, indent=2), encoding='utf-8')
    if ready:
        break
    time.sleep(300)
destination.mkdir(exist_ok=True)
(destination / 'paper_style_dense').mkdir(exist_ok=True)
shutil.copyfile(alignment, destination / 'paper_style_dense/scans_axis_alignment_matrices.json')
uv = shutil.which('uv')
assert uv
for stage, command in [
    ('collect', [sys.executable, '-u', str(repo / 'scripts/collect_eg3dvg_failure_visuals.py'),
                  '--remote', remote_cases, '--root', str(destination)]),
    ('render', [uv, 'run', '--offline', '--no-project', '--with', 'pyvista', '--with', 'scipy',
                '--with', 'pillow', 'python', str(repo / 'scripts/render_eg3dvg_failure_visuals.py'),
                '--root', str(destination)]),
    ('package', [uv, 'run', '--offline', '--no-project', '--with', 'pillow', 'python',
                 str(repo / 'scripts/package_eg3dvg_failure_visuals.py'), '--root', str(destination)]),
]:
    with (temporary / ('sr_local_' + stage + '.log')).open('xb') as log:
        code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
    (temporary / ('sr_local_' + stage + '.exit')).write_text(str(code), encoding='utf-8')
    if code:
        raise SystemExit(code)
result = json.loads((destination / 'delivery_verification.json').read_text(encoding='utf-8'))
assert result['sr3d_complete'] and result['new_model_forwards'] == 0
(temporary / 'sr_render_queue_status.json').write_text(json.dumps(
    {'time_cst': now(), 'pid': os.getpid(), 'stage': 'rendered_and_packaged',
     'destination': str(destination), 'new_model_forwards': 0,
     'manual_visual_inspection': 'pending'}, indent=2), encoding='utf-8')
print('Sr3D real-case rendering complete; visual inspection pending.', flush=True)
