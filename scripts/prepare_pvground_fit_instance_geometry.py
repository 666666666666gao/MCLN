"""Launch CPU analysis of already saved training outputs; no model replay."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

repo = Path(__file__).resolve().parents[1]
name = 'pvground_fit_instance_geometry_20260917_v1'
root = '/root/autodl-tmp/mcln_' + name
archive = repo / 'refine-logs' / name
source = (repo / 'scripts/analyze_pvground_fit_instance_geometry.py').read_bytes()
compile(source, 'analyze.py', 'exec')
c = paramiko.SSHClient()
c.load_system_host_keys()
c.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
s = c.open_sftp()
assert Path(root).name not in s.listdir('/root/autodl-tmp')
s.mkdir(root)
archive.mkdir()
controller = '''import os,subprocess
from pathlib import Path
root=Path(__file__).parent
(root/'controller.pid').write_text(str(os.getpid())+'\\n')
with (root/'run.log').open('xb') as log:
    result=subprocess.run(['/root/miniconda3/envs/bdetr/bin/python','-u',str(root/'analyze.py'),'--output',str(root)],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT)
(root/'controller.exit').write_text(str(result.returncode)+'\\n')
raise SystemExit(result.returncode)
'''
files = {'analyze.py': source, 'controller.py': controller.encode(),
    'plan.md': b'Analyze only saved normal D outputs on the preselected128 fit scenes. Bind source files and verify original scene point SHA and root boxes. Compare all annotated objects using existing overlap categories; geometry proxy only. Report chosen and highest-scoring qualifying alternative separately. No model/GPU/optimizer/formal evaluation or deployment change. Do not infer semantic identity from maximum overlap.\n'}
for filename, raw in files.items():
    (archive / filename).write_bytes(raw)
    with s.open(root + '/' + filename, 'wx') as stream:
        stream.write(raw)
command = ['screen', '-dmS', 'mcln_pvg_fit_instance_geometry_v1', '/root/miniconda3/envs/bdetr/bin/python', '-u', root + '/controller.py']
_, out, err = c.exec_command(' '.join(map(shlex.quote, command)), timeout=30)
assert out.channel.recv_exit_status() == 0, err.read().decode()
_, out, err = c.exec_command('pgrep -af ' + shlex.quote('^/root/miniconda3/envs/bdetr/bin/python -u ' + root + '/controller.py$'), timeout=30)
process = out.read().decode().strip()
assert out.channel.recv_exit_status() == 0 and process, err.read().decode()
record = dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    process=process, root=root, script_sha256=hashlib.sha256(source).hexdigest(), model_forwards=0,
    optimizer_steps=0, formal_rows=0, first_check_after_seconds=180)
raw = (json.dumps(record, indent=2) + '\n').encode()
(archive / 'launch.json').write_bytes(raw)
with s.open(root + '/launch.json', 'wx') as stream:
    stream.write(raw)
s.close()
c.close()
print(json.dumps(record), flush=True)
