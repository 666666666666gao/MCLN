"""Run one CPU export analysis and retain its exact source/results."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

repo = Path(__file__).resolve().parents[1]
name = 'pvground_fixed_memory_selection_20260917_v1'
root = '/root/autodl-tmp/mcln_' + name
archive = repo / 'refine-logs' / name
source = (repo / 'scripts/analyze_pvground_fixed_memory_selection.py').read_bytes()
compile(source, 'analyze.py', 'exec')
c = paramiko.SSHClient()
c.load_system_host_keys()
c.connect('region-9.autodl.pro', port=33476, username='root',
          password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
s = c.open_sftp()
assert Path(root).name not in s.listdir('/root/autodl-tmp')
s.mkdir(root)
archive.mkdir()
with s.open(root + '/analyze.py', 'wx') as stream:
    stream.write(source)
(archive / 'analyze.py').write_bytes(source)
args = ['/root/miniconda3/envs/bdetr/bin/python', '-u', root + '/analyze.py',
        '--d-root', '/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260917_task_observation_v1',
        '--e-root', '/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260917_fixed_memory_v1',
        '--output', root + '/analysis.json']
command = ' '.join(map(shlex.quote, args)) + ' > ' + shlex.quote(root + '/run.log') + ' 2>&1'
_, out, err = c.exec_command(command, timeout=60)
code = out.channel.recv_exit_status()
with s.open(root + '/analysis.exit', 'wx') as stream:
    stream.write(str(code) + '\n')
for filename in ['run.log', 'analysis.exit']:
    s.get(root + '/' + filename, str(archive / filename))
assert code == 0, (archive / 'run.log').read_text()
s.get(root + '/analysis.json', str(archive / 'analysis.json'))
result = json.loads((archive / 'analysis.json').read_bytes())
assert result['script_sha256'] == hashlib.sha256(source).hexdigest()
record = dict(time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              status='complete', exit=code, root=root, command=args,
              script_sha256=result['script_sha256'],
              result_sha256=hashlib.sha256((archive / 'analysis.json').read_bytes()).hexdigest(),
              model_forwards=0, optimizer_steps=0, formal_rows=0)
raw = (json.dumps(record, indent=2) + '\n').encode()
(archive / 'receipt.json').write_bytes(raw)
with s.open(root + '/receipt.json', 'wx') as stream:
    stream.write(raw)
s.close()
c.close()
print(json.dumps(result), flush=True)
