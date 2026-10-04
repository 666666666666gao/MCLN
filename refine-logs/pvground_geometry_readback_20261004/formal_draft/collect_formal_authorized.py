"""Collect closed text/results/source receipts; leave the metric-best parent chain remote."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

local = Path(__file__).resolve().parent
wait = json.loads((local / 'wait.json').read_bytes())
assert wait['observer_closed'] and not wait['terminal']['controller_alive']
launch = json.loads((local / 'launch.json').read_bytes())
spec = json.loads((local / 'evidence_hidden_fit_spec.json').read_bytes())
target = local / 'complete'
assert not target.exists()
target.mkdir()
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
    password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
probe = '''
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
print(json.dumps([str(path.relative_to(root)) for path in sorted(root.rglob('*'))
    if path.is_file() and path.suffix in ('.json','.jsonl','.log','.exit')]))
'''
_, stdout, stderr = client.exec_command(shlex.join([spec['runtime'] + '/venv/bin/python', '-c', probe,
    launch['root']]), timeout=60)
raw = stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
sftp = client.open_sftp()
files = {}
for relative in json.loads(raw):
    with sftp.open(launch['root'] + '/' + relative, 'rb') as stream:
        contents = stream.read()
    path = target / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    files[relative] = dict(bytes=len(contents), sha256=hashlib.sha256(contents).hexdigest())
sftp.close()
client.close()
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), files=files,
    remote_terminal=wait['terminal'], downloaded_weights=0, created_local_weight_archive=False,
    inference_or_optimizer_replayed=False)
(target / 'INTAKE.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(text_files=len(files), downloaded_weights=0)), flush=True)
