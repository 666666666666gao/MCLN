"""Collect only closed result/source receipts; do not load or archive weights."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko


local = Path(__file__).resolve().parent
wait = json.loads((local/'fit_wait.json').read_bytes())
assert wait['observer_closed'] and not wait['terminal']['controller_alive']
assert wait['terminal']['exitcode'] is not None
launch = json.loads((local/'fit_launch.json').read_bytes())
spec = json.loads((local/'control_spec.json').read_bytes())
target = local/'complete'
assert not target.exists()
target.mkdir()
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
probe = '''
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
print(json.dumps([str(path.relative_to(root)) for path in sorted(root.rglob('*'))
    if path.is_file() and path.suffix in ('.json','.jsonl','.log','.exit','.py','.md')]))
'''
_,stdout,stderr = client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python','-c',probe,launch['root']]),timeout=60)
raw = stdout.read()
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
sftp = client.open_sftp()
files = {}
for relative in json.loads(raw):
    remote = launch['root']+'/'+relative
    with sftp.open(remote,'rb') as stream:
        contents = stream.read()
    assert len(contents)==sftp.stat(remote).st_size
    path = target/relative
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(contents)
    files[relative] = dict(bytes=len(contents),sha256=hashlib.sha256(contents).hexdigest())
with sftp.open(launch['root']+'/fit_controller.exit','rb') as stream:
    assert int(stream.read().decode().strip())==wait['terminal']['exitcode']
sftp.close()
client.close()
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),files=files,
    remote_terminal=wait['terminal'],downloaded_weights=0,created_local_weight_archive=False,
    inference_or_optimizer_replayed=False)
(target/'INTAKE.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(text_files=len(files),bytes=sum(v['bytes'] for v in files.values()),
    actual_controller_exit=wait['terminal']['exitcode'],downloaded_weights=0)),flush=True)
