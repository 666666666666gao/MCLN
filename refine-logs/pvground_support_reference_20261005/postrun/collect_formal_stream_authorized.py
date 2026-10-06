"""Stream compressed closed texts; preserve original bytes and exclude weights."""
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import tarfile
import paramiko

local=Path(__file__).resolve().parents[1]
wait=json.loads((local/'fit_wait.json').read_bytes())
assert wait['observer_closed'] and not wait['terminal']['controller_alive'] and wait['terminal']['exitcode']==0
launch=json.loads((local/'fit_launch.json').read_bytes())
spec=json.loads((local/'control_spec.json').read_bytes())
target=local/'complete'
assert not target.exists()
probe='''
import sys,tarfile
from pathlib import Path
root=Path(sys.argv[1])
assert (root/'fit_controller.exit').read_text().strip()=='0'
paths=[path for path in sorted(root.rglob('*')) if path.is_file() and path.suffix in
    ('.json','.jsonl','.log','.exit','.py','.md')]
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
    for path in paths:
        archive.add(str(path),arcname=str(path.relative_to(root)),recursive=False)
'''
client=paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
_,stdout,stderr=client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python','-B','-c',probe,launch['root']]),timeout=120)
compressed=stdout.read()
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
client.close()
files={}
target.mkdir()
with tarfile.open(fileobj=io.BytesIO(compressed),mode='r:gz') as archive:
    for entry in archive.getmembers():
        assert entry.isfile() and Path(entry.name).suffix in ('.json','.jsonl','.log','.exit','.py','.md')
        path=(target/entry.name).resolve()
        assert path.is_relative_to(target.resolve())
        contents=archive.extractfile(entry).read()
        assert len(contents)==entry.size
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(contents)
        files[entry.name]=dict(bytes=len(contents),sha256=hashlib.sha256(contents).hexdigest())
assert (target/'fit_controller.exit').read_text().strip()=='0'
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),files=files,
    remote_terminal=wait['terminal'],downloaded_weights=0,created_local_weight_archive=False,
    inference_or_optimizer_replayed=False,collection_transport='gzip_tar_single_SSH_stdout_stream',
    compressed_wire_bytes=len(compressed),compressed_stream_sha256=hashlib.sha256(compressed).hexdigest(),
    remote_temporary_archive_created=False,local_temporary_archive_saved=False)
(target/'INTAKE.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(text_files=len(files),bytes=sum(v['bytes'] for v in files.values()),
    compressed_wire_bytes=len(compressed),actual_controller_exit=0,downloaded_weights=0)),flush=True)
