"""Archive one closed EG-Sr3D candidate cache; deletion is a separate verified action."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

root = Path(__file__).resolve().parent
assert not (root/'RETIRED_SR_ARRAY_ARCHIVE_COMPLETE.json').exists()
remote_path='/root/autodl-tmp/mcln_eg3dvg_sr3d_transfer_20260920_v1/formal/candidates.npy'
expected_bytes=308574336
inventory=json.loads((root/'LARGE_ARTIFACT_INVENTORY.json').read_bytes())
assert dict(path=remote_path,bytes=expected_bytes) in inventory['files']
target=Path('C:/Users/gb/.codex/archives/pvg_retired_candidate_arrays_20261010/eg_sr3d/candidates.npy')
partial=target.with_name(target.name+'.native_scp_partial')
assert not target.exists() and not partial.exists()
target.parent.mkdir(parents=True,exist_ok=True)
witness=json.loads((root.parent.parent/'SCP_TRANSPORT_WITNESS.json').read_bytes())
environment=dict(os.environ,
    SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
    SSH_ASKPASS_REQUIRE='force',DISPLAY='codex-byte-transfer')
transport=['-o','ProxyCommand=none','-o','StrictHostKeyChecking=yes',
    '-o','UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
    '-o','HostKeyAlgorithms='+witness['negotiated_host_key_algorithm'],'-o','NumberOfPasswordPrompts=1']
code=r'''import hashlib,json
from pathlib import Path
p=Path('/root/autodl-tmp/mcln_eg3dvg_sr3d_transfer_20260920_v1/formal/candidates.npy')
assert p.resolve()==p and p.is_file() and not p.is_symlink() and p.stat().st_size==308574336
h=hashlib.sha256()
with p.open('rb') as f:
 for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
print(json.dumps(dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())))
'''
runtime='/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
command=['C:/Windows/System32/OpenSSH/ssh.exe','-T','-p','33476']+transport+[
    'root@region-9.autodl.pro',shlex.join([runtime,'-B','-u','-c',code])]
response=subprocess.run(command,env=environment,stdin=subprocess.DEVNULL,
    stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
(root/'SR_ARRAY_IDENTITY_STDOUT.json').write_bytes(response.stdout)
(root/'SR_ARRAY_IDENTITY_STDERR.txt').write_bytes(response.stderr)
(root/'SR_ARRAY_IDENTITY_EXIT.json').write_text(json.dumps(dict(exit_code=response.returncode))+'\n',encoding='utf-8')
assert response.returncode==0,'Read failure; no archive restart or deletion'
identity=json.loads(response.stdout)
assert identity['path']==remote_path and identity['bytes']==expected_bytes
owner=dict(status='ONE_RETIRED_SR_CANDIDATE_ARRAY_ARCHIVE_RUNNING',
    local_pid=os.getpid(),started_cst=datetime.datetime.now().astimezone().isoformat(),
    identity=identity,local=str(target),estimated_seconds=expected_bytes*witness['seconds']/witness['bytes'],
    role='Closed EG-Sr3D prediction cache; never a PV/G/V99/author checkpoint or dataset',deletions=0)
(root/'RETIRED_SR_ARRAY_ARCHIVE_OWNER.json').write_text(json.dumps(owner,indent=2)+'\n',encoding='utf-8')
print(json.dumps(owner),flush=True)
started=time.monotonic()
command=['C:/Windows/System32/OpenSSH/scp.exe','-O','-P','33476']+transport+[
    'root@region-9.autodl.pro:'+remote_path,partial.as_posix()]
with (root/'SR_ARRAY_SCP_STDOUT.txt').open('wb') as stdout,(root/'SR_ARRAY_SCP_STDERR.txt').open('wb') as stderr:
    response=subprocess.run(command,env=environment,stdin=subprocess.DEVNULL,
        stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
(root/'SR_ARRAY_SCP_EXIT.json').write_text(json.dumps(dict(exit_code=response.returncode))+'\n',encoding='utf-8')
assert response.returncode==0 and partial.stat().st_size==expected_bytes
digest=hashlib.sha256()
with partial.open('rb') as stream:
    for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
assert digest.hexdigest()==identity['sha256']
partial.replace(target)
receipt=dict(status='ONE_RETIRED_SR_CANDIDATE_ARRAY_FULLY_ARCHIVED_SHA_VERIFIED',
    identity=identity,local=str(target),time_cst=datetime.datetime.now().astimezone().isoformat(),
    elapsed_seconds=time.monotonic()-started,deletions=0)
(root/'RETIRED_SR_ARRAY_ARCHIVE_COMPLETE.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt),flush=True)
