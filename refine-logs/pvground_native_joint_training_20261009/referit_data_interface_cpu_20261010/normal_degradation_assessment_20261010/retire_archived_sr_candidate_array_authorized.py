"""Retire exactly one completed, fully archived historical generated candidate array."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

root=Path(__file__).resolve().parent
receipt_path=root/'RETIRED_SR_ARRAY_ARCHIVE_COMPLETE.json'
archive=json.loads(receipt_path.read_bytes())
assert archive['status']=='ONE_RETIRED_SR_CANDIDATE_ARRAY_FULLY_ARCHIVED_SHA_VERIFIED'
target='/root/autodl-tmp/mcln_eg3dvg_sr3d_transfer_20260920_v1/formal/candidates.npy'
digest='8621cc084725688fecbe9ea8e2a9655728e7b14b922cb2c10774158d89d75937'
assert archive['identity']==dict(path=target,bytes=308574336,sha256=digest)
local=Path(archive['local'])
assert local==Path('C:/Users/gb/.codex/archives/pvg_retired_candidate_arrays_20261010/eg_sr3d/candidates.npy')
assert local.stat().st_size==308574336
h=hashlib.sha256()
with local.open('rb') as stream:
    for block in iter(lambda:stream.read(8*1024*1024),b''):
        h.update(block)
assert h.hexdigest()==digest
assert not (root/'SR_ARCHIVED_ARRAY_RETIREMENT.json').exists()
remote_code=r'''import datetime,hashlib,json,os
from pathlib import Path
p=Path('/root/autodl-tmp/mcln_eg3dvg_sr3d_transfer_20260920_v1/formal/candidates.npy')
assert p.resolve()==p and p.is_file() and p.stat().st_size==308574336
digest=hashlib.sha256()
with p.open('rb') as stream:
 for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
assert digest.hexdigest()=='8621cc084725688fecbe9ea8e2a9655728e7b14b922cb2c10774158d89d75937'
before=os.statvfs('/root/autodl-tmp');nlink=p.stat().st_nlink
p.unlink();assert not p.exists()
after=os.statvfs('/root/autodl-tmp')
print(json.dumps(dict(status='ONLY_FULLY_ARCHIVED_RETIRED_SR_CANDIDATE_ARRAY_REMOVED',
time_cst=datetime.datetime.now().astimezone().isoformat(),removed_path=str(p),bytes=308574336,
sha256=digest.hexdigest(),former_hard_links=nlink,
free_bytes_before=before.f_bavail*before.f_frsize,free_bytes_after=after.f_bavail*after.f_frsize,
deletions=1,weights_deleted=0,current_training_queries=0)))
'''
witness=json.loads((root.parent.parent/'SCP_TRANSPORT_WITNESS.json').read_bytes())
runtime='/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
environment=dict(os.environ,SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
    SSH_ASKPASS_REQUIRE='force',DISPLAY='codex-byte-transfer')
argv=['C:/Windows/System32/OpenSSH/ssh.exe','-T','-p','33476','-o','ProxyCommand=none',
    '-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
    '-o','HostKeyAlgorithms='+witness['negotiated_host_key_algorithm'],'-o','NumberOfPasswordPrompts=1',
    'root@region-9.autodl.pro',shlex.join([runtime,'-B','-u','-c',remote_code])]
response=subprocess.run(argv,env=environment,stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
(root/'SR_ARRAY_RETIREMENT_STDOUT.json').write_bytes(response.stdout)
(root/'SR_ARRAY_RETIREMENT_STDERR_PRIVATE.txt').write_bytes(response.stderr)
(root/'SR_ARRAY_RETIREMENT_EXIT.json').write_text(json.dumps(dict(exit_code=response.returncode))+'\n')
assert response.returncode==0
result=json.loads(response.stdout)
result.update(local_archive=str(local),local_archive_reverified=True,
    archive_complete_receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    local_received_cst=datetime.datetime.now().astimezone().isoformat())
(root/'SR_ARCHIVED_ARRAY_RETIREMENT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
