"""Losslessly archive three exact, retired EG candidate arrays; no deletion."""
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
archive=Path('C:/Users/gb/.codex/archives/pvg_retired_candidate_arrays_20261011')
assert not archive.exists() and not (root/'RETIRED_EG_ARRAY_ARCHIVE_OWNER.json').exists()
targets=[
 dict(name='eg_nr_adapt',path='/root/autodl-tmp/mcln_eg3dvg_nr3d_adapt_20260920_v3/evaluation/formal/candidates.npy',bytes=137505920),
 dict(name='eg_nr_transfer',path='/root/autodl-tmp/mcln_eg3dvg_nr3d_transfer_20260920_v1/formal/candidates.npy',bytes=137505920),
 dict(name='eg_scan_acceptance',path='/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1/formal/candidates.npy',bytes=136306816)]
assert sum(t['bytes'] for t in targets)==411318656
archive.mkdir()
witness=json.loads((root.parent.parent/'SCP_TRANSPORT_WITNESS.json').read_bytes())
env=dict(os.environ,SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',SSH_ASKPASS_REQUIRE='force',DISPLAY='codex-byte-transfer')
runtime='/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'

def command(code):
    return ['C:/Windows/System32/OpenSSH/ssh.exe','-T','-p','33476','-o','ProxyCommand=none','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts','-o','HostKeyAlgorithms='+witness['negotiated_host_key_algorithm'],'-o','NumberOfPasswordPrompts=1','root@region-9.autodl.pro',shlex.join([runtime,'-B','-u','-c',code])]

identify=r'''import datetime,hashlib,json,sys
from pathlib import Path
targets=json.load(sys.stdin);results=[]
for t in targets:
 p=Path(t['path']);assert p.resolve()==p and not p.is_symlink() and p.stat().st_size==t['bytes']
 assert Path('/root/autodl-tmp') in p.parents and p.name=='candidates.npy'
 h=hashlib.sha256()
 with p.open('rb') as f:
  assert f.read(6)==b'\x93NUMPY';f.seek(0)
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 results.append(dict(t,sha256=h.hexdigest()))
print(json.dumps(dict(status='EXACT_THREE_RETIRED_EG_ARRAY_IDENTITIES',time_cst=datetime.datetime.now().astimezone().isoformat(),targets=results,neural_calls=0,training_status_reads=0,deletions=0)))
'''
started=time.monotonic()
owner=dict(status='RETIRED_EG_LOSSLESS_ARCHIVE_RUNNING_NO_DELETION',local_pid=os.getpid(),runtime_executable=sys.executable,
    started_cst=datetime.datetime.now().astimezone().isoformat(),archive=str(archive),targets=targets,
    total_original_bytes=411318656,conservative_uncompressed_transfer_seconds=411318656/(308574336/7603.887),
    prior_transfer_original_bytes=308574336,prior_transfer_seconds=7603.887,
    compression_ratio_unknown=True,neural_calls=0,training_status_reads=0,deletions=0,completed_targets=[])
(root/'RETIRED_EG_ARRAY_ARCHIVE_OWNER.json').write_text(json.dumps(owner,indent=2)+'\n')
print(json.dumps(owner),flush=True)
r=subprocess.run(command(identify),env=env,input=json.dumps(targets).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
(root/'RETIRED_EG_ARRAY_IDENTITY_STDOUT.json').write_bytes(r.stdout)
(root/'RETIRED_EG_ARRAY_IDENTITY_STDERR.txt').write_bytes(r.stderr)
(root/'RETIRED_EG_ARRAY_IDENTITY_EXIT.json').write_text(json.dumps(dict(exit_code=r.returncode))+'\n')
assert r.returncode==0,'Preserve failed archive identity receipt; do not retry or delete'
identity=json.loads(r.stdout)
(root/'RETIRED_EG_ARRAY_IDENTITIES.json').write_text(json.dumps(identity,indent=2)+'\n')
stream=r'''import datetime,gzip,hashlib,json,sys
from pathlib import Path
t=json.load(sys.stdin);p=Path(t['path'])
assert p.resolve()==p and not p.is_symlink() and p.stat().st_size==t['bytes']
assert Path('/root/autodl-tmp') in p.parents and p.name=='candidates.npy'
h=hashlib.sha256();count=0
with p.open('rb') as src, gzip.GzipFile(fileobj=sys.stdout.buffer,mode='wb',compresslevel=1,mtime=0) as out:
 for chunk in iter(lambda:src.read(8*1024*1024),b''):
  h.update(chunk);count+=len(chunk);out.write(chunk)
assert count==t['bytes'] and h.hexdigest()==t['sha256'] and p.stat().st_size==count
sys.stderr.write(json.dumps(dict(status='LOSSLESS_RETIRED_ARRAY_STREAM_COMPLETE',path=str(p),bytes=count,sha256=h.hexdigest(),completed_cst=datetime.datetime.now().astimezone().isoformat(),remote_temporary_files=0,deletions=0))+'\n')
'''
complete=[]
for target in identity['targets']:
    folder=archive/target['name'];folder.mkdir()
    packed_partial=folder/'candidates.npy.partial.gz'
    packed=folder/'candidates.npy.gz'
    original_partial=folder/'candidates.npy.partial'
    original=folder/'candidates.npy'
    assert all(archive.resolve() in p.resolve().parents for p in [packed_partial,packed,original_partial,original])
    with packed_partial.open('xb') as out,(folder/'STREAM_STDERR_PRIVATE.txt').open('xb') as err:
        process=subprocess.Popen(command(stream),env=env,stdin=subprocess.PIPE,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        owner.update(current_target=target['name'],native_ssh_process_pid=process.pid,current_stream_started_cst=datetime.datetime.now().astimezone().isoformat())
        (root/'RETIRED_EG_ARRAY_ARCHIVE_OWNER.json').write_text(json.dumps(owner,indent=2)+'\n')
        process.stdin.write(json.dumps(target).encode());process.stdin.close()
        code=process.wait()
    (folder/'STREAM_EXIT.json').write_text(json.dumps(dict(exit_code=code))+'\n')
    assert code==0,'Preserve this exact partial and transport receipt; no retry/deletion'
    receipt=json.loads((folder/'STREAM_STDERR_PRIVATE.txt').read_bytes())
    assert receipt['path']==target['path'] and receipt['sha256']==target['sha256'] and receipt['bytes']==target['bytes']
    h=hashlib.sha256();size=0
    with gzip.open(str(packed_partial),'rb') as src,original_partial.open('xb') as out:
        for chunk in iter(lambda:src.read(8*1024*1024),b''):
            h.update(chunk);size+=len(chunk);out.write(chunk)
    assert size==target['bytes'] and h.hexdigest()==target['sha256']
    with original_partial.open('rb') as src:assert src.read(6)==b'\x93NUMPY'
    assert not packed.exists() and not original.exists()
    packed_partial.replace(packed);original_partial.replace(original)
    row=dict(target,local_path=str(original),compressed_path=str(packed),compressed_bytes=packed.stat().st_size,
        full_original_bytes_verified=size,full_original_sha256_verified=h.hexdigest(),completed_cst=datetime.datetime.now().astimezone().isoformat())
    complete.append(row);owner.update(completed_targets=complete,current_target=None,native_ssh_process_pid=None)
    (root/'RETIRED_EG_ARRAY_ARCHIVE_OWNER.json').write_text(json.dumps(owner,indent=2)+'\n')
    print(json.dumps(dict(status='ONE_RETIRED_ARRAY_FULLY_ARCHIVED',result=row)),flush=True)
result=dict(status='EXACT_THREE_RETIRED_EG_ARRAYS_FULLY_ARCHIVED_NOT_DELETED',completed_cst=datetime.datetime.now().astimezone().isoformat(),
    elapsed_seconds=time.monotonic()-started,targets=complete,total_original_bytes=sum(t['bytes'] for t in complete),
    total_compressed_bytes=sum(t['compressed_bytes'] for t in complete),neural_calls=0,training_status_reads=0,
    remote_temporary_files=0,deletions=0,model_weights_archived=0)
(root/'RETIRED_EG_ARRAY_ARCHIVE_COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n')
owner.update(status=result['status'],completed_cst=result['completed_cst'],elapsed_seconds=result['elapsed_seconds'])
(root/'RETIRED_EG_ARRAY_ARCHIVE_OWNER.json').write_text(json.dumps(owner,indent=2)+'\n')
print(json.dumps(result),flush=True)
