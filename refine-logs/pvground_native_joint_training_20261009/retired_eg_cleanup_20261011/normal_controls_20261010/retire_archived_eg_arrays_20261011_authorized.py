"""Delete only three closed EG arrays after full local and remote SHA checks."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

root=Path(__file__).resolve().parent
assert not (root/'RETIRED_EG_ARRAY_RETIREMENT.json').exists()
complete=json.loads((root/'RETIRED_EG_ARRAY_ARCHIVE_COMPLETE.json').read_bytes())
assert complete['status']=='EXACT_THREE_RETIRED_EG_ARRAYS_FULLY_ARCHIVED_NOT_DELETED'
assert complete['total_original_bytes']==411318656 and len(complete['targets'])==3
allowed={
 '/root/autodl-tmp/mcln_eg3dvg_nr3d_adapt_20260920_v3/evaluation/formal/candidates.npy':137505920,
 '/root/autodl-tmp/mcln_eg3dvg_nr3d_transfer_20260920_v1/formal/candidates.npy':137505920,
 '/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1/formal/candidates.npy':136306816}
assert {r['path']:r['bytes'] for r in complete['targets']}==allowed
archive=Path('C:/Users/gb/.codex/archives/pvg_retired_candidate_arrays_20261011').resolve()
verified=[]
for row in complete['targets']:
    path=Path(row['local_path'])
    assert archive in path.resolve().parents and path.stat().st_size==row['bytes']
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    assert h.hexdigest()==row['sha256']==row['full_original_sha256_verified']
    verified.append(dict(path=row['path'],local_path=str(path),bytes=row['bytes'],sha256=h.hexdigest()))
local=dict(status='THREE_FULL_LOCAL_ORIGINAL_FILES_REHASHED',verified_cst=datetime.datetime.now().astimezone().isoformat(),targets=verified,total_bytes=411318656)
(root/'RETIRED_EG_LOCAL_SHA_RECHECK.json').write_text(json.dumps(local,indent=2)+'\n')
code=r'''import datetime,hashlib,json,shutil,sys
from pathlib import Path
b=json.load(sys.stdin)
allowed={
 '/root/autodl-tmp/mcln_eg3dvg_nr3d_adapt_20260920_v3/evaluation/formal/candidates.npy':137505920,
 '/root/autodl-tmp/mcln_eg3dvg_nr3d_transfer_20260920_v1/formal/candidates.npy':137505920,
 '/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1/formal/candidates.npy':136306816}
assert b['status']=='THREE_FULL_LOCAL_ORIGINAL_FILES_REHASHED'
assert {r['path']:r['bytes'] for r in b['targets']}==allowed and len(b['targets'])==3
checked=[]
for r in b['targets']:
 p=Path(r['path']);assert p.resolve()==p and not p.is_symlink() and p.name=='candidates.npy'
 assert Path('/root/autodl-tmp') in p.parents and p.stat().st_size==r['bytes']
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 assert h.hexdigest()==r['sha256']
 checked.append(dict(r,remote_sha256=h.hexdigest(),link_count=p.stat().st_nlink))
before={p:dict(zip(['total','used','free'],shutil.disk_usage(p))) for p in ['/','/root/autodl-tmp']}
for r in checked:
 p=Path(r['path']);p.unlink();assert not p.exists()
after={p:dict(zip(['total','used','free'],shutil.disk_usage(p))) for p in ['/','/root/autodl-tmp']}
print(json.dumps(dict(status='ONLY_THREE_FULLY_ARCHIVED_RETIRED_EG_ARRAYS_DELETED',completed_cst=datetime.datetime.now().astimezone().isoformat(),
 removed=checked,logical_bytes_removed=sum(r['bytes'] for r in checked),storage_before=before,storage_after=after,
 data_free_increase_bytes=after['/root/autodl-tmp']['free']-before['/root/autodl-tmp']['free'],
 neural_calls=0,training_status_reads=0,model_weights_deleted=0,model_source_changes=0,training_restart=False)))
'''
witness=json.loads((root.parent.parent/'SCP_TRANSPORT_WITNESS.json').read_bytes())
env=dict(os.environ,SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',SSH_ASKPASS_REQUIRE='force',DISPLAY='codex-byte-transfer')
runtime='/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
argv=['C:/Windows/System32/OpenSSH/ssh.exe','-T','-p','33476','-o','ProxyCommand=none','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts','-o','HostKeyAlgorithms='+witness['negotiated_host_key_algorithm'],'-o','NumberOfPasswordPrompts=1','root@region-9.autodl.pro',shlex.join([runtime,'-B','-u','-c',code])]
r=subprocess.run(argv,env=env,input=json.dumps(local).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
(root/'RETIRED_EG_RETIREMENT_STDOUT.json').write_bytes(r.stdout)
(root/'RETIRED_EG_RETIREMENT_STDERR_PRIVATE.txt').write_bytes(r.stderr)
(root/'RETIRED_EG_RETIREMENT_EXIT.json').write_text(json.dumps(dict(exit_code=r.returncode))+'\n')
assert r.returncode==0,'Preserve failed exact retirement receipt; do not repeat deletion or restart training'
value=json.loads(r.stdout)
assert value['logical_bytes_removed']==411318656 and value['model_weights_deleted']==0
value.update(local_verification=local,archive_complete_receipt=str(root/'RETIRED_EG_ARRAY_ARCHIVE_COMPLETE.json'),original_archive_native_session=84511,original_archive_native_exit_code=0)
(root/'RETIRED_EG_ARRAY_RETIREMENT.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value),flush=True)
