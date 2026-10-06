"""Delete only the exact locally preserved old arrays after explicit user approval."""
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

local=Path(__file__).resolve().parent
approval=json.loads((local/'archived_arrays_delete_approval.json').read_bytes())
assert approval['approved'] and approval['scope']=='EXACT_4756_ARCHIVED_NPZ'
preview_path=local/'archived_arrays_delete_preview.json'
preview=json.loads(preview_path.read_bytes())
assert approval['preview_sha256']==hashlib.sha256(preview_path.read_bytes()).hexdigest()
assert preview['count']==4756 and preview['total_bytes']==571823151
review=json.loads((local/'ARRAY_CLEANUP_SOURCE_REVIEW.json').read_bytes())
assert review['execution_scope']=='SOURCE_ONLY' and review['verdict'] in ('PASS','WARN') and not review['blocking_findings']
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
for item in preview['files']:
    path=Path(preview['local_root'])/item['name']
    assert path.stat().st_size==item['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
assert not (local/'archived_arrays_delete_receipt.json').exists()
code='''import datetime,hashlib,json,shutil,sys
from pathlib import Path
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
preview=json.load(sys.stdin);root=Path(preview['remote_root'])
assert str(root)=='/root/autodl-tmp/pvground_mask_reference_20261006'
assert (root/'fit_controller.exit').read_text().strip()=='0'
assert json.loads((root/'fit_status.json').read_bytes())['status']=='complete'
best=root/'fused_mask_reference/initial.pth'
assert sha(best)=='2301e90a9391af4e4cd56edae09ce49db1cc8f67463e1c2568352ee2dafaff61'
allowed={a+'/'+s for a in ('native_reference','fused_mask_reference') for s in ('initial_formal','formal')}
targets=[]
for item in preview['files']:
    relative=Path(item['name']);assert relative.suffix=='.npz' and relative.parts[0]+'/'+relative.parts[1] in allowed
    path=root/relative;assert path.resolve()==path and root in path.parents
    assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],str(path)
    targets.append(path)
assert len(targets)==len(set(targets))==4756
before=shutil.disk_usage(root).free
for path in targets:path.unlink()
assert all(not path.exists() for path in targets)
assert sha(best)=='2301e90a9391af4e4cd56edae09ce49db1cc8f67463e1c2568352ee2dafaff61'
record=dict(status='EXACT_LOCAL_ARCHIVED_REMOTE_NPZ_REMOVED',time_cst=datetime.datetime.now().astimezone().isoformat(),
    count=len(targets),logical_bytes=sum(item['bytes'] for item in preview['files']),free_before=before,
    free_after=shutil.disk_usage(root).free,local_files_deleted=0,weights_deleted=0,text_records_deleted=0,
    best_weight_preserved=True,remote_root=str(root),local_root=preview['local_root'])
(root/'archived_arrays_delete_receipt.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))
'''
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
stdin,stdout,stderr=client.exec_command(shlex.join(['/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python','-B','-c',code]),timeout=180)
stdin.write(json.dumps(preview).encode());stdin.flush();stdin.channel.shutdown_write()
raw=stdout.read();assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
record=json.loads(raw);client.close()
(local/'archived_arrays_delete_receipt.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record),flush=True)
