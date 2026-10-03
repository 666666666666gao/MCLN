"""Delete exactly the three archived, lower-scoring remote terminals authorized by the user."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

local=Path(__file__).parent
assert not (local/'authorized_negative_cleanup_receipt.json').exists()
proposal=json.loads((local/'archived_negative_cleanup_proposal.json').read_bytes())
expected={
 '/root/autodl-tmp/pvground_p3_20261002/p3/terminal.pth',
 '/root/autodl-tmp/pvground_tail_support_20261002/tail_raw/terminal.pth',
 '/root/autodl-tmp/pvground_tail_support_fused_retry_20261002/tail_fused/terminal.pth'}
assert {item['remote'] for item in proposal['files']}==expected
for item in proposal['files']:
    path=Path(item['local'])
    assert path.stat().st_size==item['bytes']
    with path.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    assert digest==item['sha256']
spec=json.loads((local/'g_control_spec.json').read_bytes())
protected=[spec['base_terminal'],
 '/root/autodl-tmp/mcln_pvground_scan_checkpoint_inspection_20260908_v1/PV-Ground_ScanRefer.pth']
remote_code=r'''
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
items=json.loads(sys.argv[1]);protected=json.loads(sys.argv[2]);root=Path(sys.argv[3])
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not (root/'pair_status.json').exists()
before=shutil.disk_usage(str(root)).free
protected_sizes={p:Path(p).stat().st_size for p in protected}
verified=[]
for item in items:
    path=Path(item['remote'])
    assert str(path.resolve())==item['remote']
    assert path.stat().st_size==item['bytes']
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            digest.update(block)
    assert digest.hexdigest()==item['sha256']
    verified.append(dict(item,remote_rehash_before_delete=True))
for item in verified:
    Path(item['remote']).unlink()
assert all(not Path(item['remote']).exists() for item in verified)
assert all(Path(p).stat().st_size==size for p,size in protected_sizes.items())
print(json.dumps(dict(deleted=verified,deleted_bytes=sum(i['bytes'] for i in verified),
    directory_free_before=before,directory_free_after=shutil.disk_usage(str(root)).free,
    protected_parent_sizes=protected_sizes,deleted_count=len(verified),deletion_executed=True)))
'''
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',
               password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
root='/root/autodl-tmp/pvground_candidate_consistency_20261003'
_,stdout,stderr=client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python','-c',remote_code,
    json.dumps(proposal['files']),json.dumps(protected),root]),timeout=180)
raw=stdout.read()
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
receipt=json.loads(raw)
receipt.update(time_cst=datetime.datetime.now().astimezone().isoformat(),
    user_authorization='及时清理磁盘空间，只保留权重最高的权重',
    local_archives_rehashed=True,V99_chain_untouched=True)
raw=(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
(local/'authorized_negative_cleanup_receipt.json').write_bytes(raw)
sftp=client.open_sftp()
with sftp.open(root+'/authorized_negative_cleanup_receipt.json','wx') as stream:
    stream.write(raw)
client.close()
print(json.dumps({key:receipt[key] for key in ('time_cst','deleted_count','deleted_bytes',
    'directory_free_before','directory_free_after','deletion_executed','V99_chain_untouched')}))
