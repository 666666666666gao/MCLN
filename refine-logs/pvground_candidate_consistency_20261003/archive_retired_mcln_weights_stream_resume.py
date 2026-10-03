"""Archive and retire completed MCLN experiment weights, keeping PV/V99 parents."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import paramiko

local=Path(__file__).parent
archive=Path(r'C:\Users\gb\.codex\archives\mcln_superseded_weights_20261003')
assert archive.is_dir() and not (archive/'archive_receipt.json').exists()
inventory=json.loads((local/'retention_inventory_current.json').read_bytes())
targets={
 '/root/autodl-tmp/cs_mcln_scanrefer_best_20260923/best.pth',
 '/root/autodl-tmp/cs_mcln_scanrefer_native_best_20260926/best.pth',
 '/root/autodl-tmp/mcln_reference_memory_train_20260905_v1/results/query_global_final.pth',
 '/root/autodl-tmp/mcln_reference_memory_train_20260905_v1/results/query_pair_final.pth',
 '/root/autodl-tmp/mcln_reference_memory_train_20260905_v1/results/object_global_final.pth',
 '/root/autodl-tmp/mcln_reference_memory_train_20260905_v1/results/object_pair_final.pth',
 '/root/autodl-tmp/mcln_g0_view_pair_20260905/pair_readout_train_v1/results/global_final.pth',
 '/root/autodl-tmp/mcln_g0_view_pair_20260905/pair_readout_train_v1/results/pair_final.pth'}
items=[item for item in inventory['checkpoint_files'] if item['path'] in targets]
assert len(items)==8 and {item['path'] for item in items}==targets
assert shutil.disk_usage(archive.parent).free>sum(item['bytes'] for item in items)+256*1024**2

spec=json.loads((local/'g_control_spec.json').read_bytes())
assert spec['base_terminal'] not in targets
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',
               password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
code=r'''
import hashlib,json,sys
from pathlib import Path
result=[]
for item in json.loads(sys.argv[1]):
    p=Path(item['path']);assert str(p.resolve())==str(p)
    assert p.stat().st_size==item['bytes']
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    result.append(dict(path=str(p),bytes=item['bytes'],sha256=h.hexdigest()))
print(json.dumps(result))
'''
_,stdout,stderr=client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python','-c',code,
    json.dumps([dict(path=x['path'],bytes=x['bytes']) for x in items])]),timeout=180)
raw=stdout.read();assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
records=json.loads(raw)
sftp=client.open_sftp()
for item in records:
    target=archive/Path(item['path']).relative_to('/root/autodl-tmp')
    target.parent.mkdir(parents=True,exist_ok=True)
    copied=target.stat().st_size if target.exists() else 0
    assert copied<=item['bytes']
    if copied<item['bytes']:
        transfer="import shutil,sys; f=open(sys.argv[1],'rb'); f.seek(int(sys.argv[2])); shutil.copyfileobj(f,sys.stdout.buffer,1024*1024); f.close()"
        _,binary,transfer_error=client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python','-c',transfer,item['path'],str(copied)]),timeout=300)
        last_report=copied
        with target.open('ab') as destination:
            for block in iter(lambda:binary.read(1024*1024),b''):
                destination.write(block);copied+=len(block)
                if copied-last_report>=64*1024**2:
                    print(json.dumps(dict(copy_progress=item['path'],bytes=copied)),flush=True)
                    last_report=copied
        assert binary.channel.recv_exit_status()==0,transfer_error.read().decode()
        assert copied==item['bytes']
    assert target.stat().st_size==item['bytes']
    with target.open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==item['sha256']
    item.update(local=str(target),local_rehash_pass=True)
    print(json.dumps(dict(archived=item['path'],bytes=item['bytes'])),flush=True)
delete=r'''
import hashlib,json,shutil,sys
from pathlib import Path
items=json.loads(sys.argv[1]);root=Path(sys.argv[2])
before=shutil.disk_usage(str(root)).free
for item in items:
    p=Path(item['path']);assert str(p.resolve())==str(p)
    assert p.stat().st_size==item['bytes']
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    assert h.hexdigest()==item['sha256']
for item in items:Path(item['path']).unlink()
assert all(not Path(item['path']).exists() for item in items)
print(json.dumps(dict(deleted_count=len(items),deleted_bytes=sum(x['bytes'] for x in items),
    directory_free_before=before,directory_free_after=shutil.disk_usage(str(root)).free)))
'''
root='/root/autodl-tmp/pvground_candidate_consistency_20261003'
_,stdout,stderr=client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python','-c',delete,
    json.dumps(records),root]),timeout=180)
raw=stdout.read();assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
receipt=json.loads(raw)
receipt.update(time_cst=datetime.datetime.now().astimezone().isoformat(),files=records,
    user_authorization='及时清理磁盘空间，只保留权重最高的权重',
    PV_G_and_author_parents_untouched=True,V99_chain_untouched=True,
    deletion_executed=True,archive_complete=True)
raw=(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
(archive/'archive_receipt.json').write_bytes(raw)
(local/'retired_mcln_cleanup_receipt.json').write_bytes(raw)
with sftp.open(root+'/retired_mcln_cleanup_receipt.json','wx') as stream:stream.write(raw)
client.close()
print(json.dumps({k:v for k,v in receipt.items() if k!='files'}))
