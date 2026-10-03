"""Verify removed files and protected best-weight dependencies without GPU work."""
import datetime
import json
import os
from pathlib import Path
import shlex
import paramiko

local=Path(__file__).parent
cleanup=json.loads((local/'authorized_negative_cleanup_receipt.json').read_bytes())
retired=json.loads((local/'retired_mcln_cleanup_receipt.json').read_bytes())
assert retired['archive_complete'] and retired['deletion_executed']
chain=json.loads(Path(r'C:\Users\gb\.codex_mcln_g0_20260905\refine-logs\scanrefer_protected_chain_20260905.json').read_bytes())
protected=[dict(label=label,path=item['path'],sha256=item['sha256'],bytes=item['bytes'])
           for label,item in chain['artifacts'].items()]
spec=json.loads((local/'g_control_spec.json').read_bytes())
protected.extend([
 dict(label='original_PV_G',path=spec['base_terminal'],sha256=spec['base_terminal_sha256'],bytes=342299695),
 dict(label='official_PV_ScanRefer',path='/root/autodl-tmp/mcln_pvground_scan_checkpoint_inspection_20260908_v1/PV-Ground_ScanRefer.pth',sha256=spec['checkpoint_sha256'],bytes=830036622)])
deleted=[item['remote'] for item in cleanup['deleted']]+[item['path'] for item in retired['files']]
assert not set(deleted).intersection(item['path'] for item in protected)
code=r'''
import hashlib,json,shutil,sys
from pathlib import Path
deleted=json.loads(sys.argv[1]);protected=json.loads(sys.argv[2]);root=Path(sys.argv[3])
assert all(not Path(p).exists() for p in deleted)
for item in protected:
    p=Path(item['path']);assert p.stat().st_size==item['bytes']
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    assert h.hexdigest()==item['sha256']
    item['actual_sha256']=h.hexdigest()
    item['hash_matches']=True
remaining=[]
for p in [Path('/root/autodl-tmp/mcln_pvground_nr_checkpoint_inspection_20260908_v1/PV-Ground_NR3D.pth'),
          Path('/root/autodl-tmp/mcln_pvground_sr_checkpoint_inspection_20260908_v1/PV-Ground_SR3D.pth'),
          Path('/root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1/official_scanrefer.pth')]:
    assert p.is_file()
    remaining.append(dict(path=str(p),bytes=p.stat().st_size))
print(json.dumps(dict(all_deleted_absent=True,deleted_count=len(deleted),protected=protected,
    other_required_parent_files_present=remaining,directory_free_bytes=shutil.disk_usage(str(root)).free,
    GPU_forwards=0,optimizer_updates=0)))
'''
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',
               password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
root='/root/autodl-tmp/pvground_candidate_consistency_20261003'
_,stdout,stderr=client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python','-c',code,
    json.dumps(deleted),json.dumps(protected),root]),timeout=180)
raw=stdout.read();assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
receipt=json.loads(raw);receipt['time_cst']=datetime.datetime.now().astimezone().isoformat()
(local/'retention_complete_verification.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
client.close()
print(json.dumps(dict(time_cst=receipt['time_cst'],deleted_count=receipt['deleted_count'],
    protected_hashes_verified=len(receipt['protected']),directory_free_bytes=receipt['directory_free_bytes'],
    GPU_forwards=0,optimizer_updates=0)))
