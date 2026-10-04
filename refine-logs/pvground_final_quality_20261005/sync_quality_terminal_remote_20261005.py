"""Close the real SSH interruption without duplicating raw logs on system disk."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import paramiko


local=Path(__file__).resolve().parent
workspace=Path('C:/Users/gb')
destination=local/'terminal_remote_sync_publication.json'
assert not destination.exists()
prior=json.loads((local/'terminal_local_publication.json').read_bytes())
remote_prior=json.loads((local/'parent_diagnostic_publication.json').read_bytes())
capacity=json.loads((local/'PUBLICATION_CAPACITY.json').read_bytes())
assert capacity['destination_device']==capacity['system_device']!=capacity['data_device']
assert not capacity['complete_exists']
repos=[workspace/'.codex_mcln_g0_20260905',workspace/'.codex_pvground_cs_20261002',workspace/'.codex_mcln_v99_internal_20260928']
doc='docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies=[repo/doc for repo in repos]+[workspace/'Desktop/document'/Path(doc).name]
old=copies[0].read_bytes()
assert hashlib.sha256(old).hexdigest()==prior['handoff_sha256']
assert all(path.read_bytes()==old for path in copies) and b'## 20.376.61 ' not in old
remote_old=old[:remote_prior['handoff_bytes']]
assert hashlib.sha256(remote_old).hexdigest()==remote_prior['handoff_sha256']
for repo,head in zip(repos,prior['heads']):
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==head
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
prefix='refine-logs/pvground_final_quality_20261005/'
changed=subprocess.check_output(['git','-C',str(repos[0]),'diff','--name-only',remote_prior['heads'][0],prior['heads'][0]]).decode().splitlines()
names=[name for name in changed if name.startswith(prefix)]
assert len(names)==prior['payload_count']==55
names += [prefix+'PUBLICATION_CAPACITY.json',prefix+'terminal_local_publication.json',prefix+Path(__file__).name]
payloads={name:(local/name[len(prefix):]).read_bytes() for name in names}
assert all(not name.endswith(('.pth','.pt')) and '.aris' not in name for name in payloads)
stamp=datetime.datetime.now().astimezone().isoformat()
section=f'''

## 20.376.61 SSH恢复后的真实远端同步闭合（{stamp}）

§20.376.60的07:23本地/Git发布时点，远端同步确实尚未完成；两次SSH banner失败及控制台工具不可读取记录不修改。随后第三次原连接的只读探测实际成功，未更换服务器、地址、凭据或重启机器。PUBLICATION_CAPACITY.json核对文档发布目录device126属于系统盘，数据盘device2064独立，系统盘剩余50249728字节，complete目录尚不存在。

本次将远端refine-logs/pvground_final_quality_20261005/complete链接到/root/autodl-tmp/pvground_final_quality_20261005现存闭合结果；只补充本地收取INTAKE元数据，不重新复制约26.9MB原始日志。对55份原发布证据逐份比对字节，仅分析报告、状态及少量脚本写入系统盘。四份本地原始交接文档与远端最终内容一致后，更新原同步guard；Git原文前缀继续保持历史的换行协议。实际闭合收据为terminal_remote_sync_publication.json，不把之前失败改写为成功。

本次只同步记录，无GPU前向、优化器更新、训练重启或权重写入/归档。实际结果仍5606/4460，控制5615/4477，最好5616/4506；删除的非最佳1284112字节不恢复，原G和必要父权重保留。ScanRefer目标仍未达到，严格缺248；新Nr3D/Sr3D未训练。§20.376.60的same-family/provisional审查结论及全部证据边界不变。主线PV-Ground，全部256候选和一套原生评分保留，完整目标ACTIVE_UNMET。
'''
new=old+section.encode('utf-8')
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
project='/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(project+'/'+doc,'rb') as stream:
    assert stream.read()==remote_old
evidence=project+'/'+prefix.rstrip('/')
assert not any(entry.filename=='complete' for entry in sftp.listdir_attr(evidence))
assert not any(entry.filename=='INTAKE.json' for entry in sftp.listdir_attr('/root/autodl-tmp/pvground_final_quality_20261005'))
sftp.symlink('/root/autodl-tmp/pvground_final_quality_20261005',evidence+'/complete')
assert sftp.readlink(evidence+'/complete')=='/root/autodl-tmp/pvground_final_quality_20261005'
directories=set()
for name in payloads:
    parts=Path(name).parts[:-1]
    directories.update('/'.join(parts[:index]) for index in range(1,len(parts)+1))
for relative in sorted(directories,key=lambda item:(item.count('/'),item)):
    parent,_,basename=relative.rpartition('/')
    if not any(entry.filename==basename for entry in sftp.listdir_attr(project+('/'+parent if parent else ''))):
        sftp.mkdir(project+'/'+relative)
for name,raw in payloads.items():
    for repo in repos[:2]:
        path=repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    remote=project+'/'+name
    if not name.startswith(prefix+'complete/') or name==prefix+'complete/INTAKE.json':
        with sftp.open(remote,'wb') as stream:
            stream.write(raw)
    with sftp.open(remote,'rb') as stream:
        assert stream.read()==raw
for path in copies:
    path.write_bytes(new)
with sftp.open(project+'/'+doc,'wb') as stream:
    stream.write(new)
with sftp.open(project+'/'+doc,'rb') as stream:
    assert stream.read()==new
sftp.close();client.close()
heads=[]
for index,repo in enumerate(repos):
    stage=[doc]
    if index<2:
        with (repo/'MANIFEST.md').open('a',encoding='utf-8') as stream:
            stream.write('\n- '+stamp+' Actual SSH recovered and remote publication closed; complete evidence symlink reuses data disk originals. No GPU/optimizer/weight replay. Formal5606/4460 negative, best5616/4506 unchanged.\n')
        stage+=['MANIFEST.md',*payloads]
    changed=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git','-C',str(repo),'add','-f','--',*stage])
    subprocess.check_call(['git','-C',str(repo),'diff','--cached','--check','--',*stage])
    if index<2:
        for name,raw in payloads.items():
            assert subprocess.check_output(['git','-C',str(repo),'show',':'+name])==raw
    indexed=subprocess.check_output(['git','-C',str(repo),'show',':'+doc])
    previous_indexed=subprocess.check_output(['git','-C',str(repo),'show',prior['heads'][index]+':'+doc])
    assert indexed.startswith(previous_indexed) and indexed.count(b'## 20.376.61 ')==1
    subprocess.check_call(['git','-C',str(repo),'commit','--quiet','-m','Close actual remote terminal evidence sync after SSH interruption'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'-c','http.version=HTTP/1.1','push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0]==heads[0]
digest=hashlib.sha256(new).hexdigest()
guard=workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py'
raw=guard.read_bytes();assert raw.count(remote_prior['handoff_sha256'].encode())==1
guard.write_bytes(raw.replace(remote_prior['handoff_sha256'].encode(),digest.encode()))
assert all(path.read_bytes()==new for path in copies)
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.61',heads=heads,
    github_main=heads[0],handoff_bytes=len(new),handoff_sha256=digest,four_local_and_remote_equal=True,
    payload_count=len(payloads),remote_complete_reuses_data_disk_originals=True,remote_sync_pending=False,
    new_hits=[5606,4460],protected_best_hits=[5616,4506],inference_or_optimizer_replayed=False,
    new_weights_created_or_archived=False,goal_achieved=False)
destination.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
state_path=local/'active_continuation_state.json';state=json.loads(state_path.read_bytes())
state.update(time_cst=record['time_cst'],status='QUALITY_TERMINAL_NEGATIVE_FULLY_PUBLISHED',latest_publication=str(destination),
    published_heads=heads,handoff_section='20.376.61',handoff_sha256=digest,handoff_bytes=len(new),remote_sync_pending=False,
    remote_handoff_sha256=digest,active_reviewer=None,current_goal_turn_classification='PROGRESS_ACTUAL_TERMINAL_FULL_SYNC_CLOSED')
state_path.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
with (local/'NEXT_CONTINUATION.md').open('a',encoding='utf-8') as stream:
    stream.write('\nAuthoritative latest terminal_remote_sync_publication.json doc61/main '+heads[0]+'. SSH recovered on third read-only capacity check, no restart. Remote complete symlink reuses existing data disk results, guard updated, four rawlocal+remote docs exact. Best5616/4506, gap248, no owned GPU job; goal ACTIVE_UNMET. Prior full coupled publisher remains superseded/unexecuted.\n')
with (workspace/'memory/2026-10-05.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+record['time_cst']+': actual third SSH capacity probe recovered old connection. Published main '+heads[0]+'/doc61; exact4rawlocal+remote, guardupdated. Existing27MB results linked on data disk instead of duplicate on systemdisk. No training/model/weight replay. Finalquality5606/4460 negative, best5616/4506 retained; goalACTIVE_UNMET.\n')
print(json.dumps(record),flush=True)
