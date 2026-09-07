import datetime
import hashlib
import json
import os
from pathlib import Path
import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
archive=repo/'refine-logs/weight_cleanup_20260907_v3'
receipt=json.loads((archive/'receipt.json').read_bytes())
assert receipt['status']=='complete' and len(receipt['deleted'])==2
obs_path=repo/'refine-logs/scanrefer_native_box_transfer_posttraining_20260907_v1/observation_20260907_143011.json'
obs=json.loads(obs_path.read_bytes())
assert '58023' in obs['processes'] and '58296' in obs['processes']
line=next(line for line in reversed(obs['jobs']['training']['run.log']) if line.startswith('SCANREFER NATIVE BOX TRANSFER TRAIN '))
step=json.loads(line[len('SCANREFER NATIVE BOX TRANSFER TRAIN '):])['step']
assert step==768
master=repo/'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop=Path('C:/Users/gb/Desktop/document')/master.name
old=master.read_bytes()
assert desktop.read_bytes()==old and hashlib.sha256(old).hexdigest()=='7623cc7088fd9e7e45568df34fdc793d14b2908fb067e75995a37cf071f0bd25'
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
addition='\n\n### 20.127 清理已完成复核的正确mesh局部视觉终点（'+now+'）\n\n'
addition+='正确mesh局部视觉配对、正式评估及逐级诊断均已退出0，正式REC未晋级，相关结果见§20.102—20.103。本次只删除该配对仍保留的两份完整终点；它们不属于当前原生教师框转移的输入，当前训练/队列manifest删除前后SHA一致，未重启或修改运行。\n\n'
addition+='| 删除的失败终点 | 文件bytes | SHA-256 |\n|---|---:|---|\n'
for item in receipt['deleted']:
    addition+='| '+item['arm']+' | '+str(item['bytes'])+' | `'+item['sha256']+'` |\n'
addition+='\n删除限定于`/root/autodl-tmp/mcln_scanrefer_local_visual_mesh_pair_20260906_v1/`中的两个已核对文件；均只有一个hard link。实际释放'+str(receipt['freed_allocated_bytes'])+' allocated bytes，约1.154GiB；14:30观察可用'+str(obs['free_bytes'])+' bytes，约9.814GiB。E71、Parent、Geometry、V99、Nr平均E57及Nr续训E57六份受保护权重SHA删除前后逐项相同。训练/正式/逐级诊断的行级结果、审计、manifest和源码保留，核对11份证据SHA未变；Sr历史权重缺失与本次清理无关。\n\n'
addition+='清理后14:30:11实查原训练58023、接续58296存活；两臂已记录768/2482更新，elapsed2535.59秒，无新终态或正式结果。继续原固定实验，终态/条件正式接续已自动排队；本节是磁盘维护，不是性能增益，也不更改Scan通过后接Nr/Sr的要求。receipt SHA `'+hashlib.sha256((archive/'receipt.json').read_bytes()).hexdigest()+'`，证据目录`refine-logs/weight_cleanup_20260907_v3/`。\n'
new=old+addition.encode()
tracker=repo/'refine-logs/EXPERIMENT_TRACKER.md'
lines=tracker.read_text(encoding='utf-8').splitlines()
lines[2]='Updated: '+now+'. Section20.127: sealed correct-mesh local endpoints removed (1.154GiB); six protected weights and evidence unchanged; native box pair768/2482 alive, terminal pending.'
for i,line in enumerate(lines):
    if line.startswith('| Native teacher-box transfer |'):
        lines[i]=line.replace('at64/2482','at768/2482')
tracker_bytes=('\n'.join(lines)+'\n').encode()
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp();runtime='/home/gb/new butd/butd_detr-main/MCLN-main/'
with s.open(runtime+'docs/'+master.name,'rb') as f:
    f.prefetch(file_size=len(old));assert f.read()==old
for name,raw in [('docs/'+master.name,new),('refine-logs/EXPERIMENT_TRACKER.md',tracker_bytes)]:
    with s.open(runtime+name,'wb') as f:f.set_pipelined(True);f.write(raw)
    with s.open(runtime+name,'rb') as f:f.prefetch(file_size=len(raw));assert f.read()==raw
master.write_bytes(new);desktop.write_bytes(new);tracker.write_bytes(tracker_bytes)
attribute=repo/'.gitattributes';rule='refine-logs/weight_cleanup_20260907_v3/** -text'
assert rule not in attribute.read_text(encoding='utf-8')
with attribute.open('a',encoding='utf-8',newline='\n') as f:f.write(rule+'\n')
proof={'section':'20.127','time_cst':now,'master_sha256':hashlib.sha256(new).hexdigest(),'master_bytes':len(new),
 'three_master_copies_equal':True,'cleanup_receipt_sha256':hashlib.sha256((archive/'receipt.json').read_bytes()).hexdigest(),
 'training_step_observed':step,'observation_sha256':hashlib.sha256(obs_path.read_bytes()).hexdigest(),'new_formal_rows':0,'goal_complete':False}
for name,raw in [('handoff_sync.json',(json.dumps(proof,indent=2)+'\n').encode()),('document_from_local.py',Path(__file__).read_bytes())]:
    (archive/name).write_bytes(raw)
    with s.open('/root/autodl-tmp/mcln_weight_cleanup_20260907_v3/'+name,'wx') as f:f.write(raw)
s.close();c.close();print(json.dumps(proof),flush=True)
