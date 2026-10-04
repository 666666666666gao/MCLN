"""Append the actual saved-parent CPU diagnostic, without touching GPU runtime."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

import paramiko


local=Path(__file__).resolve().parent
workspace=Path('C:/Users/gb')
receipt_path=local/'parent_diagnostic_publication.json'
assert not receipt_path.exists()
previous=json.loads((local/'launch_publication.json').read_bytes())
diagnostic=json.loads((local/'PARENT_MASK_BOUNDARY_RELATION.json').read_bytes())
assert diagnostic['rows']==9508 and diagnostic['parent_hits']==[5616,4506]
assert not diagnostic['inference_or_optimizer_replayed'] and not diagnostic['new_accuracy_result']
repos=[workspace/'.codex_mcln_g0_20260905',workspace/'.codex_pvground_cs_20261002',workspace/'.codex_mcln_v99_internal_20260928']
doc='docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies=[repo/doc for repo in repos]+[workspace/'Desktop/document'/Path(doc).name]
old=copies[0].read_bytes()
assert hashlib.sha256(old).hexdigest()==previous['handoff_sha256']
assert all(path.read_bytes()==old for path in copies) and b'## 20.376.59 ' not in old
for repo,head in zip(repos,previous['heads']):
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==head
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
prefix='refine-logs/pvground_final_quality_20261005/'
names=['PARENT_MASK_BOUNDARY_RELATION.json','RECORDED_PARENT_MASK_BOUNDARY_ANALYSIS.py',
       'collect_formal_authorized.py','analyze_closed_formal.py',Path(__file__).name]
payloads={prefix+name:(local/name).read_bytes() for name in names}
stamp=datetime.datetime.now().astimezone().isoformat()
section=f'''

## 20.376.59 保留4506父模型的框／Mask关系：已有逐行结果CPU补充（{stamp}）

仅读取已保存的distribution父模型9508条正式开发验证rows.jsonl，并核对其SHA256与原receipt。该模型仍为5616/4506；没有新模型前向、优化器更新或新正式成绩。实际内联CPU分析于{diagnostic['time_cst']}执行，记录的复现配方随后保存为RECORDED_PARENT_MASK_BOUNDARY_ANALYSIS.py；不把该文件声称为当时被调用的入口，也不重复执行。源码、输出及下一步闭合后收集／CPU核验工具位于{prefix}。

| 所选同一Query，阈值均为IoU>0.5 | 表达数 |
|---|---:|
| 框和Mask都合格 | 4169 |
| 框合格、Mask不合格 | 337 |
| 框不合格、Mask合格 | 964 |
| 框和Mask都不合格 | 4038 |

Mask严格命中合计5133。964条“Mask合格、最终框不合格”组中，最终框相对root GT的最大单面误差中位数0.169697903米；同Query末次几何精修最大单面位移中位数0.006441236米。方向和尺度按真实三维中心／尺寸解码的六个面计算，不能据此把全部964条算作可兑现修复量，也不能将Mask资格称为完整语义身份已确认。

分组由最终框资格定义，因此单个最终错误组的修复数为零属于分组定义。按Mask资格合并后才比较机制：Mask合格5133条中，同Query粗框至最终框严格修复16、破坏8、净+8；Mask不合格4375条中修复4、破坏1、净+3。合计20修复／9破坏／净+11，粗框4495至最终4506。该比较是共同模型内部精修，不是独立训练消融；不把方向正确比例或修正小单独写成下降原因。

这份证据更新的是“保留的最好PV父模型也存在分割支撑尚未转成合格范围”的问题定位；它不证明其他／未匹配Query的Mask资格，也不替代已有完整256候选审计。旧MCLN已执行过的软分位范围辅助及其未转成部署REC的结果仍见历史§20.133—20.140，不将同一方案再标记为未尝试。

§20.376.58的最终质量监督正式实验继续原配置，未据本统计加入新模块、调损失或扩大batch；本节不额外轮询远端GPU。闭合后收集脚本只取JSON／JSONL／log／exit，CPU分析分别核对新组与已完成4477控制、保护4506父模型以及同框语义回放。当前新实验没有终态精度，ScanRefer目标仍为同一模型5615/4754；Nr3D／Sr3D和论文新增贡献仍未证明。
'''
new=old+section.encode('utf-8')
assert new.startswith(old) and new.count(b'## 20.376.59 ')==1
client=paramiko.SSHClient(); client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
remote_project='/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(remote_project+'/'+doc,'rb') as stream:
    assert stream.read()==old
for name,raw in payloads.items():
    for repo in repos[:2]:
        (repo/name).write_bytes(raw)
    with sftp.open(remote_project+'/'+name,'wb') as stream:
        stream.write(raw)
    with sftp.open(remote_project+'/'+name,'rb') as stream:
        assert stream.read()==raw
for path in copies:
    path.write_bytes(new)
with sftp.open(remote_project+'/'+doc,'wb') as stream:
    stream.write(new)
with sftp.open(remote_project+'/'+doc,'rb') as stream:
    assert stream.read()==new
sftp.close(); client.close()
heads=[]
for index,repo in enumerate(repos):
    stage=[doc]
    if index<2:
        with (repo/'MANIFEST.md').open('a',encoding='utf-8') as stream:
            stream.write('\n- '+stamp+' Saved protected4506 parent CPU relation:4169/337/964/4038 Box-Mask quadrants; no NN replay/new accuracy. Closed-only collector and analysis prepared; active quality fit unchanged.\n')
        stage+=['MANIFEST.md',*payloads]
    changed=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git','-C',str(repo),'add','-f','--',*stage])
    subprocess.check_call(['git','-C',str(repo),'diff','--cached','--check','--',*stage])
    if index<2:
        for name,raw in payloads.items():
            assert subprocess.check_output(['git','-C',str(repo),'show',':'+name])==raw
    git_doc=subprocess.check_output(['git','-C',str(repo),'show',':'+doc])
    old_git=subprocess.check_output(['git','-C',str(repo),'show',previous['heads'][index]+':'+doc])
    assert git_doc.startswith(old_git) and git_doc.count(b'## 20.376.59 ')==1
    subprocess.check_call(['git','-C',str(repo),'commit','--quiet','-m','Record protected parent Box Mask mismatch from saved native rows'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'-c','http.version=HTTP/1.1','push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0]==heads[0]
digest=hashlib.sha256(new).hexdigest()
guard=workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py'
raw=guard.read_bytes(); assert raw.count(previous['handoff_sha256'].encode())==1
guard.write_bytes(raw.replace(previous['handoff_sha256'].encode(),digest.encode()))
assert all(path.read_bytes()==new for path in copies)
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.59',heads=heads,
    github_main=heads[0],handoff_bytes=len(new),handoff_sha256=digest,four_local_and_remote_equal=True,
    payload_count=len(payloads),new_accuracy_result=False,parent_hits=[5616,4506],
    actual_cpu_analysis=True,inference_or_optimizer_replayed=False,formal_training_configuration_changed=False,goal_achieved=False)
receipt_path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
state_path=local/'active_continuation_state.json'; state=json.loads(state_path.read_bytes())
state.update(time_cst=record['time_cst'],latest_publication=str(receipt_path),published_heads=heads,
    handoff_section='20.376.59',handoff_sha256=digest,handoff_bytes=len(new),
    current_goal_turn_classification='PROGRESS_ACTUAL_PARENT_DIAGNOSTIC_PUBLISHED')
state_path.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
with (local/'NEXT_CONTINUATION.md').open('a',encoding='utf-8') as stream:
    stream.write('\nLatest publication: parent_diagnostic_publication.json, doc59/main '+heads[0]+'. Quality fit still sole active run; observer4460 unchanged. Parent964 Mask-qualified Box-fail analysis is existing-rows CPU only. Await actual fit_wait then collect/analyze and fresh terminal audit.\n')
with (workspace/'memory/2026-10-05.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+record['time_cst']+': existing protected4506 native9508 rows CPU Box/Mask quadrants4169/337/964/4038; Mask-qualified group strict coarse->final repairs16/damages8/net8; other group4/1/net3. SelectedMask-good Box-fail964: median max face error.169698m vs last refinement.006441m, not promised fixable count. Actual inline recipe recorded, no NN replay, active quality fit unchanged. Published main '+heads[0]+'/doc59, rawfourlocal+remote exact; targetACTIVE_UNMET.\n')
print(json.dumps(record),flush=True)
