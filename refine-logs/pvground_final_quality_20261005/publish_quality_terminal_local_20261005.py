"""Publish actual terminal evidence locally and to Git, with remote sync pending."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess



local = Path(__file__).resolve().parent
workspace = Path('C:/Users/gb')
receipt_path = local/'terminal_local_publication.json'
assert not receipt_path.exists()
previous = json.loads((local/'parent_diagnostic_publication.json').read_bytes())
summary = json.loads((local/'analysis/SUMMARY.json').read_bytes())
audit = json.loads((local/'analysis/EXPERIMENT_AUDIT.json').read_bytes())
review = json.loads((local/'analysis/TERMINAL_REVIEW_CALL.json').read_bytes())
resources = json.loads((local/'CLOSED_RESOURCES.json').read_bytes())
wait = json.loads((local/'fit_wait.json').read_bytes())
assert review['result_received'] and wait['observer_closed'] and wait['terminal']['exitcode']==0
assert not summary['target_pass'] and summary['training_order_exact']
assert summary['table'][2]['rec_hits25']==5606 and summary['table'][2]['rec_hits50']==4460
assert summary['retained_best']['hits']==[5616,4506]
assert not resources['gpu_compute_processes']
repos = [workspace/'.codex_mcln_g0_20260905',workspace/'.codex_pvground_cs_20261002',workspace/'.codex_mcln_v99_internal_20260928']
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies = [repo/doc for repo in repos]+[workspace/'Desktop/document'/Path(doc).name]
old = copies[0].read_bytes()
assert hashlib.sha256(old).hexdigest()==previous['handoff_sha256']
assert all(path.read_bytes()==old for path in copies) and b'## 20.376.60 ' not in old
for repo,head in zip(repos,previous['heads']):
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==head
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
prefix = 'refine-logs/pvground_final_quality_20261005/'
names = ['fit_wait.json','CLOSED_RESOURCES.json','check_closed_resources.py',
    'PUBLICATION_CAPACITY_ATTEMPT_1.json','PUBLICATION_CAPACITY_ATTEMPT_2.json','CONSOLE_STATE_UNAVAILABLE.json',
    'check_publication_capacity.py','LEGACY_MASK_QUALIFIED_MATCH_ROLE.json','recount_legacy_mask_match_roles.py',
    'prepare_local_terminal_publication.py','publish_quality_terminal_20261005.py',Path(__file__).name]
for directory in ('complete','analysis'):
    names.extend(str(path.relative_to(local)).replace('\\','/') for path in sorted((local/directory).rglob('*')) if path.is_file())
assert all(not name.endswith(('.pth','.pt')) and '.aris' not in name for name in names)
payloads = {prefix+name:(local/name).read_bytes() for name in names}
stamp = datetime.datetime.now().astimezone().isoformat()
verdict = str(audit['verdict']).upper()
retention = wait['terminal']['status']['weight_retention']
removed = sum(item['bytes'] for item in retention['deleted'])
effect = summary['quality_vs_control']['formal']['0.5']
direct = summary['stages']['final_quality']['formal']['fixed_frame_readback_effect']['0.5']
mask = summary['stages']['final_quality']['formal']
section = f'''

## 20.376.60 最终框质量监督已完成：未形成增量，非最佳权重已清理（{stamp}）

§20.376.58启动的唯一新增quality组已真实完成29778条fit各一次、3723次AdamW更新（3722个batch8及最后batch2，有效batch8，累积1，seed2027），6887模块留出及9508正式开发验证均闭合。fit完成于2026-10-05 06:34:08，正式验证完成于06:57:51，控制器及权重处置完成于{summary['actual_finished_cst']}，真实退出码0。复用§20.376.57已完成evidence_visible/native+G组作直接控制，不重新训练控制，不恢复已删除的负结果终点。仅R的96672个参数更新，父模型全程冻结并eval，原G、CE、对比监督与分母不变。

| 同一模型、完整9508条、原生last/bbs | Acc@0.25〔命中〕 | Acc@0.50〔命中〕 |
|---|---:|---:|
| 保留的六面分布几何父模型 | 59.0660%〔5616〕 | 47.3917%〔4506〕 |
| 已完成native+G回读控制 | 59.0555%〔5615〕 | 47.0867%〔4477〕 |
| 新增最终框质量目标 | 58.9609%〔5606〕 | 46.9079%〔4460〕 |

质量项使用真实有符号native bbs与停止梯度的最终框root IoU，在原G可信几何资格集合内匹配中心化的候选差异，预定权重1.0。不是独立质量头或部署第二套排名；GT资格仅训练使用，高几何重叠也不声称证明完整物理实例身份。全部256候选保留，最终仅原生分数选择一次，同Query输出Box及Mask。

质量组相对本轮控制严格修复{effect['repairs']}、破坏{effect['damages']}、净{effect['net']}，507条表达更换了Query；相对保留父模型严格净−46。同一质量检查点、同一批框/Mask，回读前语义输入与真实R输出比较严格修复{direct['repairs']}、破坏{direct['damages']}、净{direct['net']}，1849条表达换Query。这个同框诊断不是独立训练消融，但它说明这轮直接判断没有净改善。父模型状态冻结且逐项相同，所以这轮下降不能归因为上游、Mask或几何被共同微调破坏。不能由此否定所有质量学习，也不继续仅扩R容量、换分母或扫描质量权重。

两组严格Full256几何上界均为7890，质量组5048条严格错误中3430条仍有合格框、1618条无合格框；GT上界仅离线诊断，不能当可部署收益。质量组Mask命中为5815/5138，mIoU {mask['mask_miou']:.6f}%。按GT体积四分位，质量组相对控制严格净变化−2/−5/−8/−2，均为离线描述而非因果归因。训练质量loss前100步均值0.007576918、末100步0.006647891，损失下降没有转化为正式定位增量，且这些窗口输入不同。

只读收取32份JSON/JSONL/log/exit共26917971字节，权重下载及新负结果归档均为0；CPU从真实逐行中心/尺寸和root GT重算选中、粗框与绕过R的IoU，阈值差异为0，训练样本顺序与3723步控制完整一致。初始6887两项命中一致，但独立完整前向的Mask平均值/个别排序不逐位相同，不能称为逐位完全配对。Mask结果复核使用已保存Mask IoU，没有伪称重放原始Mask张量。

审查另核实本轮3723条训练记录中的原生matched_queries均等于batch行数，真实加载器此配置每个表达只暴露一个原生GT目标。新质量池始终等于原G重新指派候选加每行一个root；1553个训练表达仅有单候选池，中心化质量项自然为零。源码有排除其他GT匹配的规则，但本轮没有实际多GT案例，不能将其写成实证验证了多实例匹配保护。联合检测及Nr/Sr的对应责任仍需各自真实接口验证，不将旧验证root-only统计移作训练比例。

experiment-audit已收到真实fresh Codex审查结果，结论{verdict}，请求gpt-6-astra/max，实际后端未获工具证明，same-family/provisional。报告与CPU证据见{prefix}analysis/EXPERIMENT_AUDIT.md、EXPERIMENT_AUDIT.json、AUDIT_CPU_CHECK.json及输入SHA表；私有实际调用痕迹不公开。SUMMARY.json及RESULTS.md里的review_pending记录属于其生成时点，实际审查终态以TERMINAL_REVIEW_CALL.json和审查报告为准。原始数据文件未作审查重放、单seed、预训练见过的6887与长期开发使用的9508、GT几何资格代理及跨进程数值差异均保留边界；不宣称Nr3D/Sr3D有效或论文novelty已成立。

真实处置记录删除唯一非最佳新权重/root/autodl-tmp/pvground_final_quality_20261005/quality/terminal.pth，释放{removed}字节；删除前已完成实际保存恢复与正式CPU核对，无本地负权重归档。随后远端只读核验：该文件不存在，4506最佳delta、原G和官方PV三份父权重SHA256与保护记录一致；GPU空闲。数据盘余量{resources['data_free_bytes']}字节，系统盘余量{resources['system_free_bytes']}字节，所有新大文件继续放数据盘。原V99历史链未修改。

本节先完成四份本地原始交接文档与GitHub发布，远端同步尚未执行。两次SSH在banner握手阶段退出，均未执行远端容量探测或写入；控制台浏览器也因app-server缺失而无法读取，不能据此猜测服务器已关机。失败见PUBLICATION_CAPACITY_ATTEMPT_1/2.json与CONSOLE_STATE_UNAVAILABLE.json，容量结果文件未生成。远端文档仍应为§20.376.59，同步guard保持该SHA不变，不把本地发布写成远端已更新。SSH恢复后将核对实际挂载位置，让远端complete证据引用数据盘原始结果，避免再次将约26.9MB日志复制到已仅剩约48MiB的系统盘。

当前最好仍为5616/4506，ScanRefer目标同一模型至少5615/4754，严格仍差248。当前质量回读实验判为规定预算下负结果，不将冻结父模型实验又写成共同续训退化。后续回到§20.376.59已实测的964条Mask合格/Box不合格缺口，先核对这些候选是否获得原生定位训练责任，再决定是否有必要改变几何学习；不直接把964当可修复数，也不重复已有局部Mask拼接、语义正例归一化或MCLN软CDF负实验。主线仍PV-Ground，一套原生输出，ScanRefer结构及目标优先，Nr/Sr未开始新结构正式训练。完整目标ACTIVE_UNMET。

额外仅对旧失败tail_fused保存的完整9508条候选诊断rows作CPU计数，SHA与原receipt一致：其1008条已选Mask>0.5、Box≤0.5样本中，117条匹配root、891条未匹配。注意该诊断的原生GT槽全部仅[0]，是对验证输入运行Hungarian的root-only结果；不是当前4506父模型的训练匹配，也没有验证多实例匹配保护。它支持下一步先检查真实训练批次定位责任，不证明该责任差异已经造成当前失败，不把旧模型比例移作新模型发生率；没有GPU前向、优化器更新、新监督或推理GT门控。
'''
new = old+section.encode('utf-8')
assert new.startswith(old) and new.count(b'## 20.376.60 ')==1
for name,raw in payloads.items():
    for repo in repos[:2]:
        destination=repo/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(raw)
for path in copies:
    path.write_bytes(new)
# No SSH writes or remote document guard change in this local publication.
heads=[]
for index,repo in enumerate(repos):
    stage=[doc]
    if index<2:
        with (repo/'MANIFEST.md').open('a',encoding='utf-8') as stream:
            stream.write('\n- '+stamp+' Actual final-quality negative5606/4460 vs native+G control5615/4477; protected best5616/4506 unchanged. CPU rows/order verified, actual fresh terminal review '+verdict+' same-family/provisional. Nonbest1284112B removed, no archive.\n')
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
    assert git_doc.startswith(old_git) and git_doc.count(b'## 20.376.60 ')==1
    subprocess.check_call(['git','-C',str(repo),'commit','--quiet','-m','Record closed native final quality loss negative result and weight cleanup'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'-c','http.version=HTTP/1.1','push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0]==heads[0]
digest=hashlib.sha256(new).hexdigest()
guard=workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py'
assert guard.read_bytes().count(previous['handoff_sha256'].encode())==1
assert all(path.read_bytes()==new for path in copies)
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.60',heads=heads,
    github_main=heads[0],handoff_bytes=len(new),handoff_sha256=digest,four_local_equal=True,remote_sync_pending=True,
    payload_count=len(payloads),new_accuracy_result=True,new_hits=[5606,4460],protected_best_hits=[5616,4506],
    source_control_hits=[5615,4477],deleted_nonbest_bytes=removed,negative_weight_archived=False,
    integrity_verdict=verdict,review_independence='same-family',acceptance_status='provisional',goal_achieved=False)
record['remote_complete_reuses_data_disk_originals']=False
record['remote_handoff_expected_sha256']=previous['handoff_sha256']
record['remote_guard_unchanged']=True
receipt_path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
state_path=local/'active_continuation_state.json'; state=json.loads(state_path.read_bytes())
state.update(time_cst=record['time_cst'],status='QUALITY_TERMINAL_LOCAL_GITHUB_PUBLISHED_REMOTE_PENDING',latest_publication=str(receipt_path),
    published_heads=heads,handoff_section='20.376.60',handoff_sha256=digest,handoff_bytes=len(new),
    current_goal_turn_classification='PROGRESS_ACTUAL_QUALITY_TERMINAL_LOCAL_PUBLISHED_REMOTE_PENDING',collector_session_id=None,
    closed_result_tools_executed=True,formal_observer_closed=True,sole_formal_observer_session_id=None)
assert 84556 in state['closed_native_sessions']
state['remote_sync_pending']=True
state['active_reviewer']=None
state['remote_handoff_sha256']=previous['handoff_sha256']
state_path.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
with (local/'NEXT_CONTINUATION.md').open('a',encoding='utf-8') as stream:
    stream.write('\nLatest actual local/Git publication terminal_local_publication.json; remote sync pending doc60/main '+heads[0]+'. Quality closed5606/4460; reusedcontrol5615/4477; protectedbest5616/4506. Review '+verdict+' same-family/provisional. Nonbest1284112B deleted; no GPU job or observer remains. Fullgoal ACTIVE_UNMET; no Nr/Sr or target claim.\n')
with (workspace/'memory/2026-10-05.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+record['time_cst']+': final-quality formal closed5606/4460 (58.9609/46.9079), vs reusednative+G control5615/4477 strict14repairs31damages/net-17; sameframeR72/118/net-46. Frozenparentexact, Full2567890. Actual fresh integrity '+verdict+' same-family/provisional; no weights/NN replay in intake. Ownnonbest1284112B deleted, best5616/4506 and G/PV parents SHA verified. Published main '+heads[0]+'/doc60, rawfourlocal exact, remote sync pending and guard unchanged. GoalACTIVE_UNMET strictgap248; no current GPU job.\n')
print(json.dumps(record),flush=True)
