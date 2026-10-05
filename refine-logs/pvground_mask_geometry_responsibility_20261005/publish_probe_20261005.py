"""Publish the actually closed fit-role diagnostic without duplicating remote logs."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import paramiko

local = Path(__file__).resolve().parent
workspace = Path('C:/Users/gb')
previous = json.loads((local.parent/'pvground_final_quality_20261005/terminal_remote_sync_publication.json').read_bytes())
summary = json.loads((local/'analysis/SUMMARY.json').read_bytes())
audit = json.loads((local/'analysis/EXPERIMENT_AUDIT.json').read_bytes())
review = json.loads((local/'analysis/TERMINAL_REVIEW_CALL.json').read_bytes())
wait = json.loads((local/'wait.json').read_bytes())
assert not (local/'publication.json').exists()
assert summary['status'] == 'CPU_RECOUNT_PASS' and not summary['accuracy_result']
assert review['result_received'] and audit['verdict'] in ('PASS','WARN')
assert not audit['blocking_findings']
assert wait['observer_closed'] and wait['exitcode'] == 0 and wait['status']['protected_parent_hashes_exact']
repos = [workspace/'.codex_mcln_g0_20260905',workspace/'.codex_pvground_cs_20261002',workspace/'.codex_mcln_v99_internal_20260928']
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies = [repo/doc for repo in repos]+[workspace/'Desktop/document'/Path(doc).name]
old = copies[0].read_bytes()
assert hashlib.sha256(old).hexdigest() == previous['handoff_sha256']
assert all(path.read_bytes() == old for path in copies)
assert b'## 20.376.62 ' not in old
for repo, head in zip(repos, previous['heads']):
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip() == head
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
names = ['EXPERIMENT_PLAN.md','GEOMETRY_TARGET_NEXT_PLAN.md','SOURCE_REVIEW.md','SOURCE_REVIEW.json','SOURCE_REVIEW_CALL.json',
    'prepare_probe.py','probe_body.py','run_mask_geometry_probe.py','spec.json','GENERATION.json',
    'controller.py','launch_probe_authorized.py','observe_probe_authorized.py','collect_probe_authorized.py',
    'analyze_probe.py','resource_check.json','launch.json','wait.json',Path(__file__).name]
for folder in ('complete','analysis','observations'):
    names.extend(str(path.relative_to(local)).replace('\\','/') for path in sorted((local/folder).rglob('*')) if path.is_file())
assert all(not name.endswith(('.pth','.pt')) and '.aris' not in name for name in names)
prefix = 'refine-logs/pvground_mask_geometry_responsibility_20261005/'
payloads = {prefix+name:(local/name).read_bytes() for name in names}
counts = summary['counts']
stamp = datetime.datetime.now().astimezone().isoformat()
section = f'''

## 20.376.62 真实fit批次的Mask合格/Box不足候选定位职责（{stamp}）

§20.376.61之后实际执行了只读训练接口诊断。当前保护的官方PV→原G→4506六面分布头重建不变，源端口安装全新零输出R后冻结全部状态、eval；并未加载已删除的负质量终点。按照seed2027取已完成fit的前8个shuffle batch，共64条、每批8条，点云和检测对象增强开启、train体素处理；row顺序与实际旧fit前64条一致。保留全部256候选。不是再训练、不是9508正式精度评估，也不是回收旧训练每一步的历史匹配。

本次在64×256＝16384个候选中，得到Mask IoU>0.5且Box IoU≤0.5的{counts['mask_only']}个候选：未匹配{counts['mask_only_unmatched']}、匹配root {counts['mask_only_matched_root']}、匹配其他实例{counts['mask_only_matched_other']}。{counts['rows_with_mask_only_unmatched']}/64条输入存在这类未匹配候选。它们的最大GT单面误差中位数为{summary['mask_only_unmatched_median_max_face_error_m']:.6f}米，当前末次精修最大单面变化中位数为{summary['mask_only_unmatched_median_max_face_change_m']:.6f}米。已选答案中Mask合格而Box不足有{counts['selected_mask_only']}条，其中未匹配{counts['selected_mask_only_unmatched']}条。这些是当前模型在固定增强训练输入上的诊断，不是总体发生率、可兑现涨点或真实语义身份的无条件证明。

另按已部署face_targets参数化在CPU核对：这{counts['mask_only_unmatched']}个未匹配候选中，{summary['mask_only_unmatched_boundary_target_outside_candidates']}个至少一面的GT目标超出当前[-4,4]偏移位置范围，共{summary['mask_only_unmatched_boundary_target_outside_faces']}个面。它们的粗框已过0.5的只有{summary['mask_only_unmatched_coarse_box_hits50']}个。超出范围指完整GT边界的精确表达受限，不代表任何可表达框都无法达到IoU>0.5；也不能自动决定扩大偏移范围。下一项几何职责实验需要记录这部分，而不是把全部候选当成同等可恢复。

实际原生criterion顺序为proposal、last、0head…4head；诊断捕获真实last Hungarian结果，并按有效GT槽映射。对detach后的最终框叶张量重放同一个原生L1/GIoU项，对detach后的边界logit叶张量重放当前distribution_loss，全部未匹配候选的两个直接输出梯度均为0。Mask合格/Box不足候选中，直接框梯度非零{counts['mask_only_box_gradient_nonzero']}、直接边界梯度非零{counts['mask_only_boundary_gradient_nonzero']}。这证明该固定面板存在已提供root分割支撑但没有直接末层定位目标的候选；不意味着共享参数、其他层或其他任务对它没有影响，也不意味着扩大定位目标一定改善精度。

64条实际有效原生GT槽均为root-only的结论为{summary['all_native_target_slot_lists_root_only']}。本次不能宣称非空多GT保护被实测验证；Nr/Sr和联合检测行职责仍需要真实对应接口。GT只用于本次训练诊断，没有新增推理GT门控、候选裁剪、质量loss、教师、第二套评分或优化器。

源代码把同一个Text Mask扩展到256个候选，再与每个候选自己的Query Mask融合。因此，融合Mask合格不等于该候选独立的实例身份已确认。本次未保存两条Mask各自的GT交并计数；下一步先区分Query自身支撑和公共Text支撑，不能直接把1112个候选全部改成root几何正例。

CPU使用保存的全部候选框/root框及Mask交并点计数重算资格、分组和梯度范围并核对通过，Box与Mask资格阈值差异均为0。Mask未独立重放原始点响应；只校验保存交并计数，GPU还逐行对已选Query做点展开一致性检查。模型所有参数梯度为空、状态逐元素不变；优化步0、新权重0。控制器实际退出0，三份必要父权重SHA保持原值。Fresh source review与实际终态integrity审查完成，后者为{audit['verdict']}，requested Astra/max、实际backend未attested、same-family/provisional；不是跨家族验收。

原始SUMMARY.json的cpu_checks=548是执行脚本的手动分组计数，不是548个不同断言。保留原始证据而不改写成功运行；实际独立审查的核对项与梯度重算以analysis/AUDIT_CPU.json和EXPERIMENT_AUDIT.md为准，数字标签区别不影响候选分组和阈值结果。

这项结果支持把下一项几何研究限定为“候选自身的训练GT确认Mask-root支撑是否需要明确边界目标”，而非继续扫R质量权重、普通语义正例或分母。先核对两路支撑职责，再设计独立、有限权重的几何责任对照，保留原Hungarian主责任与其他已匹配实例；不得据这64条直接启动三数据集或宣称严格指标上涨。模型仍保留一套native bbs、全256候选、同Query Box/Mask。最好5616/4506，目标同模型至少5615/4754，严格缺248，完整目标ACTIVE_UNMET。

本节代码与证据位于{prefix}，远端complete链接当前数据盘原始闭合目录，避免把日志再复制到空间有限的系统盘；本次启动前resource_check记录系统盘29908992字节，而非沿用上一轮50249728字节。未创建或归档权重；此前负质量权重清理结果不改。四份本地原始交接文档和远端逐字节同步后更新guard。实际新计数仅属于此诊断，历史旧CS、tail_fused和4506正式成绩不回写。
'''
new = old+section.encode('utf-8')
spec = json.loads((local/'spec.json').read_bytes())
client = paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp = client.open_sftp()
project = '/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(project+'/'+doc,'rb') as stream:
    assert stream.read() == old
evidence = project+'/'+prefix.rstrip('/')
assert not any(entry.filename==Path(prefix.rstrip('/')).name for entry in sftp.listdir_attr(project+'/refine-logs'))
sftp.mkdir(evidence)
sftp.symlink(spec['root'],evidence+'/complete')
directories = set()
for name in payloads:
    parts = Path(name).parts[:-1]
    directories.update('/'.join(parts[:index]) for index in range(1,len(parts)+1))
for relative in sorted(directories,key=lambda item:(item.count('/'),item)):
    parent,_,basename = relative.rpartition('/')
    if not any(entry.filename==basename for entry in sftp.listdir_attr(project+('/'+parent if parent else ''))):
        sftp.mkdir(project+'/'+relative)
for name, raw in payloads.items():
    for repo in repos[:2]:
        path = repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    if not name.startswith(prefix+'complete/') or name==prefix+'complete/INTAKE.json':
        with sftp.open(project+'/'+name,'wb') as stream:
            stream.write(raw)
    with sftp.open(project+'/'+name,'rb') as stream:
        assert stream.read() == raw
for path in copies:
    path.write_bytes(new)
with sftp.open(project+'/'+doc,'wb') as stream:
    stream.write(new)
with sftp.open(project+'/'+doc,'rb') as stream:
    assert stream.read() == new
sftp.close();client.close()
heads=[]
for index,repo in enumerate(repos):
    stage=[doc]
    if index<2:
        with (repo/'MANIFEST.md').open('a',encoding='utf-8') as stream:
            stream.write('\n- '+stamp+' Actual fixed64 augmentedfit geometry-role probe closed; no optimizer/weights. Direct output gradients absent for unmatched Mask-qualified Box-poor candidates. Best5616/4506 unchanged, counts qualified by fixedpanel scope.\n')
        stage+=['MANIFEST.md',*payloads]
    changed=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git','-C',str(repo),'add','-f','--',*stage])
    subprocess.check_call(['git','-C',str(repo),'diff','--cached','--check','--',*stage])
    if index<2:
        for name,raw in payloads.items():
            assert subprocess.check_output(['git','-C',str(repo),'show',':'+name])==raw
    indexed=subprocess.check_output(['git','-C',str(repo),'show',':'+doc])
    old_index=subprocess.check_output(['git','-C',str(repo),'show',previous['heads'][index]+':'+doc])
    assert indexed.startswith(old_index) and indexed.count(b'## 20.376.62 ')==1
    subprocess.check_call(['git','-C',str(repo),'commit','--quiet','-m','Record actual fit Mask Box geometry supervision responsibility'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'-c','http.version=HTTP/1.1','push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0]==heads[0]
digest=hashlib.sha256(new).hexdigest()
guard=workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py'
raw=guard.read_bytes();assert raw.count(previous['handoff_sha256'].encode())==1
guard.write_bytes(raw.replace(previous['handoff_sha256'].encode(),digest.encode()))
assert all(path.read_bytes()==new for path in copies)
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.62',heads=heads,
    github_main=heads[0],handoff_bytes=len(new),handoff_sha256=digest,four_local_and_remote_equal=True,
    remote_sync_pending=False,payload_count=len(payloads),optimizer_steps=0,accuracy_result=False,
    new_weights_created_or_archived=False,protected_best_hits=[5616,4506],full_goal_status='ACTIVE_UNMET')
(local/'publication.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
state=json.loads((local/'active_continuation_state.json').read_bytes())
state.update(time_cst=record['time_cst'],status='FIT_ROLE_PROBE_FULLY_PUBLISHED',active_reviewer=None,
    owned_gpu_job_active=False,observer_closed=True,latest_publication=str(local/'publication.json'),
    published_heads=heads,handoff_sha256=digest,handoff_bytes=len(new),handoff_section='20.376.62',
    next_action='Check separate Query/Text Mask root qualification on current fit inputs before making fusedMask-only candidates geometry positives. Shared Text Mask is expanded to all256. Then design bounded geometry-target contrast, no inference GT gate; preserve native matched-other responsibility. No new loss or training executed yet.')
(local/'active_continuation_state.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
with (workspace/'memory/2026-10-05.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+record['time_cst']+': actual fixed64 augmentedfit Mask/Box geometry-role probe closed, CPUchecked, fresh integrity '+audit['verdict']+' same-family/provisional; mask-only '+str(counts['mask_only'])+', unmatched '+str(counts['mask_only_unmatched'])+', no direct final nativeBox/edge targets for unmatched. All64 root-only. No optimizer/newweights/accuracy result; published main '+heads[0]+'/doc62 with4local+remote exact and guardupdated. Protectedbest5616/4506 remains, gap248, goalACTIVE_UNMET.\n')
print(json.dumps(record),flush=True)
