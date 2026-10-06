"""Publish actual closed Mask-reference evidence after audit and best-only retention."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

import paramiko


root = Path(__file__).resolve().parents[1]
receipt_path = root/'terminal_publication.json'
assert not receipt_path.exists()
review = json.loads((root/'postrun/PUBLISH_TERMINAL_SOURCE_REVIEW.json').read_bytes())
assert review['execution_scope']=='SOURCE_ONLY'
assert review['verdict'] in ('PASS','WARN') and not review['blocking_findings']
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
acceptance = json.loads((root/'analysis/TERMINAL_ACCEPTANCE.json').read_bytes())
assert acceptance['status']=='ACTUAL_TERMINAL_EVIDENCE_ACCEPTED_WITH_QUALIFICATIONS'
assert acceptance['audit_result_received'] and acceptance['selected_restore_complete']
audit_path = root/'analysis/EXPERIMENT_AUDIT.json'
audit = json.loads(audit_path.read_bytes())
assert audit['verdict'] in ('PASS','WARN') and not audit['blocking_findings']
assert audit['execution_scope']=='TERMINAL_SOURCE_AND_ACTUAL_ARTIFACTS'
assert acceptance['audit_sha256']==hashlib.sha256(audit_path.read_bytes()).hexdigest()
summary_path = root/'analysis/SUMMARY.json'
summary = json.loads(summary_path.read_bytes())
assert acceptance['summary_sha256']==hashlib.sha256(summary_path.read_bytes()).hexdigest()
assert [summary['metric_best_candidate'][key] for key in ('rec_hits25','rec_hits50')]==[5598,4848]
assert summary['metric_best_candidate']['optimizer_updates']==0
retention = json.loads((root/'weight_retention.json').read_bytes())
assert retention['status']=='CLOSED_NONBEST_WEIGHTS_REMOVED'
assert retention['retained_best']['candidate']=='fused_mask_reference/initial_formal'
assert retention['weights_after']==1 and len(retention['deleted'])==4
assert retention['fresh_audit_sha256']==hashlib.sha256(audit_path.read_bytes()).hexdigest()
previous = json.loads((root/'retention_tools_publication.json').read_bytes())
assert previous['section']=='20.376.85'
state_path = root.parent/'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state = json.loads(state_path.read_bytes())
assert Path(state['latest_publication']).resolve()==(root/'retention_tools_publication.json').resolve()
assert not state['owned_gpu_job_active']
workspace = Path('C:/Users/gb')
repos = [workspace/'.codex_mcln_g0_20260905',workspace/'.codex_pvground_cs_20261002',
         workspace/'.codex_mcln_v99_internal_20260928']
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies = [repo/doc for repo in repos]+[workspace/'Desktop/document'/Path(doc).name]
old = copies[0].read_bytes()
assert hashlib.sha256(old).hexdigest()==previous['handoff_sha256']
assert all(path.read_bytes()==old for path in copies)
for repo,head in zip(repos,previous['heads']):
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==head
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
stamp = datetime.datetime.now().astimezone().isoformat()
prefix = 'refine-logs/pvground_mask_reference_20261006/terminal_evidence/'
section = f'''

## 20.376.86 Mask参考范围真实对照闭合：最高结果来自零更新版本（{stamp}）

本轮实际训练于2026-10-06 18:07:26.410720 UTC+8结束，controller exit0；既有观察器18:10:23闭合。两组各29778条fit输入一次、3723次更新，seed2027、有效batch8，最后batch2。4819份实际文件647499194字节收集并校验，未收集权重或重新运行推理。下表每行均为完整9508条ScanRefer开发验证，原生last/bbs及同一Query的Box/Mask。

| 版本 | 新优化器更新数 | Acc@0.25〔命中〕 | Acc@0.5〔命中〕 |
|---|---:|---:|---:|
| 原保留几何父模型 | 3723，几何累计11169 | 59.0660%〔5616〕 | 47.4443%〔4511〕 |
| native参考，初始 | 0 | 59.0555%〔5615〕 | 47.2760%〔4495〕 |
| native参考，训练终点 | 3723 | 59.0766%〔5617〕 | 47.4337%〔4510〕 |
| **融合Mask参考，初始** | **0** | **58.8767%〔5598〕** | **50.9886%〔4848〕** |
| 融合Mask参考，训练终点 | 3723 | 58.8241%〔5593〕 | 50.8204%〔4832〕 |

按事先规定的Acc@0.5优先规则，最佳为fused_mask_reference/initial_formal：相对原4511父模型−18/+337命中，相对历史V99同一行5572/4797为+26/+51。两项达到58.3%/50.0%的开发线，但最高结果来自更换空间参考后的零更新前向；初始最终框等于融合Mask参考范围。不能把它写成新增训练的收益、分布头已学得的337条增量或完整V99知识蒸馏。融合组继续训练后相对自身初始−5/−16；严格修复13、破坏29。

该实现对全部256候选，以预测融合Mask与真实超点成员范围构造一个空间参考，供局部读取和六面分布精修使用；无效支撑保留原粗框作为参考。仍使用一套原生bbs，没有GT推理资格、双源排名、七版本选择器或教师。共同重置output.weight/output.bias两项，保留8项隐藏几何状态并重新初始化Adam；隐藏状态此前累计11169次几何更新，训练终点累计14892，重置输出只增加3723。原G及官方PV的先前训练历史另计，不称为从官方PV仅训练本轮。

CPU核对四份9508结果的全部256个已保存prior/reference/final框、原生分数和root GT：阈值标签与oracle不一致均为0。跨进程初始两组Query选择变化2条，终点两组0条；冻结评分不代表CUDA逐位一致。没有从原始点重新计算所有正式行的Mask，原始成员精确验证限预检8条。独立审查为{audit['verdict']}、0个未解决阻断项，same-family/provisional、backend未attest；初始通信中断后在同一审查上下文恢复，未将源码审查当终态审查。审查限定单seed、反复使用的开发划分及实际已保存数组，不证明多seed稳健性、零样本泛化或新Nr3D/Sr3D效果。

最佳状态已实际严格CPU恢复：完整1304项模型状态一致，真实step0与空Adam状态恢复一致，直接工厂不依赖旧4511几何权重；此检查未重跑GPU验证。随后依用户授权删除4份已闭合非最佳权重，共{retention['released_bytes']}字节，保留唯一4848版本；官方PV、原G、V99和全部文本/原始结果证据保留，不创建负权重归档。详见{prefix}内真实清理、恢复、审查及接收记录。SUMMARY/NARRATIVE的先前pending字段保留原字节，由TERMINAL_ACCEPTANCE明确取代。

后续主线仍为PV-Ground，保留该空间参考机制并如实区分零更新与训练效果。ScanRefer门槛已达到；Nr3D/Sr3D需使用各自公平对应初始化、原生文本/对象接口完成独立训练与验证，当前没有新结果，整体目标仍未完成。長任务按预估结束前几分钟检查，必要复查间隔180～300秒；不提前重复查询。
'''
assert chr(65533) not in section
new = old+section.encode('utf-8')
assert new.startswith(old) and new.count(b'## 20.376.86 ')==1
names = ['analysis/SUMMARY.json','analysis/NARRATIVE_REPORT.md','analysis/EXPERIMENT_TRACKER.md',
         'analysis/EXPERIMENT_AUDIT.json','analysis/EXPERIMENT_AUDIT.md','analysis/TERMINAL_ACCEPTANCE.json',
         'analysis/TERMINAL_REVIEW_CALL.json','selected_candidate_CPU_restore.json','weight_retention.json',
         'fit_wait.json','complete_fit/INTAKE.json','complete_fit/fit_status.json','complete_fit/fit_controller.exit',
         'native_reference_spec.json','fused_mask_reference_spec.json',
         'postrun/selected_mask_reference_factory.py','postrun/restore_candidate_state.py',
         'postrun/publish_terminal_authorized.py','postrun/PUBLISH_TERMINAL_SOURCE_REVIEW.json',
         'postrun/PUBLISH_TERMINAL_SOURCE_REVIEW.md']
for arm in ('native_reference','fused_mask_reference'):
    names += [f'complete_fit/{arm}/receipt.json',f'complete_fit/{arm}/train.jsonl']
    for stage in ('initial_formal','formal'):
        names += [f'complete_fit/{arm}/{stage}/receipt.json',f'complete_fit/{arm}/{stage}/rows.jsonl']
payloads = {prefix+name:(root/name).read_bytes() for name in names}
payloads[prefix+'.gitattributes'] = b'** -text\n'
assert all(not name.endswith(('.pth','.pt','.npz')) and '.aris' not in name for name in payloads)
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
spec = json.loads((root/'native_reference_spec.json').read_bytes())
remote_code = '''import base64,hashlib,json,sys
from pathlib import Path
project=Path(sys.argv[1]);bundle=json.load(sys.stdin)
doc=project/bundle['doc'];old=doc.read_bytes()
assert hashlib.sha256(old).hexdigest()==bundle['old_sha256']
evidence=(project/bundle['prefix']).resolve()
assert str(evidence)=='/root/autodl-tmp/mcln_published_evidence_20261005/pvground_mask_reference_20261006/terminal_evidence'
assert not evidence.exists()
for name,encoded in bundle['files'].items():
    assert name.startswith(bundle['prefix']) and not name.endswith(('.pt','.pth','.npz')) and '.aris' not in name
    path=project/name;assert evidence in path.resolve().parents
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=base64.b64decode(encoded)
    with path.open('xb') as stream:stream.write(raw)
    assert path.read_bytes()==raw
new=base64.b64decode(bundle['new_doc']);assert new.startswith(old)
doc.write_bytes(new);assert doc.read_bytes()==new
print(json.dumps(dict(files=len(bundle['files']),handoff_sha256=hashlib.sha256(new).hexdigest())))
'''
bundle = dict(doc=doc,prefix=prefix,old_sha256=previous['handoff_sha256'],
    new_doc=base64.b64encode(new).decode(),files={name:base64.b64encode(raw).decode() for name,raw in payloads.items()})
stdin,stdout,stderr = client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python','-B','-c',remote_code,
    '/home/gb/new butd/butd_detr-main/MCLN-main']),timeout=180)
stdin.write(json.dumps(bundle).encode());stdin.flush();stdin.channel.shutdown_write()
raw = stdout.read()
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
remote = json.loads(raw)
client.close()
for name,raw in payloads.items():
    for repo in repos[:2]:
        path = repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
for path in copies:
    path.write_bytes(new)
heads = []
for index,repo in enumerate(repos):
    stage = [doc]
    if index<2:
        with (repo/'MANIFEST.md').open('a',encoding='utf-8') as stream:
            stream.write('\n- '+stamp+' Actual Mask-reference pair closed; initial5598/4848 best, trained5593/4832; audit/CPU restore/best-only retention complete.\n')
        stage += ['MANIFEST.md',*payloads]
    changed = subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git','-C',str(repo),'add','-f','--',*stage],stderr=subprocess.DEVNULL)
    subprocess.check_call(['git','-C',str(repo),'-c','core.whitespace=cr-at-eol,-blank-at-eof','diff','--cached','--check','--',*stage])
    if index<2:
        assert all(subprocess.check_output(['git','-C',str(repo),'show',':'+name])==raw for name,raw in payloads.items())
    assert subprocess.check_output(['git','-C',str(repo),'show',':'+doc]).startswith(
        subprocess.check_output(['git','-C',str(repo),'show',previous['heads'][index]+':'+doc]))
    subprocess.check_call(['git','-C',str(repo),'commit','--quiet','-m','Record actual Mask-reference results and retain the strict-metric best'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'-c','http.version=HTTP/1.1','push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0]==heads[0]
digest = hashlib.sha256(new).hexdigest()
assert digest==remote['handoff_sha256'] and all(path.read_bytes()==new for path in copies)
guard = workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py'
raw = guard.read_bytes();assert raw.count(previous['handoff_sha256'].encode())==1
guard.write_bytes(raw.replace(previous['handoff_sha256'].encode(),digest.encode()))
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.86',heads=heads,
    handoff_sha256=digest,four_local_and_remote_equal=True,github_main=heads[0],payload_count=len(payloads),
    execution_scope='ACTUAL_TERMINAL_EVIDENCE_PUBLISHED',best_hits=[5598,4848],best_actual_optimizer_step=0,
    trained_fused_hits=[5593,4832],retention_executed=True,released_bytes=retention['released_bytes'],
    raw_npz_published=False,weights_published=False,new_GPU_execution=False)
receipt_path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
state.update(time_cst=record['time_cst'],latest_publication=str(receipt_path),handoff_section=record['section'],
    handoff_sha256=digest,published_heads=heads,reference_retention_executed=True,
    reference_terminal_audit_result_received=True,metric_best_hits=[5598,4848],metric_best_optimizer_step=0,
    owned_gpu_job_active=False,overall_goal_complete=False)
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (workspace/'memory/2026-10-06.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPV-Ground '+record['time_cst']+': Doc86 actual terminal published; zero-update fused5598/4848 selected, trained5593/4832 below own start. Audit qualified, CPU strict1304/emptyAdam complete; four nonbest weights removed '+str(retention['released_bytes'])+'B. OriginalPV/G/V99 preserved. Scan dev target reached, Nr/Sr still untrained/goalunmet. Main '+heads[0]+'.\n')
print(json.dumps(record),flush=True)
