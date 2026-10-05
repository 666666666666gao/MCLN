import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import paramiko

local=Path(__file__).resolve().parent
workspace=Path('C:/Users/gb')
previous=json.loads((local.parent/'pvground_mask_geometry_responsibility_20261005/publication.json').read_bytes())
summary=json.loads((local/'analysis/SUMMARY.json').read_bytes())
audit=json.loads((local/'analysis/EXPERIMENT_AUDIT.json').read_bytes())
review=json.loads((local/'analysis/TERMINAL_REVIEW_CALL.json').read_bytes())
wait=json.loads((local/'wait.json').read_bytes())
assert summary['status']=='CPU_RECOUNT_PASS' and not summary['accuracy_result']
assert audit['verdict'] in ('PASS','WARN') and not audit['blocking_findings'] and review['result_received']
assert wait['observer_closed'] and wait['exitcode']==0 and wait['status']['protected_parent_hashes_exact']
assert not (local/'publication.json').exists()
repos=[workspace/'.codex_mcln_g0_20260905',workspace/'.codex_pvground_cs_20261002',workspace/'.codex_mcln_v99_internal_20260928']
doc='docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies=[repo/doc for repo in repos]+[workspace/'Desktop/document'/Path(doc).name]
old=copies[0].read_bytes()
assert hashlib.sha256(old).hexdigest()==previous['handoff_sha256'] and all(path.read_bytes()==old for path in copies)
for repo,head in zip(repos,previous['heads']):
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==head
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
stamp=datetime.datetime.now().astimezone().isoformat()
c=summary['counts']
section=f'''

## 20.376.63 候选自身Query支撑与公共Text支撑的真实训练接口核对（{stamp}）

沿§62的具体缺测执行了新的只读分路核对：同一保护的官方PV→原G→4506边界头，加全新零输出R并全部冻结/eval；seed2027、前8个增强fit batch，共64条/16384候选。与§62逐行身份、增强点与GT框一致。不是正式9508条精度评估，也没有新loss更新、优化器或权重文件。

分别保存Text、Query、原生融合Mask相对root的交并整数计数、IoU、alpha、native匹配槽及Box输出。Text logit在全部256候选中相同；每行所选Query的三种Mask另外在原始输入点上展开核验。CPU从整数交并重算资格一致。当前融合Mask合格/Box不足/未匹配候选共{c['fused_mask_only_unmatched']}；其中候选自己的Query Mask也合格{c['own_query_confirmed']}（{summary['own_query_fraction_of_fused_mask_only_unmatched']*100:.4f}%），自身Query不合格而公共Text合格{c['query_not_qualified_text_qualified']}；两条独立分支都不合格但融合合格{c['neither_branch_qualified']}。自身Query确认集合出现在{c['rows_with_own_query_confirmed']}/64条；Text本身合格{summary['rows_text_mask_qualified']}/64条。不能把这些资格全部说成真实物理身份已确认，也不能推定正式验证发生率。

独立CUDA前向相对§62，Box过0.5资格变化{summary['prior_box_threshold_differences']}个，融合Mask过0.5变化{summary['prior_fused_threshold_differences']}个，匹配槽变化{summary['prior_matched_slot_differences']}个；旧数组和诊断数字原样保留。64行仍为root-only，因此没有非空多实例责任的真实运行证明。模型状态逐元素不变、梯度为空，保护父权重SHA一致，0优化步/0新权重/0正式精度结果。

下一项准备限定为候选自身Query与融合Mask同时资格通过的未匹配框不足候选的直接几何责任对照，而不靠Text单独资格；所有原Hungarian匹配候选排除，原监督保持，附加项独立按表达/实际batch归一化。只更新现有几何头，冻结父网络与R；一套native bbs、全部256候选、同Query框/Mask不变。当前仍是实现草稿，不能记成已经训练或有效。有效batch8、累积1、29778条一次对应3723更新（3722×8+末批2），两组均重新初始化优化器；不能把4506起点当作同预算续训控制。

独立source/terminal复核已实际返回，终态{audit['verdict']}，same-family/provisional，requested Astra/max且实际backend未attested。原始Mask全张量未在CPU重新执行，只重算保存交并并核对GPU点展开见证。证据在refine-logs/pvground_mask_branch_responsibility_20261005；complete链接原数据盘目录，无重复系统盘候选数组。最好5616/4506保持，目标同模型至少5615/4754，严格缺248，ACTIVE_UNMET。未产生新权重，无负结果归档。
'''
new=old+section.encode('utf-8')
names=['EXPERIMENT_PLAN.md','prepare_probe.py','probe_body.py','run_mask_branch_probe.py','spec.json','GENERATION.json',
    'controller.py','launch_probe_authorized.py','observe_probe_authorized.py','collect_probe_authorized.py','analyze_probe.py',
    'SOURCE_REVIEW.md','SOURCE_REVIEW.json','SOURCE_REVIEW_CALL.json','resource_check.json','launch.json','wait.json',Path(__file__).name]
for folder in ('complete','analysis','observations'):
    names.extend(str(path.relative_to(local)).replace('\\','/') for path in sorted((local/folder).rglob('*')) if path.is_file())
prefix='refine-logs/pvground_mask_branch_responsibility_20261005/'
payloads={prefix+name:(local/name).read_bytes() for name in names}
assert all(not name.endswith(('.pth','.pt')) and '.aris' not in name for name in payloads)
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
project='/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(project+'/'+doc,'rb') as stream:
    assert stream.read()==old
evidence=project+'/'+prefix.rstrip('/')
assert not any(entry.filename==Path(prefix.rstrip('/')).name for entry in sftp.listdir_attr(project+'/refine-logs'))
sftp.mkdir(evidence)
sftp.symlink(json.loads((local/'spec.json').read_bytes())['root'],evidence+'/complete')
for folder in ('analysis','observations'):
    sftp.mkdir(evidence+'/'+folder)
for name,raw in payloads.items():
    if not name.startswith(prefix+'complete/') or name==prefix+'complete/INTAKE.json':
        with sftp.open(project+'/'+name,'wb') as stream:
            stream.write(raw)
    with sftp.open(project+'/'+name,'rb') as stream:
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
        for name,raw in payloads.items():
            path=repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        with (repo/'MANIFEST.md').open('a',encoding='utf-8') as stream:
            stream.write('\n- '+stamp+' Separate native Query/Text/fused root-support probe, fixed augmentedfit64, audited; no optimizer or weights. Best5616/4506 unchanged.\n')
        stage+=['MANIFEST.md',*payloads]
    changed=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git','-C',str(repo),'add','-f','--',*stage],stderr=subprocess.DEVNULL)
    subprocess.check_call(['git','-C',str(repo),'diff','--cached','--check','--',*stage])
    if index<2:
        for name,raw in payloads.items():
            indexed=subprocess.check_output(['git','-C',str(repo),'show',':'+name])
            assert indexed==(raw if name.endswith('.npz') else raw.replace(b'\r\n',b'\n')),name
    indexed=subprocess.check_output(['git','-C',str(repo),'show',':'+doc])
    old_index=subprocess.check_output(['git','-C',str(repo),'show',previous['heads'][index]+':'+doc])
    assert indexed.startswith(old_index) and indexed.count(b'## 20.376.63 ')==1
    subprocess.check_call(['git','-C',str(repo),'commit','--quiet','-m','Separate candidate Query support from shared Text Mask qualification'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'-c','http.version=HTTP/1.1','push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0]==heads[0]
digest=hashlib.sha256(new).hexdigest()
guard=workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py'
raw=guard.read_bytes();assert raw.count(previous['handoff_sha256'].encode())==1
guard.write_bytes(raw.replace(previous['handoff_sha256'].encode(),digest.encode()))
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.63',heads=heads,
    github_main=heads[0],handoff_bytes=len(new),handoff_sha256=digest,four_local_and_remote_equal=True,
    remote_sync_pending=False,payload_count=len(payloads),optimizer_steps=0,accuracy_result=False,
    protected_best_hits=[5616,4506],full_goal_status='ACTIVE_UNMET')
(local/'publication.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
state=json.loads((local/'active_continuation_state.json').read_bytes())
state.update(status='BRANCH_PROBE_FULLY_PUBLISHED',owned_gpu_job_active=False,observer_closed=True,active_reviewer=None,
    latest_publication=str(local/'publication.json'),time_cst=record['time_cst'],next_work_owner=str(local.parent/'pvground_query_supported_geometry_20261005'))
(local/'active_continuation_state.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
with (workspace/'memory/2026-10-05.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+stamp+': separate Query/Text/fused fixed64 probe closed+audited, Query-confirmed '+str(c['own_query_confirmed'])+'/'+str(c['fused_mask_only_unmatched'])+', published doc63/main'+heads[0]+'. No optimizer/newweights/accuracy; best5616/4506, gap248. Next conditional geometry target draft, source/preflight gates before fullfit.\n')
print(json.dumps(record),flush=True)
