"""Append actual passed checks and actual launch, preserving historical bytes."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import paramiko

local=Path(__file__).resolve().parent
workspace=Path('C:/Users/gb')
previous=json.loads((local.parent/'pvground_mask_branch_responsibility_20261005/publication.json').read_bytes())
launch=json.loads((local/'fit_launch.json').read_bytes())
assert (local/'fit_observer_started.json').exists()
wait=json.loads((local/'preflight_wait.json').read_bytes())
proofs={arm:json.loads((local/'preflight_complete'/arm/'preflight.json').read_bytes()) for arm in ('control','query_supported')}
assert wait['observer_closed'] and wait['exitcode']==0 and wait['status']['status']=='complete'
assert all(proof['status']=='pass' and proof['optimizer_steps']==2 for proof in proofs.values())
assert not launch['accuracy_result'] and launch['updates_per_arm']==3723
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
section=f'''

## 20.376.64 Query支撑候选的额外几何责任：两组真实预检闭合与正式对照启动（{stamp}）

承接§63的分支核对，现仅对未匹配、自己的Query Mask与原生融合Mask对root均IoU>0.5、最终Box IoU≤0.5的候选增加训练期几何责任。公共Text Mask单独合格不作资格；所有原Hungarian已匹配Query均排除，保护root和其他实例原职责。资格仍是训练GT下的支撑/重叠代理，不能称物理实例身份真值，也不进入推理。保留全部256候选，唯一原生bbs，Box/Mask同Query。

两组均从官方PV→原G→当前4506边界头重建，父网络与全零R冻结/eval，仅更新既有456102参数/10状态项的candidate_box_refiner。控制保留native+G+匹配六面分布loss(1/7)；策略另加既有(10L1+2GIoU+面均值DFL)/7。新增候选先在每条表达内取均值，空集合贡献0，再按实际batch均值；原匹配、原标签、原损失分母、分布节点/范围/clipping不变，超范围目标继续记录。没有新模型层、第二评分、教师、Mask再生成或推理GT门控。

真实两步GPU预检已经全部闭合，退出0。每组2优化步，均证明额外loss单独梯度仅到资格输出/已有几何头，非资格及原匹配输出无该直接梯度；父网络与R状态逐元素不变。缓存同一上游输入，在几何头更新后重放头与零R，原生bbs和全部Text/Query/alpha Mask张量逐元素一致；这是缓存输入见证，不宣称跨CUDA前向逐位一致。实际执行了空行和分布目标超范围情形，并在内存中保存/重载模型增量与优化器，未创建临时磁盘权重。控制峰值allocated/reserved={proofs['control']['peak_allocated_bytes']}/{proofs['control']['peak_reserved_bytes']}字节；策略={proofs['query_supported']['peak_allocated_bytes']}/{proofs['query_supported']['peak_reserved_bytes']}字节。工程预检没有精度结果。

正式两组已于{launch['time_cst']}在唯一GPU上顺序启动：control先fit+formal，再query_supported同流程。两组分别重新加载同一4506起点、重新初始化AdamW，不承接预检两步状态；seed2027，lr1e-5，WD0.0005，clip0.1，物理/有效batch8，累积1，每组29778条fit各一次，3722个完整batch8+末batch2=3723次更新。初始/终点模块留出各6887条，正式验证各9508条last/bbs。已有几何头阶段3723次，本轮终点几何累计7446次；原G适配历史另外披露。有效batch变化还会改变同样遍历次数下的更新预算，不能只线性缩放学习率。

按已完成上一项实际fit+两次6887评估6427.53秒、正式9508评估1422.47秒估计，两组约15700秒；几何反向的实际吞吐可能不同。唯一observer在6200秒首次检查，之后240秒间隔。资源/保存余量已按真实文件大小核对，产物写数据盘；保留active recovery，完整验证及审计后删除本轮非最佳权重，不新增负权重归档。当前最好5616/4506未被替换，目标同模型至少5615/4754，严格差248，ACTIVE_UNMET。不能将“正式已启动”写成完成或涨点。

source及启动补充复核均SOURCE_ONLY实际返回，requested Astra/max、actual backend未attested，same-family/provisional。证据目录refine-logs/pvground_query_supported_geometry_20261005；preflight_complete在远端使用数据盘目录链接，系统盘不重复存日志/模型。此快照仅报告工程检查和正式启动，不改历史结果。
'''
new=old+section.encode('utf-8')
names=['EXPERIMENT_PLAN.md','EXPERIMENT_TRACKER.md','query_supported_geometry.py','fit_body.py','prepare_fit.py','run_geometry_fit.py','GENERATION.json',
    'controller.py','control_spec.json','query_supported_spec.json','PREFLIGHT_PANEL.json','prepare_preflight_panel.py',
    'deploy_preflight_authorized.py','observe_preflight_authorized.py','collect_preflight_authorized.py',
    'launch_geometry_fit_authorized.py','prepare_fit_observer.py','observe_fit_authorized.py',
    'SOURCE_REVIEW.md','SOURCE_REVIEW.json','SOURCE_REVIEW_CALL.json','LAUNCH_REVIEW.md','LAUNCH_REVIEW.json','LAUNCH_REVIEW_CALL.json',
    'resource_check.json','preflight_launch.json','preflight_wait.json','fit_resource_check.json','fit_launch.json','fit_observer_started.json',Path(__file__).name]
for folder in ('preflight_complete','preflight_observations'):
    names.extend(str(path.relative_to(local)).replace('\\','/') for path in sorted((local/folder).rglob('*')) if path.is_file())
prefix='refine-logs/pvground_query_supported_geometry_20261005/'
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
snapshot=launch['root']+'/preflight_complete'
sftp.mkdir(snapshot)
for arm in ('control','query_supported'):
    sftp.mkdir(snapshot+'/'+arm)
for path in sorted((local/'preflight_complete').rglob('*')):
    if path.is_file():
        target=snapshot+'/'+str(path.relative_to(local/'preflight_complete')).replace('\\','/')
        with sftp.open(target,'wx') as stream:
            stream.write(path.read_bytes())
sftp.symlink(snapshot,evidence+'/preflight_complete')
sftp.mkdir(evidence+'/preflight_observations')
for name,raw in payloads.items():
    if not name.startswith(prefix+'preflight_complete/') or name==prefix+'preflight_complete/INTAKE.json':
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
            stream.write('\n- '+stamp+' Both Query-supported geometry preflights closed; same-start head-only control/strategy fit launched. No new accuracy result; best5616/4506 retained.\n')
        stage+=['MANIFEST.md',*payloads]
    changed=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git','-C',str(repo),'add','-f','--',*stage],stderr=subprocess.DEVNULL)
    subprocess.check_call(['git','-C',str(repo),'diff','--cached','--check','--',*stage])
    if index<2:
        for name,raw in payloads.items():
            indexed=subprocess.check_output(['git','-C',str(repo),'show',':'+name])
            assert indexed==raw.replace(b'\r\n',b'\n'),name
    indexed=subprocess.check_output(['git','-C',str(repo),'show',':'+doc])
    old_index=subprocess.check_output(['git','-C',str(repo),'show',previous['heads'][index]+':'+doc])
    assert indexed.startswith(old_index) and indexed.count(b'## 20.376.64 ')==1
    subprocess.check_call(['git','-C',str(repo),'commit','--quiet','-m','Start same-budget Query-supported geometry responsibility comparison'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'-c','http.version=HTTP/1.1','push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0]==heads[0]
digest=hashlib.sha256(new).hexdigest()
guard=workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py'
raw=guard.read_bytes();assert raw.count(previous['handoff_sha256'].encode())==1
guard.write_bytes(raw.replace(previous['handoff_sha256'].encode(),digest.encode()))
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.64',heads=heads,
    github_main=heads[0],handoff_bytes=len(new),handoff_sha256=digest,four_local_and_remote_equal=True,
    remote_sync_pending=False,payload_count=len(payloads),preflight_updates_per_arm=2,accuracy_result=False,
    fit_launched=True,protected_best_hits=[5616,4506],full_goal_status='ACTIVE_UNMET')
(local/'publication.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
state=json.loads((local/'active_continuation_state.json').read_bytes())
state.update(status='TWO_ARM_GEOMETRY_FIT_PUBLISHED_RUNNING',owned_gpu_job_active=True,
    latest_publication=str(local/'publication.json'),time_cst=record['time_cst'])
(local/'active_continuation_state.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
with (workspace/'memory/2026-10-05.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+stamp+': Query-supported geometry two-arm preflight PASS and fullfit launched, doc64/main'+heads[0]+'. Each3723 updates/B8/29778 once; geometric head total7446 after terminal. No accuracy yet; best5616/4506 gap248 ACTIVE_UNMET.\n')
print(json.dumps(record),flush=True)
