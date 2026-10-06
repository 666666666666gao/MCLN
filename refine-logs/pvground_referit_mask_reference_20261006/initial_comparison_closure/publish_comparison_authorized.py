"""Append only actual R1 failure and bounded initial comparison to Doc87."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

import paramiko

root=Path(__file__).resolve().parent
assert not (root/'comparison_publication.json').exists()
review=json.loads((root/'PUBLISH_COMPARISON_SOURCE_REVIEW.json').read_bytes())
assert review['execution_scope']=='SOURCE_ONLY' and review['verdict'] in ('PASS','WARN') and not review['blocking_findings']
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
goal=json.loads((root/'current_research_goals.json').read_bytes())
assert goal['scanrefer']['minimum_hits025']==5620 and goal['scanrefer']['minimum_hits050']==4764 and not goal['multiseed']
diag=root/'initial_comparison'
source_review=json.loads((diag/'SOURCE_REVIEW.json').read_bytes())
assert source_review['execution_scope']=='SOURCE_ONLY' and source_review['verdict']=='PASS' and not source_review['blocking_findings']
for item in source_review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
failed=json.loads((root/'preflight_wait.json').read_bytes())
assert failed['observer_closed'] and failed['terminal']['controller_exit']==1
assert not failed['terminal']['controller_alive'] and not failed['terminal']['receipts']
wait=json.loads((diag/'wait.json').read_bytes())
assert wait['observer_closed'] and wait['terminal']['controller_exit']==0 and not wait['terminal']['controller_alive']
result=json.loads((diag/'complete/nr3d_native/comparison.json').read_bytes())
assert result['model_forwards']==3 and result['optimizer_steps']==0 and result['formal_rows']==0 and result['new_weights_saved']==0
launch=json.loads((diag/'launch.json').read_bytes())
previous_path=root/'preflight_publication.json'
previous=json.loads(previous_path.read_bytes());assert previous['section']=='20.376.87'
state_path=root.parent/'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state=json.loads(state_path.read_bytes());assert Path(state['latest_publication']).resolve()==previous_path.resolve()
workspace=Path('C:/Users/gb')
repos=[workspace/'.codex_mcln_g0_20260905',workspace/'.codex_pvground_cs_20261002',workspace/'.codex_mcln_v99_internal_20260928']
doc='docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies=[repo/doc for repo in repos]+[workspace/'Desktop/document'/Path(doc).name]
old=copies[0].read_bytes();assert hashlib.sha256(old).hexdigest()==previous['handoff_sha256']
assert all(path.read_bytes()==old for path in copies)
for repo,head in zip(repos,previous['heads']):
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==head
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
contract='idea-stage/docs/research_contract.md'
old_contract=(root/'research_contract_before_goal_update.md').read_bytes()
new_contract=(root/'current_research_contract.md').read_bytes()
assert all((repo/contract).read_bytes()==old_contract for repo in repos[:2])
stamp=datetime.datetime.now().astimezone().isoformat()
prefix='refine-logs/pvground_referit_mask_reference_20261006/initial_comparison_closure/'
pairs={}
for key in ('native_A_vs_native_B','native_A_vs_installed_C','native_B_vs_installed_C'):
    item=result[key]
    pairs[key]=dict(semantic_max_abs=item['semantic_max_abs'], x_query_max_abs=item['x_query_max_abs'],
        query_xyz_max_abs=item['query_xyz_max_abs'], center_max_abs=item['center_max_abs'],
        size_max_abs=item['size_max_abs'], mask_allclose_rows=sum(row['query_mask_original_allclose'] for row in item['rows']),
        own_sign_changes=sum(row['query_mask_sign_changes'] for row in item['rows']),
        fused_sign_changes=sum(row['fused_mask_sign_changes'] for row in item['rows']),
        mask_max_abs=max(row['query_mask_max_abs'] for row in item['rows']),
        super_xyz_max_abs=max(row['super_xyz_max_abs'] for row in item['rows']))
section=f'''

## 20.376.88 Nr/Sr首次预检的实际失败与零更新三前向诊断（{stamp}）

§87启动记录保留，不改写成通过。原控制器766996实际退出1，首个Nr-native在语义allclose之后、Query Mask的allclose(1e-5/1e-5)断言处退出；尚未创建优化器，0更新、0正式评估，其他三项未启动。原失败run.log、源码和审查均保留；旧status的running字段由实际exit文件和已关闭观察器纠正。这是初始化对照检查失败，不是训练后精度负结果。

为区分原路径重复波动和安装路径差异，另启隔离诊断：对应Nr作者权重、相同真实4指代＋4检测增强输入，固定seed2027，各次重置同一RNG、eval/no_grad、fresh dict(inputs)。A/B为相同原生分支，C恢复新reader并安装零初始化几何/全零R；恰好3次前向、0优化步、0权重保存、0正式验证。实际启动{launch['time_cst']}，控制器{launch['controller_pid']}，实际完成{result['time_cst']}，耗时{result['elapsed_seconds']:.2f}秒。唯一观察器按启动后五分钟首次查看，随后240秒，实际查询{wait['observation_count']}次。未重跑失败四项、未放宽误差门槛。

实际成对摘要：

```json
{json.dumps(pairs,ensure_ascii=False,indent=2)}
```

报告保留逐行Query Mask差值、自身/融合logit符号变化及超点中心差值；语义/x_query/query坐标为全batch最大差。1235官方状态相等见证对应初始化和C前，不是C后恢复声明。全256原始成员空间参考也执行实际见证。SOURCE_ONLY补充审查PASS、0阻断，same-family/provisional、实际backend和运行成功不由审查者认证。

三前向只区分本batch的实际重复波动与跨路径差异，不能独自确定scatter_add_是原因、建立全数据容差或宣称所有安装输出等价。原模型超点中心经GPU scatter_mean/grouping进入Mask分支，几何头位于Mask生成之后；该源码路径只是诊断线索。依据实际数值的最小R2源码修正仅作为准备保留，本节不将诊断替代M0或Nr/Sr完整精度。用户在本次明确调整研究顺序，R2 GPU及Nr/Sr正式训练不启动。

用户最新目标已更新：ScanRefer同一完整模型Acc@0.25严格超过59.1%、Acc@0.5严格超过50.1%，9508条至少5620/4764命中。当前5598/4848，宽松项还差22条，严格项已满足这项新门槛；还需三项有直接消融和实际作用的有效模块，零输出或未验证分支不计作贡献。先在ScanRefer满足两阈值与三模块证据，再固定完整结构、分别加载作者Sr3D/Nr3D对应预训练权重独立训练；对应原版baseline采用相同作者核心起点；不能混合不同最终方法版本或用Scan终点零更新冒充独立训练。候选方向为支撑空间参考、边界观测精修、参考约束几何学习，后两项仍待验证。旧“接近V99即可转Nr/Sr”准入被本次要求替代，当前无活动GPU任务。

Scan best5598/4848及原PV/G/V99保持；用户已许可的4756份旧远端NPZ清理收据在§87，未新增删除。当前固定seed2027，不做多seed挑选/集成。Nr/Sr正式训练仍未启动，三个基准目标未完成。实际源码、原失败文本、诊断和审查见{prefix}；不发布权重、原始NPZ、凭据或.aris。
'''
new=old+section.encode('utf-8');assert new.startswith(old) and new.count(b'## 20.376.88 ')==1
base_names=['CURRENT_RESEARCH_GOALS.md','current_research_goals.json','current_research_contract.md',
    'research_contract_before_goal_update.md','prepare_goal_publication.py',
    'RUNTIME_FAILURE_SOURCE_DIAGNOSIS.md','preflight_wait.json',
    'complete_preflight/nr3d_native/run.log','complete_preflight/nr3d_native/preflight.exit',
    'complete_preflight/preflight_controller.exit','prepare_initial_comparison.py',
    'publish_comparison_authorized.py','PUBLISH_COMPARISON_SOURCE_REVIEW.json','PUBLISH_COMPARISON_SOURCE_REVIEW.md']
diag_names=['compare_initial.py','spec.json','launch_comparison_authorized.py',
    'referit_training_targets.py','pvground_referit_fit_dataset.py','mask_reference.py','query_supported_geometry.py',
    'SOURCE_REVIEW.json','SOURCE_REVIEW.md','launch.json','observer_wait.json','observe_comparison_authorized.py',
    'wait.json','complete/nr3d_native/comparison.json','complete/run.log','complete/controller.exit']
payloads={prefix+name:(root/name).read_bytes() for name in base_names}
payloads.update({prefix+'initial_comparison/'+name:(diag/name).read_bytes() for name in diag_names})
payloads[prefix+'.gitattributes']=b'** -text\n'
assert all(not name.endswith(('.pt','.pth','.npz')) and '.aris' not in name for name in payloads)
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
spec=json.loads((root/'preflight_spec.json').read_bytes())
code='''import base64,hashlib,json,sys
from pathlib import Path
project=Path(sys.argv[1]);b=json.load(sys.stdin);doc=project/b['doc'];old=doc.read_bytes()
assert hashlib.sha256(old).hexdigest()==b['old_sha256']
contract=project/b['contract']
assert contract.read_bytes()==base64.b64decode(b['old_contract'])
evidence=(project/b['prefix']).resolve()
assert str(evidence)=='/root/autodl-tmp/mcln_published_evidence_20261005/pvground_referit_mask_reference_20261006/initial_comparison_closure'
assert not evidence.exists()
for name,encoded in b['files'].items():
    assert name.startswith(b['prefix']) and not name.endswith(('.pt','.pth','.npz')) and '.aris' not in name
    p=project/name;assert evidence in p.resolve().parents;p.parent.mkdir(parents=True,exist_ok=True)
    raw=base64.b64decode(encoded)
    with p.open('xb') as f:f.write(raw)
    assert p.read_bytes()==raw
new=base64.b64decode(b['new_doc']);assert new.startswith(old);doc.write_bytes(new)
assert doc.read_bytes()==new
new_contract=base64.b64decode(b['new_contract']);contract.write_bytes(new_contract)
assert contract.read_bytes()==new_contract
print(json.dumps(dict(files=len(b['files']),handoff_sha256=hashlib.sha256(new).hexdigest())))
'''
bundle=dict(doc=doc,prefix=prefix,contract=contract,old_contract=base64.b64encode(old_contract).decode(),
    new_contract=base64.b64encode(new_contract).decode(),old_sha256=previous['handoff_sha256'],new_doc=base64.b64encode(new).decode(),
    files={name:base64.b64encode(raw).decode() for name,raw in payloads.items()})
stdin,stdout,stderr=client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python','-B','-c',code,
    '/home/gb/new butd/butd_detr-main/MCLN-main']),timeout=180)
stdin.write(json.dumps(bundle).encode());stdin.flush();stdin.channel.shutdown_write()
raw=stdout.read();assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
remote=json.loads(raw);client.close()
for name,raw in payloads.items():
    for repo in repos[:2]:
        path=repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
for path in copies:path.write_bytes(new)
for repo in repos[:2]:(repo/contract).write_bytes(new_contract)
heads=[]
for index,repo in enumerate(repos):
    stage=[doc]
    if index<2:
        with (repo/'MANIFEST.md').open('a',encoding='utf-8') as stream:
            stream.write('\n- '+stamp+' Actual R1 failed before updates; bounded three-forward zero-update diagnostic closed, no new formal metrics.\n')
        stage+=['MANIFEST.md',contract,*payloads]
    changed=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git','-C',str(repo),'add','-f','--',*stage],stderr=subprocess.DEVNULL)
    subprocess.check_call(['git','-C',str(repo),'-c','core.whitespace=cr-at-eol,-blank-at-eof','diff','--cached','--check','--',*stage])
    if index<2:
        assert all(subprocess.check_output(['git','-C',str(repo),'show',':'+name])==raw for name,raw in payloads.items())
    assert subprocess.check_output(['git','-C',str(repo),'show',':'+doc]).startswith(
        subprocess.check_output(['git','-C',str(repo),'show',previous['heads'][index]+':'+doc]))
    subprocess.check_call(['git','-C',str(repo),'commit','--quiet','-m','Record ReferIt preflight failure and bounded zero-update diagnostic'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'-c','http.version=HTTP/1.1','push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0]==heads[0]
digest=hashlib.sha256(new).hexdigest();assert digest==remote['handoff_sha256'] and all(p.read_bytes()==new for p in copies)
guard=workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py';raw=guard.read_bytes()
assert raw.count(previous['handoff_sha256'].encode())==1
guard.write_bytes(raw.replace(previous['handoff_sha256'].encode(),digest.encode()))
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.88',heads=heads,
    handoff_sha256=digest,four_local_and_remote_equal=True,github_main=heads[0],payload_count=len(payloads),
    execution_scope='ACTUAL_INITIAL_COMPARISON_NOT_TRAINING',new_formal_result=False,
    controller_pid=launch['controller_pid'],raw_npz_published=False,weights_published=False)
(root/'comparison_publication.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
state.update(time_cst=record['time_cst'],latest_publication=str(root/'comparison_publication.json'),
    handoff_section=record['section'],handoff_sha256=digest,published_heads=heads,
    referit_initial_comparison_launch=str(diag/'launch.json'),referit_initial_comparison_controller_pid=launch['controller_pid'],
    owned_gpu_job_active=False,overall_goal_complete=False,next_action='ScanRefer >59.1/>50.1 and three effective modules first; R2 preparation retained, Nr/Sr formal deferred',
    current_research_goals=str(root/'current_research_goals.json'),scanrefer_minimum_hits=[5620,4764],
    effective_modules_required=3,single_seed=2027,no_multiseed=True)
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (workspace/'memory/2026-10-06.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPV-Ground '+record['time_cst']+': Doc88 actual originalM0 exit1/update0 and bounded3forward diagnostic closed controller'+str(launch['controller_pid'])+'; originalfailure kept, no tolerancechange/newformal/Scan transfer. Main '+heads[0]+'. Fixed2027/no multiseed.\n')
print(json.dumps(record),flush=True)
