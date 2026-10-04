"""Publish actual source review, two-step proof and formal launch, without weights."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

import paramiko


local = Path(__file__).resolve().parent
receipt_path = local/'launch_publication.json'
assert not receipt_path.exists()
review = json.loads((local/'SOURCE_REVIEW.json').read_bytes())
call = json.loads((local/'SOURCE_REVIEW_CALL.json').read_bytes())
assert call['result_received'] and review['verdict']=='PASS' and not review['blocking_findings']
assert review['execution_scope']=='SOURCE_ONLY'
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
wait = json.loads((local/'preflight_wait.json').read_bytes())
assert wait['observer_closed'] and wait['terminal']['exitcode']==0 and not wait['terminal']['controller_alive']
proof = json.loads((local/'preflight_complete/preflight.json').read_bytes())
assert proof['status']=='pass' and proof['optimizer_steps']==2 and proof['batch_size']==8
assert proof['isolated_quality_route_verified'] and proof['geometry_provider_and_g_states_exact']
assert proof['weight_files_created']==0 and not proof['accuracy_result']
launch = json.loads((local/'fit_launch.json').read_bytes())
assert launch['formal_training_started'] and not launch['accuracy_result']
workspace = Path('C:/Users/gb')
previous_path = local.parent/'pvground_geometry_readback_20261004/terminal_publication_20261005.json'
previous = json.loads(previous_path.read_bytes())
repos = [workspace/'.codex_mcln_g0_20260905',workspace/'.codex_pvground_cs_20261002',workspace/'.codex_mcln_v99_internal_20260928']
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies = [r/doc for r in repos]+[workspace/'Desktop/document'/Path(doc).name]
old = copies[0].read_bytes()
assert hashlib.sha256(old).hexdigest()==previous['handoff_sha256']
assert all(p.read_bytes()==old for p in copies) and b'## 20.376.58 ' not in old
for repo,head in zip(repos,previous['heads']):
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==head
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
prefix = 'refine-logs/pvground_final_quality_20261005/'
names = ['EXPERIMENT_PLAN.md','native_final_quality.py','prepare_quality_draft.py','controller.py',
    'launch_quality_preflight_authorized.py','launch_quality_fit_authorized.py','observe_quality_authorized.py',
    'quality_fit_spec.json','quality_preflight_spec.json','IMPLEMENTATION_STATUS.json',
    'SOURCE_REVIEW.md','SOURCE_REVIEW.json','SOURCE_REVIEW_CALL.json','preflight_resource_check.json',
    'preflight_launch.json','preflight_wait.json','fit_resource_check.json','fit_launch.json',Path(__file__).name]
names += [str(p.relative_to(local)).replace('\\','/') for p in sorted((local/'runtime_bundle').glob('*.py'))]
names += [str(p.relative_to(local)).replace('\\','/') for p in sorted((local/'preflight_complete').iterdir()) if p.is_file()]
payloads = {prefix+n:(local/n).read_bytes() for n in names}
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp = client.open_sftp()
remote_project = '/home/gb/new butd/butd_detr-main/MCLN-main'
remote_run = launch['root']
with sftp.open(remote_run+'/fit_status.json','rb') as stream:
    status = json.loads(stream.read())
assert status['status']=='running' and status['stage']=='fit'
with sftp.open(remote_project+'/'+doc,'rb') as stream:
    assert stream.read()==old
with sftp.open(remote_run+'/preflight/preflight.json','rb') as stream:
    assert stream.read()==(local/'preflight_complete/preflight.json').read_bytes()
assert sftp.stat('/root/autodl-tmp/pvground_boundary_fit_20261004/distribution/terminal.pth').st_size==5587141
stamp = datetime.datetime.now().astimezone().isoformat()
section = f'''

## 20.376.58 原生bbs最终质量差异监督：源码与两步真实预检通过，正式训练已启动（{stamp}）

接续§20.376.57的冻结几何回写负结果，本项只改变训练目标，不扩大R、增加来源或重跑相同控制。实际起点仍是官方PV→原G→保留的4506分布几何父模型；全部父状态冻结/eval，新R从相同seed2027和零输出重新初始化，只训练96672参数/23项状态。不是从已经删除的4477负终点恢复。

当前实际前向为：原生点—体素骨干、跨模态编码与原G读取→Query/粗框及原生两路Mask→完整融合Mask范围统计→已训练的六面分布几何头→R读取44维真实几何证据、完整文本和六面角色→唯一原生语义头→原生last/bbs。保留256候选，同一所选Query输出Box和Mask；没有教师、第二套质量排名、推理GT门控或新对比正例。

新增native_final_quality.py仅在训练中使用最终部署框/root GT的停止梯度IoU。资格集合C是最后层真实Hungarian匹配root加原G已有的IoU>0.5未匹配候选；其他已匹配实例排除。资格仍是几何代理，不声称已确认完整语义身份。对每条表达采用mean_C[(s-u)-mean_C(s-u)]²，再对batch取均值，权重预先固定1.0；s为实际原生bbs，u为最终IoU。该式等于一半的有序候选对分数差/IoU差均方误差。原native+G、CE和对比的分母与目标不变，不再扩大正例集合或扫描归一化。

原生bbs是token softmax后的主语、修饰、代词、关系证据减其他实体证据，可能有符号；此处没有把它当作普通[0,1]概率，也没有对所有token广播质量标量或新增推理加权。目标是让既有root资格候选之间的实际分数差体现最终范围优劣，而非只认可更多候选。

真实fresh源码审查为PASS/SOURCE_ONLY，55个实际输入身份、23份新Python的3.7语法及15份继承工具原样一致已检查；same-family/provisional，Astra/max只是请求路由，实际后端未获证实。该审查不授予精度PASS。

真实GPU预检完成2次更新，批量8，无完整精度结果、无权重文件。已独立比较成对公式和梯度，检查池外及其他已匹配实例的直接语义梯度为0、同分情况下高IoU获得正确方向、单独质量梯度进入R，以及首步零输出和第二步内部梯度；同一前向几何/Mask保持，父状态与模型/AdamW内存保存恢复均实际验证。首步单独质量项的R输出权重梯度范数为{proof['witnesses'][0]['quality_route']['parameter_gradient_norms']['output.weight']:.8f}，原语义项为{proof['witnesses'][0]['semantic_route']['parameter_gradient_norms']['output.weight']:.8f}，并非只检查总loss梯度。原始证据位于{prefix}preflight_complete/；预检梯度不冒充正式增益。生成器的IMPLEMENTATION_STATUS.json保留创建时草稿状态，实际运行状态以新增预检及启动凭证为准。

正式训练于{launch['time_cst']}启动，screen为{launch['screen']}，实际控制进程为{launch['process']}。仅新质量组，比较已完成的同结构可见证据/native+G控制5615/4477及保留最佳5616/4506。29778条fit各一次，batch/有效batch8、累积1，3722个完整batch和2条尾batch，共3723次AdamW更新；lr1e-5、weight_decay0.0005、clip0.1不变。控制器逐批核对实际控制的完整样本顺序。有效batch、样本遍历量、更新次数一起固定，不机械按batch放大学习率。

实测相同结构旧控制fit+6887模块留出为6529.66秒，独立9508正式验证1491.62秒；本项预计约8200秒，首次检查在启动后6200秒，之后240秒轮询。6887留出是预训练见过场景的模块开发数据，不是正式9508验证。当前未完成新正式精度，不承诺超过50%或获得Nr3D/Sr3D增益。

两组此前较差R权重已删除，保留4506及必需父链；新组仅保留活动恢复与完成后严格指标更佳的终点。正式模型/AdamW恢复与9508条CPU阈值核对后，若未超过4506，控制器只删除本项所属terminal.pth且不新建权重归档；历史V99不在删除路径。

当前目标仍是同一模型5615/4754，最佳严格指标尚差248条。计划、实现、源码审查、两步运行证据和启动凭证均如实发布；新质量损失属于待验证训练方法，不能写成第三个已有效模块或已完成论文创新。
'''
new = old+section.encode('utf-8')
assert new.startswith(old) and new.count(b'## 20.376.58 ')==1
# All destination directories are inside the already authorized project tree.
parents = sorted({str(Path(p).parent).replace('\\','/') for p in payloads})
_,stdout,stderr = client.exec_command(shlex.join(['mkdir','-p','--',*[remote_project+'/'+p for p in parents]]),timeout=60)
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
for name,raw in payloads.items():
    for repo in repos[:2]:
        dest = repo/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(raw)
    with sftp.open(remote_project+'/'+name,'wb') as stream:
        stream.write(raw)
    with sftp.open(remote_project+'/'+name,'rb') as stream:
        assert stream.read()==raw
for p in copies:
    p.write_bytes(new)
with sftp.open(remote_project+'/'+doc,'wb') as stream:
    stream.write(new)
with sftp.open(remote_project+'/'+doc,'rb') as stream:
    assert stream.read()==new
sftp.close()
client.close()
heads = []
for index,repo in enumerate(repos):
    stage = [doc]
    if index<2:
        with (repo/'MANIFEST.md').open('a',encoding='utf-8') as stream:
            stream.write('\n- '+stamp+' final-native-bbs quality difference loss: sourcePASS, actual2update preflightPASS, one3723-update fit LAUNCHED, no new accuracy; protected4506, control4477.\n')
        stage += ['MANIFEST.md',*payloads]
    changed = subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git','-C',str(repo),'add','-f','--',*stage])
    subprocess.check_call(['git','-C',str(repo),'diff','--cached','--check','--',*stage])
    if index<2:
        for name,raw in payloads.items():
            assert subprocess.check_output(['git','-C',str(repo),'show',':'+name])==raw
    subprocess.check_call(['git','-C',str(repo),'commit','--quiet','-m','Add training-only final native quality supervision with real GPU preflight and launch'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'-c','http.version=HTTP/1.1','push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0]==heads[0]
digest = hashlib.sha256(new).hexdigest()
guard = workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py'
raw = guard.read_bytes()
assert raw.count(previous['handoff_sha256'].encode())==1
guard.write_bytes(raw.replace(previous['handoff_sha256'].encode(),digest.encode()))
assert all(p.read_bytes()==new for p in copies)
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.58',heads=heads,
    github_main=heads[0],handoff_bytes=len(new),handoff_sha256=digest,four_local_and_remote_equal=True,
    payload_count=len(payloads),source_review='PASS_SOURCE_ONLY_same_family_provisional',
    actual_two_step_preflight=True,formal_training_started=True,new_accuracy_result=False,
    protected_best_hits=[5616,4506],target_hits=[5615,4754],control_hits=[5615,4477],goal_achieved=False)
receipt_path.write_bytes((json.dumps(record,indent=2)+'\n').encode())
state_path = local/'active_continuation_state.json'
state = json.loads(state_path.read_bytes())
state.update(time_cst=record['time_cst'],status='FORMAL_QUALITY_FIT_ACTIVE_PUBLISHED',active_reviewer=None,
    source_only=False,gpu_preflight_executed=True,formal_training_started=True,latest_publication=str(receipt_path),
    published_heads=heads,handoff_section='20.376.58',handoff_sha256=digest,handoff_bytes=len(new),
    last_goal_turn_classification='PROGRESS_ACTUAL_QUALITY_PREFLIGHT_AND_FORMAL_LAUNCH_PUBLISHED')
state_path.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
for sibling in ('pvground_geometry_readback_20261004','pvground_face_conditioned_20261004'):
    p=local.parent/sibling/'NEXT_CONTINUATION.md'
    with p.open('a',encoding='utf-8') as stream:
        stream.write('\n\nSuccessor active actual run: '+str(local/'active_continuation_state.json')+'. Readback pair is CLOSED; final-quality fit is the only active run. Latest publication '+str(receipt_path)+', main'+heads[0]+', doc58. Do not relaunch old observers or controllers.\n')
with (workspace/'memory/2026-10-05.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+record['time_cst']+': final-quality signed-native-bbs/final-IoU differences inside originalG root pool sourcePASS and real2-step isolated-gradient/restore preflightPASS; fit actualLAUNCHED3723updates with reused4477control, all256/oneScore/frozen4506. Publishedmain'+heads[0]+'/doc58, fourlocal+remoteexact; no newaccuracy, targetACTIVE_UNMET. Current cursor '+str(state_path)+'.\n')
print(json.dumps(record),flush=True)
