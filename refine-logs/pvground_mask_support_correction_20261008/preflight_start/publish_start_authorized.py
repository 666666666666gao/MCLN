"""Append the implemented architecture and actual M0 launch; no accuracy claim."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

import paramiko


root = Path(__file__).resolve().parent
assert not (root / 'start_publication.json').exists()
review = json.loads((root / 'SOURCE_REVIEW_SUPPLEMENT.json').read_bytes())
assert review['execution_scope'] == 'SOURCE_ONLY'
assert review['verdict'] in ('PASS', 'WARN') and not review['blocking_findings']
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
launch = json.loads((root / 'preflight_launch.json').read_bytes())
observer = json.loads((root / 'observer_wait.json').read_bytes())
assert launch['status'] == 'PREFLIGHT_LAUNCHED_NOT_PASSED' and not launch['accuracy_result']
assert launch['controller_pid'] == observer['controller_pid']
assert launch['root'] == '/root/autodl-tmp/pvground_mask_support_correction_20261008'
assert not (root / 'fit_launch.json').exists()
previous = json.loads((root.parent / 'pvground_support_boundary_cases_20261007/terminal_publication.json').read_bytes())
assert previous['section'] == '20.376.97'
workspace = Path('C:/Users/gb')
repos = [workspace / '.codex_mcln_g0_20260905', workspace / '.codex_pvground_cs_20261002',
         workspace / '.codex_mcln_v99_internal_20260928']
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies = [repo / doc for repo in repos] + [workspace / 'Desktop/document' / Path(doc).name]
old = copies[0].read_bytes()
assert hashlib.sha256(old).hexdigest() == previous['handoff_sha256']
assert all(path.read_bytes() == old for path in copies)
for repo, head in zip(repos, previous['heads']):
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip() == head
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
stamp = datetime.datetime.now().astimezone().isoformat()
prefix = 'refine-logs/pvground_mask_support_correction_20261008/preflight_start/'
section = f'''

## 20.376.98 候选Mask支撑修正：实现、同起点对照与真实预检启动（{stamp}）

前节191表达的成员分析已经封存：仅支持fresh-forward描述及旧缓存面坐标的保守来源分析，不是历史预测恢复，不产生新9508成绩或有效模块结论。本节不重跑该诊断。当前最好仍是5598/4848，目标仍为同一完整ScanRefer模型5620/4764及三个实际有效贡献；之后固定完整方法，从作者对应权重独立训练Sr3D/Nr3D。单seed2027，保留全部256候选、一套原生last/bbs和同一Query的Box/Mask。

新的最小结构在原PV/G生成两路Mask之后、原生融合及精确成员Mask空间参考之前，修正候选自己的Query Mask logit。每个Query/超点的输入为32维Query内容、32维超点内容、Text/Query/fused概率及分歧/原生scalar-alpha共5维、真实成员数量1维、成员均值/最小/最大位置相对原粗框中心与尺寸9维，合计79维。32维投影后以79→64→64→1的MLP产生残差，末层零初始化；每组新增27841参数、10状态张量。control仅将9维显式粗框几何置零，treatment读取它们；共同原生特征已经含空间信息，不能写成control完全没有几何。更正后的Query logit仍与原Text logit按原scalar-alpha融合，没有新阈值、第二排名、Box质量分数或教师。

父PV/G、文本评分、原几何头与全零R全部冻结。两组共享每个真实batch的一次冻结父前向，各自维护相同初始状态、独立AdamW。候选对应先由未修正父输出建立原生Hungarian匹配，每个候选保留其实际有效GT，不把其他实例写成GT槽0、不扩大正例。仅训练原生中有新增头梯度的Mask项：5 Query focal +1 Query Dice +10 fused focal +2 fused Dice，按原有效GT数归一化，不除7；冻结Text/语义/Box及hard-pseudo对应项不构造额外训练目标。精确Mask参考的硬阈值/极值路径仍detach，不称为Box损失端到端更新Mask。原生语义头仍调用一次；改变支撑会影响同一个原生评分选中Query的Mask及空间范围，不增加后处理选择链。

源码已通过Python3.7语法检查和独立SOURCE_ONLY审查（{review['verdict']}，blocking=0；同系列暂定审查、实际后台/effort未认证）。隔离源端口复制完整98份已声明旧PV文件并改一份pv_ground.py；未修改旧训练目录、数据、预训练权重或warm环境。当前实际启动的是两步GPU预检：{launch['time_cst']}，controller {launch['controller_pid']}，screen {launch['screen']}。应核对零初始化Mask/Box等价、原criterion四项数值、第一步输出层/第二步内部梯度、冻结父状态、完整CPU模型重建、新头与Adam的实际GPU恢复，以及实际集成PV一次前向所捕获的同帧输入上配对路径与Mask/最终Box的严格相等；不是比较两次独立backbone的结果，也不冒充完整冷启动GPU父模型恢复。启动不等于这些检查已通过。

初次源码审查发现并保留两个阻断项：已有同RNG重复完整前向的Query Mask漂移约4.2e-5至1.2e-4，以及旧冻结父6887遍历中row16804参考框尺寸差0.03451145米。故接线比较改为一次实际前向的缓存，不放宽阈值来宣称重复前向相等。initial/terminal只强制输入、GT身份及父参数不变，完整记录父Query/Box数值变化；初始正式的两个零头必须与当次父输出一致，若父计数相对历史5598/4848略变，不更新历史best或计为方法收益。后续方法增量使用同一次前向父对照和本轮另一组，另保留相对历史最好差值。

唯一观察器Windows PID {observer['observer_local_pid']}，首查{observer['first_observation_cst']}，之后仅必要时240秒复查；根据既有共享父M0约469.56秒、此次增加原生集成前向，暂估900秒，首查在预计终点前三分钟。预检不生成新权重、无正式精度。正式训练尚未启动；通过真实预检和保存空间检查后，重新初始化模型/优化器，以batch8、LR1e-5、WD5e-4、clip0.1、29778条fit各一次/3723更新每组完成同预算对照。6887已见训练场景留出与9508完整正式验证分开；完整初始/终态遍历保持原batch序列，不能用只执行选中batch的RNG状态声称历史恢复。所有正式变化需同时比较本轮control和5598/4848强起点，记录修复/破坏及Mask指标。

清理长期授权已保存，已归档191远端临时数组126198802字节只删除一次。最近只读库存中5份权重均为所需官方Scan/Nr/Sr、原G及当前最好delta，未再删除必要依赖。新对照闭合并留存证据后，仅保留指标最好状态及必要依赖。三模块有效性、Scan5620/4764与新的Nr/Sr训练仍未完成，不因源码或预检启动提前晋级。
'''
new = old + section.encode('utf-8')
assert new.startswith(old) and new.count(b'## 20.376.98 ') == 1
spec = json.loads((root / 'pair_spec.json').read_bytes())
names = list(spec['new_runner_files']) + ['controller.py','pair_spec.json','source_port.json',
    'EXPERIMENT_PLAN.md','LOCAL_SOURCE_PREPARATION.json','SOURCE_REVIEW.json','SOURCE_REVIEW.md',
    'SOURCE_REVIEW_SUPPLEMENT.json','SOURCE_REVIEW_SUPPLEMENT.md','REPEATABILITY_EVIDENCE.json',
    'deploy_preflight_authorized.py','observe_preflight_authorized.py','publish_start_authorized.py',
    'resource_check.json','preflight_launch.json','runtime_overlay/PV-Ground/models/pv_ground.py']
payloads = {prefix + name: (root / name).read_bytes() for name in names}
payloads[prefix + '.gitattributes'] = b'** -text\n'
assert all(not name.endswith(('.pt','.pth','.npz')) and '.aris' not in name for name in payloads)
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
code = '''import base64,hashlib,json,sys
from pathlib import Path
project=Path(sys.argv[1]);bundle=json.load(sys.stdin);doc=project/bundle['doc'];old=doc.read_bytes()
assert hashlib.sha256(old).hexdigest()==bundle['old_sha256']
evidence=(project/bundle['prefix']).resolve()
expected=Path('/root/autodl-tmp/mcln_published_evidence_20261005/pvground_mask_support_correction_20261008/preflight_start')
assert evidence==expected and not evidence.exists()
for name,encoded in bundle['files'].items():
    assert name.startswith(bundle['prefix']) and not name.endswith(('.pt','.pth','.npz')) and '.aris' not in name
    path=project/name;assert evidence in path.resolve().parents;path.parent.mkdir(parents=True,exist_ok=True)
    raw=base64.b64decode(encoded)
    with path.open('xb') as stream:stream.write(raw)
    assert path.read_bytes()==raw
new=base64.b64decode(bundle['new_doc']);assert new.startswith(old);doc.write_bytes(new)
assert doc.read_bytes()==new
print(json.dumps(dict(files=len(bundle['files']),handoff_sha256=hashlib.sha256(new).hexdigest())))
'''
bundle = dict(doc=doc,prefix=prefix,old_sha256=previous['handoff_sha256'],new_doc=base64.b64encode(new).decode(),
    files={name:base64.b64encode(raw).decode() for name,raw in payloads.items()})
stdin,stdout,stderr = client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python','-B','-c',code,
    '/home/gb/new butd/butd_detr-main/MCLN-main']),timeout=180)
stdin.write(json.dumps(bundle).encode());stdin.flush();stdin.channel.shutdown_write()
raw=stdout.read();assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
remote=json.loads(raw);client.close()
for name,raw in payloads.items():
    for repo in repos[:2]:
        path=repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
for path in copies:path.write_bytes(new)
heads=[]
for index,repo in enumerate(repos):
    stage=[doc]
    if index<2:
        with (repo/'MANIFEST.md').open('a',encoding='utf-8') as stream:
            stream.write('\n- '+stamp+' Mask-support correction implementation and real M0 launch; no new accuracy claim.\n')
        stage+=['MANIFEST.md',*payloads]
    changed=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git','-C',str(repo),'add','-f','--',*stage],stderr=subprocess.DEVNULL)
    subprocess.check_call(['git','-C',str(repo),'-c','core.whitespace=cr-at-eol,-blank-at-eof','diff','--cached','--check','--',*stage])
    if index<2:
        assert all(subprocess.check_output(['git','-C',str(repo),'show',':'+name])==raw for name,raw in payloads.items())
    assert subprocess.check_output(['git','-C',str(repo),'show',':'+doc]).startswith(old)
    subprocess.check_call(['git','-C',str(repo),'commit','--quiet','-m','Implement native Query Mask support correction and start GPU preflight'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'-c','http.version=HTTP/1.1','push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0]==heads[0]
digest=hashlib.sha256(new).hexdigest()
assert digest==remote['handoff_sha256'] and all(path.read_bytes()==new for path in copies)
guard=workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py';raw=guard.read_bytes()
assert raw.count(previous['handoff_sha256'].encode())==1
guard.write_bytes(raw.replace(previous['handoff_sha256'].encode(),digest.encode()))
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.98',heads=heads,
    handoff_sha256=digest,four_local_and_remote_equal=True,github_main=heads[0],payload_count=len(payloads),
    execution_scope='ACTUAL_SOURCE_IMPLEMENTATION_AND_GPU_PREFLIGHT_START',formal_accuracy_result=False,
    retained_best_hits=[5598,4848],overall_goal_complete=False)
(root/'start_publication.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
state_path=root.parent/'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state=json.loads(state_path.read_bytes())
state.update(latest_publication=str(root/'start_publication.json'),handoff_section='20.376.98',handoff_sha256=digest,
    published_heads=heads,status='MASK_SUPPORT_CORRECTION_GPU_PREFLIGHT_RUNNING',owned_gpu_job_active=True,
    mask_support_correction_status='ACTUAL_GPU_PREFLIGHT_STARTED_NOT_PASSED',full_goal_status='ACTIVE_UNMET',
    next_action='Resume original sole M0 observer at its declared estimated endpoint; no early NN query/restart. Inspect actual two-step/native route/state-restore evidence before any formal fit; preserve best5598/4848 and full goal5620/4764+3effective then author-initNr/Sr.')
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record),flush=True)
