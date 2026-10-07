"""Publish actual import failure and immutable v2 M0 start, with no result claim."""
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
prior_root = root.parent / 'pvground_mask_support_correction_20261008'
previous = json.loads((prior_root / 'start_publication.json').read_bytes())
assert previous['section'] == '20.376.98'
review = json.loads((root / 'SOURCE_REVIEW.json').read_bytes())
assert review['execution_scope'] == 'SOURCE_ONLY'
assert review['verdict'] in ('PASS', 'WARN') and not review['blocking_findings']
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
launch = json.loads((root / 'preflight_launch.json').read_bytes())
observer = json.loads((root / 'observer_wait.json').read_bytes())
failed = json.loads((prior_root / 'preflight_wait.json').read_bytes())
assert failed['observer_closed'] and failed['exitcode'] == 1
assert not failed['controller_alive']
assert launch['status'] == 'PREFLIGHT_LAUNCHED_NOT_PASSED'
assert launch['controller_pid'] == observer['controller_pid']
assert launch['root'] == '/root/autodl-tmp/pvground_mask_support_correction_20261008_v2'
assert not (root / 'fit_launch.json').exists()
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
    assert subprocess.check_output(['git','-C',str(repo),'show','HEAD:'+doc]) == old
stamp = datetime.datetime.now().astimezone().isoformat()
section = f'''

## 20.376.99 首次M0导入失败封存、原生函数修复与v2真实预检启动（{stamp}）

上一节记录的是首次源码实现与启动，不代表实际预检通过。原controller870103在2026-10-08T02:23:14.613804+08:00以exit1结束，模块导入在新目标文件的`from torch_scatter import scatter_mean`处报ModuleNotFoundError。它尚未执行模型构建、模型前向或优化器更新；入口此前已经import Torch并重置CUDA峰值，不能写成完全没有调用Torch。原唯一观察器57442按原定02:34:53首查封存失败，未因超时重启。首次SOURCE_ONLY审查漏掉了这一实际依赖问题，原FAIL、补充PASS和真实失败日志均保留，静态PASS不能代替运行验证。

v2唯一神经源码修复为`from models.losses import dice_loss, sigmoid_focal_loss, scatter_mean`。实际ported criterion已从现有`utils.scatter_util`导出此函数，复用相同实现，无安装、环境重建、fallback、loss主体或标签变化。8个新runner中仅这一import变化，其余7份与PV overlay逐字节相同；98文件port、17已有helper及warm环境绑定一致。新隔离目录为{launch['root']}，旧目录不覆盖。独立SOURCE_ONLY复核{review['verdict']}、blocking0，62路径绑定/13Python及2嵌入源码检查；同上下文、同系临时接受，实际后台model/effort未认证。formal launcher明确排除，尚不能启动正式训练。

真实v2M0已于{launch['time_cst']}启动，controller{launch['controller_pid']}，screen `{launch['screen']}`。唯一Windows观察器PID{observer['observer_local_pid']}，原定首查{observer['first_observation_cst']}，之后仅需要时240秒复查；不在此前查询NN/日志/进度。预计M0约900秒，首查在预计终点前3分钟。预检应检查两步真实任务梯度、1304冻结状态、1314模型CPU重建、GPU新头/Adam恢复、同一次原生前向缓存下Mask/Box严格一致，以及峰值显存。源码通过不是这些检查已通过；本节没有新9508成绩、训练终点或有效贡献结论，M0不保存新权重。

两组仍是同一冻结PV/G父模型每batch一次前向、独立相同初始化27841参数Mask残差头。control仅隐藏9维显式粗框相对几何，treatment读取这些量；256候选、原Text/Query融合、单一bbs和同Query的Box/Mask不变。原生匹配决定每个候选真实GT，仅训练5Query focal+1Query Dice+10fused focal+2fused Dice，按原有效GT数归一化。精确成员范围仍硬阈值与detach，不能声称Box损失通过它端到端更新Mask。实际M0与容量条件通过后，才从新初始状态按batch8、seed2027、29778fit一次/3723更新每组执行完整初始与终态验证，不承接两步预检优化器。

上一节发布器曾在继承的PV源码尾随空格检查处关闭；原NN源码与旧实验不修改，已有25份payload和四份/远端文档只完成剩余Git同步，未重写远端或重新启动NN。恢复过程中确认旧Git文档blob曾按core.autocrlf转换，与四份本地/远端字节差异仅CRLF。仅对该文档增加`-text`属性并重新规范化index，保留原文，最终Git/local/Desktop/server文档相同；原发布失败与两次恢复失败记录均保留。Doc98正式提交为{previous['heads'][0]}，不能把这项字节修复写成模型改动。

用户再次明确长期授权：清理我们产生的已结束非最佳权重及核验本地归档后的冗余远端数组，不再逐项审批。此前191NPZ共126198802字节已清理一次，最新盘点仅5份必需权重（Scan/Nr/Sr作者权重、原G、当前最佳delta），未删除必要依赖或重复清理。活动恢复、数据、文本结果与证据继续保留。

当前最佳仍5598/4848；完整目标仍同一ScanRefer模型至少5620/4764、三项真实有效贡献，之后固定完整模型从各自作者权重独立训练Sr3D/Nr3D。单seed2027、不恢复V99双源；这次预检启动不改变目标完成状态。
'''
assert all(line.rstrip() == line for line in section.splitlines())
new = old + section.encode('utf-8')
assert new.count(b'## 20.376.99 ') == 1
spec = json.loads((root / 'pair_spec.json').read_bytes())
prefix = 'refine-logs/pvground_mask_support_correction_20261008_v2/preflight_start/'
names = list(spec['new_runner_files']) + ['controller.py','pair_spec.json','source_port.json',
    'EXPERIMENT_PLAN.md','SOURCE_REVIEW.json','SOURCE_REVIEW.md','INITIAL_IMPORT_FAILURE.json',
    'deploy_preflight_authorized.py','observe_preflight_authorized.py','publish_start_authorized.py',
    'resource_check.json','preflight_launch.json','observer_wait.json',
    'runtime_overlay/PV-Ground/models/pv_ground.py']
payloads = {prefix + name: (root / name).read_bytes() for name in names}
payloads[prefix + '.gitattributes'] = b'** -text\n'
for name in ['preflight.log','preflight_status.json']:
    payloads[prefix+'previous_m0/'+name] = (prior_root/'preflight_complete'/name).read_bytes()
payloads[prefix+'previous_m0/preflight_wait.json'] = (prior_root/'preflight_wait.json').read_bytes()
payloads[prefix+'previous_m0/publication_resume.py'] = (prior_root/'resume_start_publication_authorized.py').read_bytes()
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
code = '''import base64,hashlib,json,sys
from pathlib import Path
project=Path(sys.argv[1]);bundle=json.load(sys.stdin);doc=project/bundle['doc'];old=doc.read_bytes()
assert hashlib.sha256(old).hexdigest()==bundle['old_sha256']
evidence=(project/bundle['prefix']).resolve()
assert evidence==Path('/root/autodl-tmp/mcln_published_evidence_20261005/pvground_mask_support_correction_20261008_v2/preflight_start')
assert not evidence.exists()
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
            stream.write('\n- '+stamp+' Preserve failed M0 and publish native-import repair/v2 preflight start, no accuracy claim.\n')
        stage+=['MANIFEST.md',*payloads]
    changed=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git','-C',str(repo),'add','-f','--',*stage],stderr=subprocess.DEVNULL)
    lint=[name for name in stage if name not in (doc,prefix+'runtime_overlay/PV-Ground/models/pv_ground.py')]
    subprocess.check_call(['git','-C',str(repo),'-c','core.whitespace=cr-at-eol,-blank-at-eof','diff','--cached','--check','--',*lint])
    if index<2:
        assert all(subprocess.check_output(['git','-C',str(repo),'show',':'+name])==raw for name,raw in payloads.items())
    assert subprocess.check_output(['git','-C',str(repo),'show',':'+doc])==new
    subprocess.check_call(['git','-C',str(repo),'commit','--quiet','-m','Record failed import and start isolated native Mask-support preflight v2'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'-c','http.version=HTTP/1.1','push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0]==heads[0]
digest=hashlib.sha256(new).hexdigest()
assert digest==remote['handoff_sha256'] and all(path.read_bytes()==new for path in copies)
guard=workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py';raw=guard.read_bytes()
assert raw.count(previous['handoff_sha256'].encode())==1
guard.write_bytes(raw.replace(previous['handoff_sha256'].encode(),digest.encode()))
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.99',heads=heads,
    handoff_sha256=digest,four_local_and_remote_equal=True,github_main=heads[0],payload_count=len(payloads),
    execution_scope='ACTUAL_FAILED_M0_AND_NATIVE_IMPORT_REPAIR_V2_START',formal_accuracy_result=False,
    prior_m0_exitcode=1,prior_m0_optimizer_steps=0,v2_pass_unobserved=True,
    retained_best_hits=[5598,4848],overall_goal_complete=False)
(root/'start_publication.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
state_path=root.parent/'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state=json.loads(state_path.read_bytes())
state.update(latest_publication=str(root/'start_publication.json'),handoff_section='20.376.99',handoff_sha256=digest,
    published_heads=heads,status='MASK_SUPPORT_M0_V2_RUNNING',owned_gpu_job_active=True,
    mask_support_correction_root=str(root),mask_support_correction_status='IMPORT_REPAIR_V2_ACTUAL_PREFLIGHT_STARTED',
    mask_support_correction_controller_pid=launch['controller_pid'],
    mask_support_correction_observer_local_pid=observer['observer_local_pid'],
    mask_support_correction_observer_native_session=83549,
    mask_support_correction_first_observation_cst=observer['first_observation_cst'],
    mask_support_correction_observer_closed=False,full_goal_status='ACTIVE_UNMET',
    next_action='Wait for the SAME83549 sole observer at declared first03:14:21.415090CST, then240s only if needed. No early remote NN/log polls or duplicate observer/restart. ActualM0 acceptance and separately reviewed formal launcher required before fresh complete paired fit. Preserve best5598/4848 and Scan5620/4764+3effective then author-initNrSr.')
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record),flush=True)
