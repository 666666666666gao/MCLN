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
assert not (root / 'fit_publication.json').exists()
prior_root = root.parent / 'pvground_mask_support_correction_20261008'
previous = json.loads((root / 'start_publication.json').read_bytes())
assert previous['section'] == '20.376.99'
review = json.loads((root / 'FIT_LAUNCH_SOURCE_REVIEW.json').read_bytes())
assert review['execution_scope'] == 'SOURCE_ONLY'
assert review['verdict'] in ('PASS', 'WARN') and not review['blocking_findings']
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
launch = json.loads((root / 'fit_launch.json').read_bytes())
observer = json.loads((root / 'fit_observer_started.json').read_bytes())
failed = json.loads((prior_root / 'preflight_wait.json').read_bytes())
assert failed['observer_closed'] and failed['exitcode'] == 1
assert not failed['controller_alive']
assert launch['status'] == 'FIT_LAUNCHED_NOT_COMPLETED'
assert launch['controller_pid'] == observer['controller_pid']
assert launch['root'] == '/root/autodl-tmp/pvground_mask_support_correction_20261008_v2'
proof = json.loads((root/'preflight_complete/preflight.json').read_bytes())
closed = json.loads((root/'preflight_wait.json').read_bytes())
assert closed['observer_closed'] and not closed['controller_alive'] and closed['exitcode']==0
assert proof['status']=='pass' and proof['optimizer_steps_per_arm']==2
assert proof['weight_files_created']==0 and proof['parent_and_box_head_frozen']
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

## 20.376.100 修复后两步真实M0通过、完整单seed支撑对照启动（{stamp}）

上一节只记录v2源码修复与启动，现在已有真实运行终态。v2M0 controller872139于03:10:22以exit0结束，总474.38秒；唯一观察器83549在原定03:14首查封存，未提前查询或重启，原始首次import失败继续保留。两组各两步任务训练、零更新Mask/Box与同次父前向完全一致、原criterion的5/1/10/2系数数值一致；第一步输出层获得梯度，第二步内部Query/支撑投影及成员编码获得非零梯度，两个优化器与梯度独立，1304父状态及几何头冻结不变。

两组完整1314状态CPU重建和新头/Adam实际GPU恢复通过；在一次原生PV前向捕获的同帧输入上，集成与paired路径Query Mask、最终Box、语义值最大差均0，原生语义头只调用一次，R零输出不变。没有声称两次独立backbone逐位一致或冷启动GPU父模型重建。M0峰值分配10308729856字节、保留11624513536字节，仅为这次真实batch测量；无新增磁盘权重、无正式精度或有效模块结论。

正式启动器与观察器SOURCE_ONLY复核PASS、blocking0，22实际路径绑定；运行资格来自上述真实M0收据而非静态审查本身。部署前GPU空闲、原M0关闭、保护权重一致，数据盘935514112字节空闲，大于完整初始/终态数组及小头状态预留546015744字节。未为腾空间误删5份必要权重，未安装或重建环境。

完整对照实际于{launch['time_cst']}启动，controller{launch['controller_pid']}，screen `{launch['screen']}`，目录{launch['root']}。初始状态和优化器重新建立，不承接两步预检。固定seed2027、batch8、LR1e-5、WD5e-4、clip0.1；每组29778fit各一次、3723更新，两个独立27841参数头共用每batch一次冻结父前向。control隐藏9维显式粗框相对几何，treatment读取这些量；256候选、同Query Box/Mask、原生bbs和原匹配职责不变。初始及终态完整9508原顺序评估，6887已见训练场景留出另记；不能把零输出重新评估的历史数值差写成新模块增益。

唯一观察器native63704、Windows PID{observer['observer_local_pid']}。根据已关闭上一套完整pair的19266.40秒估计，本轮约19267秒；固定预算无需中途go/no-go，首查定在预计整体终点前300秒：{observer['first_observation_cst']}。此前不查询NN/日志/进度，不创建第二观察器；首查后仍需等待时240秒复查。经过时间不证明完成，预计2倍仍活动则记录超时并保留原任务，不盲目重启。

本节是实际预检通过与正式启动记录，尚未观察本轮完整初始或训练终态成绩。当前保护最好仍5598/4848，目标仍同一ScanRefer至少5620/4764且三项有效贡献，之后固定完整模型从各自作者权重独立训练Sr3D/Nr3D。按用户长期清理授权，当前恢复状态在消费者关闭前保留；完整评估与结果封存后只保留实际指标最好的新增权重及必要依赖，自动清理其余，不重复请求审批。
'''
assert all(line.rstrip() == line for line in section.splitlines())
new = old + section.encode('utf-8')
assert new.count(b'## 20.376.100 ') == 1
spec = json.loads((root / 'pair_spec.json').read_bytes())
prefix = 'refine-logs/pvground_mask_support_correction_20261008_v2/fit_start/'
names = ['FIT_LAUNCH_SOURCE_REVIEW.json','FIT_LAUNCH_SOURCE_REVIEW.md',
    'launch_fit_authorized.py','observe_fit_authorized.py','publish_fit_start_authorized.py',
    'fit_launch.json','fit_observer_started.json','fit_capacity.json','preflight_wait.json']
payloads = {prefix + name: (root / name).read_bytes() for name in names}
payloads[prefix + '.gitattributes'] = b'** -text\n'
for name in ['preflight.json','preflight_status.json','preflight_controller.exit','preflight.exit',
             'preflight.log','preflight_controller.log']:
    payloads[prefix+'actual_preflight/'+name] = (root/'preflight_complete'/name).read_bytes()
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
code = '''import base64,hashlib,json,sys
from pathlib import Path
project=Path(sys.argv[1]);bundle=json.load(sys.stdin);doc=project/bundle['doc'];old=doc.read_bytes()
assert hashlib.sha256(old).hexdigest()==bundle['old_sha256']
evidence=(project/bundle['prefix']).resolve()
assert evidence==Path('/root/autodl-tmp/mcln_published_evidence_20261005/pvground_mask_support_correction_20261008_v2/fit_start')
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
            stream.write('\n- '+stamp+' Publish actual passed M0 and complete Mask-support fit start, no accuracy claim.\n')
        stage+=['MANIFEST.md',*payloads]
    changed=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git','-C',str(repo),'add','-f','--',*stage],stderr=subprocess.DEVNULL)
    lint=[name for name in stage if name not in (doc,)]
    subprocess.check_call(['git','-C',str(repo),'-c','core.whitespace=cr-at-eol,-blank-at-eof','diff','--cached','--check','--',*lint])
    if index<2:
        assert all(subprocess.check_output(['git','-C',str(repo),'show',':'+name])==raw for name,raw in payloads.items())
    assert subprocess.check_output(['git','-C',str(repo),'show',':'+doc])==new
    subprocess.check_call(['git','-C',str(repo),'commit','--quiet','-m','Record passed two-step M0 and start complete paired Mask support training'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'-c','http.version=HTTP/1.1','push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0]==heads[0]
digest=hashlib.sha256(new).hexdigest()
assert digest==remote['handoff_sha256'] and all(path.read_bytes()==new for path in copies)
guard=workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py';raw=guard.read_bytes()
assert raw.count(previous['handoff_sha256'].encode())==1
guard.write_bytes(raw.replace(previous['handoff_sha256'].encode(),digest.encode()))
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.100',heads=heads,
    handoff_sha256=digest,four_local_and_remote_equal=True,github_main=heads[0],payload_count=len(payloads),
    execution_scope='ACTUAL_M0_PASS_AND_COMPLETE_PAIRED_FIT_START',formal_accuracy_result=False,
    prior_m0_exitcode=1,prior_m0_optimizer_steps=0,v2_preflight_pass=True,fit_terminal_unobserved=True,
    retained_best_hits=[5598,4848],overall_goal_complete=False)
(root/'fit_publication.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
state_path=root.parent/'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state=json.loads(state_path.read_bytes())
state.update(latest_publication=str(root/'fit_publication.json'),handoff_section='20.376.100',handoff_sha256=digest,
    published_heads=heads,status='MASK_SUPPORT_COMPLETE_PAIRED_FIT_RUNNING',owned_gpu_job_active=True,
    mask_support_correction_root=str(root),mask_support_correction_status='ACTUAL_M0_PASS_COMPLETE_FIT_STARTED_NO_NEW_RESULT',
    mask_support_correction_controller_pid=launch['controller_pid'],
    mask_support_correction_fit_observer_local_pid=observer['observer_local_pid'],
    mask_support_correction_fit_observer_native_session=63704,
    mask_support_correction_fit_first_observation_cst=observer['first_observation_cst'],
    mask_support_correction_observer_closed=True,mask_support_correction_fit_observer_closed=False,full_goal_status='ACTIVE_UNMET',
    next_action='Wait for the SAME63704 sole observer at declared first08:32:38.017046CST near estimated whole-job finish, then240s only if needed. No early remote NN/log polls or duplicate observer/restart. Consume actual closed-task receipt and collect original full results/arrays before best selection and authorized nonbest cleanup. Preserve best5598/4848 and Scan5620/4764+3effective then author-initNrSr.')
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record),flush=True)
