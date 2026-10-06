"""Append actual M0 closure and formal ScanRefer reference-keep launch to Doc89."""
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
assert not (root/'fit_publication.json').exists()
review=json.loads((root/'PUBLISH_FIT_SOURCE_REVIEW.json').read_bytes())
assert review['execution_scope']=='SOURCE_ONLY' and review['verdict'] in ('PASS','WARN') and not review['blocking_findings']
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
source_review=json.loads((root/'FIT_SOURCE_REVIEW.json').read_bytes())
assert source_review['execution_scope']=='SOURCE_ONLY' and source_review['verdict'] in ('PASS','WARN') and not source_review['blocking_findings']
for item in source_review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
launch=json.loads((root/'fit_launch.json').read_bytes())
assert launch['accuracy_result'] is False and launch['updates_per_arm']==3723
assert launch['protected_best_hits']==[5598,4848] and launch['candidate_gate_hits']==[5620,4764]
assert launch['root']=='/root/autodl-tmp/pvground_reference_keep_20261006'
m0=json.loads((root/'preflight_wait.json').read_bytes())
assert m0['observer_closed'] and m0['exitcode']==0 and m0['status']['status']=='complete'
observer=json.loads((root/'fit_observer_started.json').read_bytes())
assert observer['first_check_seconds']==24300 and observer['poll_seconds']==240
previous_path=root/'preflight_publication.json'
previous=json.loads(previous_path.read_bytes());assert previous['section']=='20.376.89'
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
stamp=datetime.datetime.now().astimezone().isoformat()
prefix='refine-logs/pvground_reference_keep_20261006/fit_registration/'
section=f'''

## 20.376.90 参考保持约束：真实两步预检通过，正式同预算对照启动（{stamp}）

用户目标保持：同一完整模型9508条ScanRefer Acc@0.25严格>59.1%、Acc@0.5严格>50.1%，至少5620/4764命中；三个模块分别有直接消融和有效贡献之后，固定完整方法，Sr3D/Nr3D分别从作者对应预训练权重公平初始化并独立训练。固定seed2027，不做多seed。当前受保护最好模型5598/4848；这次工程预检不是新的精度结果，也不是第三项贡献已经成立。

上一节M0启动记录已获得真实终态：控制器776037于{m0['status']['finished_cst']}结束，exit0，两组各执行两次优化器更新，没有产生预检权重文件。两组仅更新456102参数/10状态的已有几何头，父模型、全零R、原生bbs和Mask保持；中性初始解码严格等于预测Mask参考，保持项初始严格为0。输出层更新后，新约束有非零直接输出梯度，且只作用可信对应集合；几何头及Adam的键、组、步数和moments实际内存CPU保存恢复一致。39条真实空支撑样本保持既有原框规则，全部256候选的原始点成员极值见证通过。

控制λ=0，策略λ=1，均使用同一融合Mask参考、同一保留initial.pth重建和fresh optimizer；不承接M0两步状态。训练内参考退化项为表达内均值再实际batch均值的squared ReLU(stopgrad(IoU(reference,GT))-IoU(final,GT))。原匹配候选对应自己的过滤后真实GT；额外候选要求自身Query及融合Mask均得到root支撑确认，排除全部原匹配，没有Box>0.5截断。GT资格不进入推理；原native/G/匹配DFL及已有额外定位目标不变，一套last/bbs、同Query框与Mask、256候选保留。两组跨进程M0有极小浮点差异，应将后续称为同起点同预算经验对照，不宣称逐位配对。

正式控制器{launch['controller_pid']}实际启动于{launch['time_cst']}，screen {launch['screen']}，共享GPU锁串行运行。每组fit29778条一次、3723更新、batch8/累积1、lr1e-5/WD5e-4/clip0.1；初始及终点各评估6887模块留出和9508正式开发验证。6887场景已被作者预训练见过，不作为正式泛化证据。当前仅启动，没有发布新命中数。保留隐藏状态此前11169次更新，终点14892；本轮重置输出层终点3723，原G历史另计。

实际启动资源：数据盘空闲{launch['resources']['data_free_bytes']}字节，系统盘{launch['resources']['system_free_bytes']}字节，保存预留{launch['resources']['required_reserve_bytes']}字节；官方PV、原G和当前4848权重保持。沿用已有A100/Torch1.10.2环境，不安装新包。上一同预算完整pair耗时24792秒，当前估计约7小时；唯一观察器PID {observer['observer_local_pid']}，首次远端查看{observer['first_observation_cst']}，此前不轮询，临近结束按240秒复查。旧M0观察器32309及Doc89发布80406已消费exit0，不能重复运行。

本轮需同时超过同预算控制与自身4848起点，并核对5620/4764双门槛；保留修复/破坏和完整候选覆盖。若只追回控制退化，不认定新增模块有效。实验闭合后按已授权范围清理自身非最佳权重，保留日志、指标、必要依赖与最好权重；不自动扩大原始数组或数据集删除权限。Nr/Sr正式训练继续等待ScanRefer与三模块证据，不因这次M0通过而提前启动。
'''
new=old+section.encode('utf-8');assert new.startswith(old) and new.count(b'## 20.376.90 ')==1
names=['EXPERIMENT_PLAN.md','EXPERIMENT_TRACKER.md','research_contract.md','control_spec.json','keep_spec.json',
    'reference_keep.py','run_reference_keep_fit.py','selected_mask_reference_factory.py','controller.py',
    'preflight_wait.json','preflight_complete/control/preflight.json','preflight_complete/keep/preflight.json',
    'FIT_SOURCE_REVIEW.json','FIT_SOURCE_REVIEW.md','fit_launch.json','fit_resource_check.json','fit_observer_started.json',
    'prepare_fit_launch.py','launch_fit_authorized.py','observe_fit_authorized.py','collect_closed_fit_authorized.py',
    'prepare_fit_publication.py','publish_fit_authorized.py','PUBLISH_FIT_SOURCE_REVIEW.json','PUBLISH_FIT_SOURCE_REVIEW.md']
payloads={prefix+name:(root/name).read_bytes() for name in names};payloads[prefix+'.gitattributes']=b'** -text\n'
assert all(not name.endswith(('.pt','.pth','.npz')) and '.aris' not in name for name in payloads)
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
spec=json.loads((root/'control_spec.json').read_bytes())
code='''import base64,hashlib,json,sys
from pathlib import Path
project=Path(sys.argv[1]);b=json.load(sys.stdin);doc=project/b['doc'];old=doc.read_bytes()
assert hashlib.sha256(old).hexdigest()==b['old_sha256']
evidence=(project/b['prefix']).resolve()
assert str(evidence)=='/root/autodl-tmp/mcln_published_evidence_20261005/pvground_reference_keep_20261006/fit_registration'
assert not evidence.exists()
for name,encoded in b['files'].items():
    assert name.startswith(b['prefix']) and not name.endswith(('.pt','.pth','.npz')) and '.aris' not in name
    p=project/name;assert evidence in p.resolve().parents;p.parent.mkdir(parents=True,exist_ok=True)
    raw=base64.b64decode(encoded)
    with p.open('xb') as f:f.write(raw)
    assert p.read_bytes()==raw
new=base64.b64decode(b['new_doc']);assert new.startswith(old);doc.write_bytes(new)
assert doc.read_bytes()==new
print(json.dumps(dict(files=len(b['files']),handoff_sha256=hashlib.sha256(new).hexdigest())))
'''
bundle=dict(doc=doc,prefix=prefix,old_sha256=previous['handoff_sha256'],new_doc=base64.b64encode(new).decode(),
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
heads=[]
for index,repo in enumerate(repos):
    stage=[doc]
    if index<2:
        with (repo/'MANIFEST.md').open('a',encoding='utf-8') as stream:
            stream.write('\n- '+stamp+' Reference-preservation real M0 PASS and same-budget formal fit launched; no new formal metrics.\n')
        stage+=['MANIFEST.md',*payloads]
    changed=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git','-C',str(repo),'add','-f','--',*stage],stderr=subprocess.DEVNULL)
    subprocess.check_call(['git','-C',str(repo),'-c','core.whitespace=cr-at-eol,-blank-at-eof','diff','--cached','--check','--',*stage])
    if index<2:
        assert all(subprocess.check_output(['git','-C',str(repo),'show',':'+name])==raw for name,raw in payloads.items())
    assert subprocess.check_output(['git','-C',str(repo),'show',':'+doc]).startswith(
        subprocess.check_output(['git','-C',str(repo),'show',previous['heads'][index]+':'+doc]))
    subprocess.check_call(['git','-C',str(repo),'commit','--quiet','-m','Record reference-preservation M0 PASS and formal ScanRefer contrast launch'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'-c','http.version=HTTP/1.1','push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0]==heads[0]
digest=hashlib.sha256(new).hexdigest();assert digest==remote['handoff_sha256'] and all(p.read_bytes()==new for p in copies)
guard=workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py';raw=guard.read_bytes()
assert raw.count(previous['handoff_sha256'].encode())==1
guard.write_bytes(raw.replace(previous['handoff_sha256'].encode(),digest.encode()))
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.90',heads=heads,
    handoff_sha256=digest,four_local_and_remote_equal=True,github_main=heads[0],payload_count=len(payloads),
    execution_scope='REFERENCE_KEEP_M0_PASSED_FORMAL_FIT_LAUNCHED_NOT_COMPLETE',new_formal_result=False,
    controller_pid=launch['controller_pid'],raw_npz_published=False,weights_published=False)
(root/'fit_publication.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
state.update(time_cst=record['time_cst'],latest_publication=str(root/'fit_publication.json'),
    handoff_section=record['section'],handoff_sha256=digest,published_heads=heads,
    reference_keep_fit_launch=str(root/'fit_launch.json'),reference_keep_fit_controller_pid=launch['controller_pid'],reference_keep_observer_pid=observer['observer_local_pid'],reference_keep_first_observation_cst=observer['first_observation_cst'],
    owned_gpu_job_active=True,overall_goal_complete=False,next_action='Wait until '+observer['first_observation_cst']+' sole formal-fit observer; Scan first, Sr/Nr author-init deferred')
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (workspace/'memory/2026-10-07.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPV-Ground '+record['time_cst']+': Doc90 actualM0bothPASS and formal controller'+str(launch['controller_pid'])+' launched, MAIN'+heads[0]+' ONEdocSHA'+digest+' exactfourlocal+remote+Git; no newformal result. Current5598/4848 protected. Single2027/no multiseed; Sr/Nr author-init AFTERScan5620/4764+threeeffective. Sole observer'+str(observer['observer_local_pid'])+' first'+observer['first_observation_cst']+', noearlyNNquery.\n')
print(json.dumps(record),flush=True)
