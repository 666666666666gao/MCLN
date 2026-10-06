"""Append a cached selected-Query geometry diagnosis to Doc90; no training query."""
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
assert not (root/'cached_error_publication.json').exists()
review=json.loads((root/'PUBLISH_CACHED_ERROR_SOURCE_REVIEW.json').read_bytes())
assert review['execution_scope']=='SOURCE_ONLY' and review['verdict'] in ('PASS','WARN') and not review['blocking_findings']
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
source_review=json.loads((root/'CACHED_REFERENCE_ERROR_REVIEW.json').read_bytes())
assert source_review['execution_scope']=='SOURCE_AND_CACHED_EVIDENCE' and source_review['verdict'] in ('PASS','WARN') and not source_review['blocking_findings']
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
previous_path=root/'fit_publication.json'
previous=json.loads(previous_path.read_bytes());assert previous['section']=='20.376.90'
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
prefix='refine-logs/pvground_reference_keep_20261006/cached_error_analysis/'
summary=json.loads((root/'cached_reference_errors/SUMMARY.json').read_bytes())
assert summary['source_model_hits']==[5598,4848]
assert summary['new_model_evaluations']==summary['new_optimizer_steps']==summary['remote_queries']==0
damage=summary['cohorts']['0.25_damaged']
assert damage['rows']==191 and damage['gt_covered95_and_reference_volume_over4']==128 and damage['mask_good50']==52
assert summary['cohorts']['0.25_repaired']['rows']==174
assert summary['cohorts']['0.5_repaired']['rows']==724 and summary['cohorts']['0.5_damaged']['rows']==371
section=f'''

## 20.376.91 当前Mask参考的Acc@0.25损失：只读缓存范围诊断（{stamp}）

本次仅分析已完成的5598/4848最好模型的9508逐行缓存，没有模型前向、优化器更新、远端训练进度查询或活动代码变更。当前参考保持λ0/1对照仍按上一节运行；首次观察时间仍为{observer['first_observation_cst']}，此前不轮询。用户Scan5620/4764及三个有效模块门槛不变，之后完整模型从作者对应权重分别训练Sr/Nr，单seed2027。

固定该模型已经选中的同一个Query，对比其原生粗框先验与中性融合Mask参考：Acc@0.25修复174、破坏191，净−17；Acc@0.5修复724、破坏371，净+353。由此原粗框先验为5615/4495，参考为5598/4848。这是同一Query的内部几何比较，不能替换历史4511整模型到4848的−18/+337，也不能拼接不同模型成绩。预测评分和所选Query在本次比较中不变。

191条宽松阈值破坏中，128条参考框覆盖至少95%的GT体积，同时参考体积超过GT四倍；63条GT体积覆盖低于95%。这两个组描述实际几何重叠：多数该类损失表现为覆盖目标但范围过大。191条的参考/GT体积比中位数4.8803916，最大单面绝对误差中位数0.503725米；其中52条所选融合Mask自身IoU>0.5，说明即便分割IoU合格，包围范围仍可能过大。所有191条参考有效，没有触发空支撑原框保留规则。

两阈值破坏集合共有118条，另有73条仅宽松破坏和253条仅严格破坏；严格修复724条中655条融合Mask IoU>0.5。上述数值来自同一9508缓存，不是本轮训练成绩。体积、覆盖及Mask重叠不能证明真实物理实例身份，也没有仅凭缓存证明过大范围由哪些远端误分点造成。GT只用于离线描述，不形成推理门控、硬退回原框或候选过滤规则。

这份新证据把下一项边界结构的观察重点放在“已覆盖目标但范围过大”的参考上；仍先等当前参考保持正式对照结束，再决定边界读取改法，不中途叠加采样/注意力/教师/质量排序。不能把128条全部计作可恢复收益。CPU源与独立缓存核验报告随{prefix}发布，原始候选NPZ、权重和其他私有文件不发布。
'''
new=old+section.encode('utf-8');assert new.startswith(old) and new.count(b'## 20.376.91 ')==1
names=['postrun/analyze_cached_reference_errors.py','cached_reference_errors/SUMMARY.json','cached_reference_errors/EXECUTION.json',
    'CACHED_REFERENCE_ERROR_REVIEW.json','CACHED_REFERENCE_ERROR_REVIEW.md',
    'prepare_cached_error_publication.py','publish_cached_error_authorized.py',
    'PUBLISH_CACHED_ERROR_SOURCE_REVIEW.json','PUBLISH_CACHED_ERROR_SOURCE_REVIEW.md']
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
assert str(evidence)=='/root/autodl-tmp/mcln_published_evidence_20261005/pvground_reference_keep_20261006/cached_error_analysis'
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
            stream.write('\n- '+stamp+' Cached same-Query reference geometry diagnosis; no training or model replay.\n')
        stage+=['MANIFEST.md',*payloads]
    changed=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git','-C',str(repo),'add','-f','--',*stage],stderr=subprocess.DEVNULL)
    subprocess.check_call(['git','-C',str(repo),'-c','core.whitespace=cr-at-eol,-blank-at-eof','diff','--cached','--check','--',*stage])
    if index<2:
        assert all(subprocess.check_output(['git','-C',str(repo),'show',':'+name])==raw for name,raw in payloads.items())
    assert subprocess.check_output(['git','-C',str(repo),'show',':'+doc]).startswith(
        subprocess.check_output(['git','-C',str(repo),'show',previous['heads'][index]+':'+doc]))
    subprocess.check_call(['git','-C',str(repo),'commit','--quiet','-m','Record cached Mask-reference overextension diagnosis'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'-c','http.version=HTTP/1.1','push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0]==heads[0]
digest=hashlib.sha256(new).hexdigest();assert digest==remote['handoff_sha256'] and all(p.read_bytes()==new for p in copies)
guard=workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py';raw=guard.read_bytes()
assert raw.count(previous['handoff_sha256'].encode())==1
guard.write_bytes(raw.replace(previous['handoff_sha256'].encode(),digest.encode()))
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.91',heads=heads,
    handoff_sha256=digest,four_local_and_remote_equal=True,github_main=heads[0],payload_count=len(payloads),
    execution_scope='CACHED_REFERENCE_DIAGNOSIS_NO_NEW_MODEL_RESULT',new_formal_result=False,
    controller_pid=launch['controller_pid'],raw_npz_published=False,weights_published=False)
(root/'cached_error_publication.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
state.update(time_cst=record['time_cst'],latest_publication=str(root/'cached_error_publication.json'),
    handoff_section=record['section'],handoff_sha256=digest,published_heads=heads,
    reference_keep_fit_launch=str(root/'fit_launch.json'),reference_keep_fit_controller_pid=launch['controller_pid'],reference_keep_observer_pid=observer['observer_local_pid'],reference_keep_first_observation_cst=observer['first_observation_cst'],
    owned_gpu_job_active=True,overall_goal_complete=False,next_action='Wait until '+observer['first_observation_cst']+' sole formal-fit observer; Scan first, Sr/Nr author-init deferred')
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (workspace/'memory/2026-10-07.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPV-Ground '+record['time_cst']+': Doc91 cached reference geometry diagnosis; original formal controller'+str(launch['controller_pid'])+' unchanged, MAIN'+heads[0]+' ONEdocSHA'+digest+' exactfourlocal+remote+Git; no newformal result. Current5598/4848 protected. Single2027/no multiseed; Sr/Nr author-init AFTERScan5620/4764+threeeffective. Sole observer'+str(observer['observer_local_pid'])+' first'+observer['first_observation_cst']+', noearlyNNquery.\n')
print(json.dumps(record),flush=True)
