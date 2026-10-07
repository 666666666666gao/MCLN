"""Append actual closed reference-keep comparison to Doc91 after fresh audit and best-only retention."""
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
assert not (root/'terminal_publication.json').exists()
review=json.loads((root/'PUBLISH_TERMINAL_SOURCE_REVIEW.json').read_bytes())
assert review['execution_scope']=='SOURCE_ONLY' and review['verdict'] in ('PASS','WARN') and not review['blocking_findings']
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
audit_path=root/'analysis/EXPERIMENT_AUDIT.json'
audit=json.loads(audit_path.read_bytes())
call=json.loads((root/'analysis/TERMINAL_REVIEW_CALL.json').read_bytes())
assert call['result_received'] and audit['fresh_context']
assert audit['execution_scope']=='TERMINAL_SOURCE_AND_ACTUAL_ARTIFACTS'
assert audit['verdict'] in ('PASS','WARN') and not audit['blocking_findings']
for name in ('analysis/SUMMARY.json','closed_weight_inspection.json'):
    assert audit['actual_file_digests'][name]['sha256']==hashlib.sha256((root/name).read_bytes()).hexdigest()
wait=json.loads((root/'fit_wait.json').read_bytes())
assert wait['observer_closed'] and not wait['terminal']['controller_alive'] and wait['terminal']['exitcode']==0
retention=json.loads((root/'weight_retention.json').read_bytes())
assert retention['status']=='CLOSED_NONBEST_WEIGHTS_REMOVED' and retention['raw_evidence_deleted'] is False
assert retention['fresh_audit_sha256']==hashlib.sha256(audit_path.read_bytes()).hexdigest()
assert retention['retained_best']['candidate']=='protected_geometry_parent' and retention['retained_best']['hits']==[5598,4848]
launch=json.loads((root/'fit_launch.json').read_bytes())
assert launch['accuracy_result'] is False and launch['updates_per_arm']==3723
assert launch['protected_best_hits']==[5598,4848] and launch['candidate_gate_hits']==[5620,4764]
assert launch['root']=='/root/autodl-tmp/pvground_reference_keep_20261006'
m0=json.loads((root/'preflight_wait.json').read_bytes())
assert m0['observer_closed'] and m0['exitcode']==0 and m0['status']['status']=='complete'
observer=json.loads((root/'fit_observer_started.json').read_bytes())
assert observer['first_check_seconds']==24300 and observer['poll_seconds']==240
previous_path=root/'cached_error_publication.json'
previous=json.loads(previous_path.read_bytes());assert previous['section']=='20.376.91'
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
prefix='refine-logs/pvground_reference_keep_20261006/completed_comparison/'
summary=json.loads((root/'analysis/SUMMARY.json').read_bytes())
assert summary['status']=='ACTUAL_CLOSED_ALL256_REFERENCE_KEEP_PAIR_RECOUNTED'
assert summary['metric_best_candidate']['arm']=='protected_geometry_parent'
assert [(row['rec_hits25'],row['rec_hits50']) for row in summary['table']]==[(5598,4848),(5598,4848),(5593,4832),(5598,4848),(5593,4832)]
assert summary['arms_compared']['formal']['0.25']['net']==summary['arms_compared']['formal']['0.5']['net']==0
section=f'''

## 20.376.92 参考保持对照实际终态：无新增定位收益，保留5598/4848（{stamp}）

本轮完整对照实际于{wait['terminal']['status']['finished_cst']}结束，退出码0；原观察器首次07:16查看，此后240秒间隔，07:36确认闭合。两组各29778条fit输入一次、3723次更新，物理/有效batch8、累积1、LR1e-5、WD5e-4、clip0.1、seed2027；父模型/Mask/语言/语义及零输出R冻结，只训练原456102参数、10状态的六面分布几何头，优化器重新初始化。本轮未加教师、排序、文本注意力或推理GT。

| 同一模型/阶段，完整9508开发验证，原生last/bbs | Acc@0.25命中 | Acc@0.5命中 |
|---|---:|---:|
| 保护的融合Mask参考，新增输出零更新 | 5598（58.8767%） | 4848（50.9886%） |
| 本轮control初始 | 5598 | 4848 |
| control终点，参考保持权重0 | 5593（58.8241%） | 4832（50.8204%） |
| 本轮keep初始 | 5598 | 4848 |
| keep终点，参考保持权重1 | 5593（58.8241%） | 4832（50.8204%） |

keep与control的两阈值命中、修复/破坏集合均相同。两组各自相对初始：宽松修复4/破坏9，严格修复13/破坏29；训练后仍低于保护起点。keep loss实际参与训练，整轮未加权原始和为{summary['reference_keep_training']['keep']['loss_sum']:.12f}，control仅记录而权重0的对应和为{summary['reference_keep_training']['control']['loss_sum']:.12f}；这不能证明本配置有性能收益，也不等于从未构建或运行该路径。此次结论为“该参考保持实现/权重/预算未带来净增量”，不再以未训练解释，也不继续无界延长或把它列为已有效第三贡献。

完整归档4823文件、583437053字节，0权重复制；CPU与新鲜独立审查重算4756 NPZ中全部256候选的prior/reference/final框，共29208576个IoU，已记录的所选Query及Full-256上界阈值判断均无翻转。数据GT只用于训练资格/评估。四次本轮formal的native scores和original_prior数组精确相同、所选Query相同；所选reference框也精确相同。全256 reference仍有跨forward浮点/validity差异，stored Mask IoU有少量行漂移；与历史保护参考有3条Query选择变化但命中集合相同。不能把冻结状态说成所有跨forward输出逐位相同，也不把漂移归因于keep。实际审计{audit['verdict']}、same-family/provisional，身份未认证；具体范围以随附审计报告为准，CPU缓存复算未重做原始Mask前向。

当前网络仍为PV-Ground＋原G六源观测/语义几何读取→Text/Query Mask与融合权重→预测前景真实成员极值构成空间参考→原7位置×16成员及109维完整范围输入→六面33位置分布→唯一最终框；原生bbs选同一Query的框/Mask。保护最好版本的新增分布输出仍中性零更新，最终框等于Mask参考，不能写成学习式精修或蒸馏带来337条增量。原G和历史训练预算仍计入方法历史。

固定ScanRefer新准入为同一个完整模型Acc@0.25>59.1%且Acc@0.5>50.1%，即至少5620/4764命中；当前严格已满足，宽松还差22条。并需三个真实有效贡献及直接消融，才冻结完整方法并分别训练Sr3D/Nr3D。允许使用作者各自对应权重，baseline/完整方法同对应核心起点，仍为单seed2027、独立训练，不把ScanRefer权重直测视为Nr/Sr训练结果。

下一项优先改变六面实际边界观测，保留有效参考/原生评分/全部256候选/现有分布表示，不恢复失败普通回写/质量排序，也不把128条过大参考都算作可恢复收益。为隔离已实测的跨forward Mask漂移，拟在对照中共享一次冻结父模型前向，再由独立几何头/优化器读取同一预测支撑；这是待实现的实验协议，不是已发布新精度。保持原参数量与训练规则，先真实两步预检，再同预算center邻域与face范围观测对照。没有源码审查与预检通过前不启动完整fit。

按用户best-only要求，在实际CPU检查、审计与固定路径核验后，删除本轮4份非最佳权重，共{retention['released_bytes']}字节；仍保留原5598/4848最好状态及官方PV/原G/V99所需链。原始NPZ/日志/数据集没有删除，额外候选数组清理须另列精确范围与许可。源码、实际结果/审计/恢复/清理记录随{prefix}发布，不上传权重/NPZ或私有审查trace。
'''
new=old+section.encode('utf-8');assert new.startswith(old) and new.count(b'## 20.376.92 ')==1
names=['postrun/analyze_reference_keep_formal.py','analysis/SUMMARY.json','analysis/EXPERIMENT_AUDIT.json','analysis/EXPERIMENT_AUDIT.md',
    'analysis/RESULT_TO_CLAIM.json','analysis/RESULT_TO_CLAIM.md','closed_weight_inspection.json','weight_retention.json',
    'postrun/inspect_closed_weights.py','postrun/inspect_closed_weights_authorized.py','postrun/INSPECTION_SOURCE_REVIEW.json',
    'postrun/retain_closed_best.py','postrun/retain_closed_best_authorized.py','postrun/RETENTION_SOURCE_REVIEW.json',
    'postrun/ANALYSIS_SOURCE_REVIEW.json','prepare_terminal_publication.py','publish_terminal_authorized.py',
    'PUBLISH_TERMINAL_SOURCE_REVIEW.json','PUBLISH_TERMINAL_SOURCE_REVIEW.md','research_contract.md',
    'complete_fit/fit_status.json','complete_fit/control_spec.json','complete_fit/keep_spec.json',
    'complete_fit/control/receipt.json','complete_fit/keep/receipt.json',
    'complete_fit/control/formal/receipt.json','complete_fit/keep/formal/receipt.json']
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
assert str(evidence)=='/root/autodl-tmp/mcln_published_evidence_20261005/pvground_reference_keep_20261006/completed_comparison'
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
            stream.write('\n- '+stamp+' Closed reference-keep comparison: no metric gain; original best retained and closed nonbest weights cleared.\n')
        stage+=['MANIFEST.md',*payloads]
    changed=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git','-C',str(repo),'add','-f','--',*stage],stderr=subprocess.DEVNULL)
    subprocess.check_call(['git','-C',str(repo),'-c','core.whitespace=cr-at-eol,-blank-at-eof','diff','--cached','--check','--',*stage])
    if index<2:
        assert all(subprocess.check_output(['git','-C',str(repo),'show',':'+name])==raw for name,raw in payloads.items())
    assert subprocess.check_output(['git','-C',str(repo),'show',':'+doc]).startswith(
        subprocess.check_output(['git','-C',str(repo),'show',previous['heads'][index]+':'+doc]))
    subprocess.check_call(['git','-C',str(repo),'commit','--quiet','-m','Record completed reference-preservation comparison'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'-c','http.version=HTTP/1.1','push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0]==heads[0]
digest=hashlib.sha256(new).hexdigest();assert digest==remote['handoff_sha256'] and all(p.read_bytes()==new for p in copies)
guard=workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py';raw=guard.read_bytes()
assert raw.count(previous['handoff_sha256'].encode())==1
guard.write_bytes(raw.replace(previous['handoff_sha256'].encode(),digest.encode()))
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.92',heads=heads,
    handoff_sha256=digest,four_local_and_remote_equal=True,github_main=heads[0],payload_count=len(payloads),
    execution_scope='REFERENCE_KEEP_ACTUAL_COMPLETED_AUDITED_RETAINED',new_formal_result=True,
    controller_pid=launch['controller_pid'],raw_npz_published=False,weights_published=False)
(root/'terminal_publication.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
state.update(time_cst=record['time_cst'],latest_publication=str(root/'terminal_publication.json'),
    handoff_section=record['section'],handoff_sha256=digest,published_heads=heads,
    reference_keep_fit_launch=str(root/'fit_launch.json'),reference_keep_fit_controller_pid=launch['controller_pid'],reference_keep_observer_pid=observer['observer_local_pid'],reference_keep_first_observation_cst=observer['first_observation_cst'],
    owned_gpu_job_active=False,overall_goal_complete=False,next_action='Reference-keep closed with no metric gain; prepare same-forward center/face boundary observation contrast, SOURCE review and actual M0 first. Scan joint5620/4764 and three effective contributions before corresponding-author Sr/Nr full training.')
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (workspace/'memory/2026-10-07.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPV-Ground '+record['time_cst']+': Doc92 actual closed pair control/keep both5593/4832, protected5598/4848 retained; no reference-keep gain. Source/actual audit/CPU restore/4 nonbest cleanup published. MAIN'+heads[0]+' ONEdocSHA'+digest+' fourlocal+remote+Git exact. Target5620/4764+3 real contributions unmet; corresponding-author Nr/Sr full training deferred. No new NN launched.\n')
print(json.dumps(record),flush=True)
