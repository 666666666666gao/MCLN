"""Publish the one completed observation and its next passive owner."""
import ast
import datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
data = root/'normal_controls_20261010'
summary = json.loads((data/'NORMAL_FIRST_OBSERVATION_SUMMARY.json').read_bytes())
plan = json.loads((data/'NORMAL_NEXT_OBSERVATION_PLAN.json').read_bytes())
owner = json.loads((data/'NORMAL_EPOCH2_AUTHORIZED_OBSERVER_OWNER.json').read_bytes())
resource = json.loads((data/'EPOCH1_RESOURCE_ATTEMPT2_RECEIPT.json').read_bytes())
assert summary['epoch1_delta_vs_own_E0']==[-33,-316] and summary['formal_results_unaudited']
assert owner['local_pid']==51540 and owner['first_due_cst']==plan['due_cst']
assert all(row['alive'] for row in resource['processes'])
assert json.loads((data/'EPOCH1_RESOURCE_ATTEMPT2_EXIT.json').read_bytes())['exit_code']==0
notes = '''## §20.376.145 — C-off正常训练第1轮观察，仍未保住起点

本节记录原定首查在2026年10月11日01:03:12取得的快照，不是训练终态。原观察会话21631成功完成并消费，没有在01:03之前读取训练状态。

| 同一9508条ScanRefer，原生last/bbs | @0.25命中／百分比 | @0.50命中／百分比 |
|---|---:|---:|
| 共同E0，保留最好 | 5677／59.7076% | 4920／51.7459% |
| 历史C-on正常训练E1 | 5652／59.4447% | 4596／48.3382% |
| 当前C-off正常训练E1 | 5644／59.3605% | 4604／48.4224% |

C-off E1相对自身E0为−33／−316；相对历史C-on E1为−8／+8。此次起点完整验证计数一致，原始native_metrics、输出日志与取回摘要字节及SHA均核对。这里只是单seed早期观察，终态精度审查未完成；8条严格差值不足以证明C是退化主因，更不能写成额外监督在所有训练下无效。两组父状态已有此前C适配历史，本对照关闭本轮C，不能称为从未训练C的消融。

01:03快照中控制器185395与原生训练子进程185396均存活；日志已进入E2，完成269／4583次更新，用时817秒。E1完整1189批验证用时1446秒。按该实测速率估计E2验证结束{eta}，唯一后续首查安排在提前5分钟的{due}；若届时尚未完成，再按240秒复查并使用新实测速率，不根据时间推断结果。

新增的直接启动资源补查在01:08因认证拒绝而失败；原失败输出保留。随后通过原有授权包装器，01:09:27完成仅资源检查：训练两进程仍存活，A100 28655／40960MiB、利用率100%，系统盘余312578048字节，数据盘余1458102272字节。本次资源补查不读取指标、不执行网络、不删除权重。当前best615023752字节、latest841676832字节均保留；本轮没有新增清理，没有把既往归档释放量当成当前余量。

第2轮被动观察器最初直接启动，因已确认同一路径缺少授权环境，提前取消的只是尚未查询服务器的本地定时器38028／会话48982；没有停止或重启训练。授权通道资源检查成功后，替换为唯一有效观察器51540／会话52851，首查仍为{due}。旧定时器取消与失败资源证据保留，后续不复用其旧owner。

活动网络、native criterion、A/B/G、C-off、seed2027、B8、三轮13749次更新及学习率不变。本轮没有模型晋级；最好仍是共同E0的5677／4920，完整终点与三个有效机制尚未证实。Nr/Sr来源、作者权重、同Query输出与GPU预检准备沿用上一节；尚未正式开训，也未固定最终结构。不得把本轮E1与旧E3拼接成终态结果。

下一步：完成当前规定预算并验证最佳、最新恢复状态；根据同预算完整结果判断C以及正常联合适配，而不是继续扩大普通监督。预备的独立面几何方案不在活动源码中插入，Nr/Sr也不抢占当前唯一A100。研究目标仍为同一完整模型ScanRefer>59.5%／>51%、三项有直接证据的有效改进、随后对应作者权重起点训练完整Nr3D和Sr3D，ACTIVE_UNMET。
'''.format(eta=plan['estimated_epoch2_full_validation_end_cst'],due=plan['due_cst'])
(data/'HANDOFF_C_OFF_EPOCH1_20261011.md').write_text(notes,encoding='utf-8')
names=['normal_controls_20261010/'+name for name in (
    'HANDOFF_C_OFF_EPOCH1_20261011.md','NORMAL_FIRST_OBSERVATION.json','NORMAL_FIRST_OBSERVATION_STDOUT.json',
    'NORMAL_FIRST_OBSERVATION_EXIT.json','NORMAL_FIRST_OBSERVATION_SUMMARY.json','NORMAL_NEXT_OBSERVATION_PLAN.json',
    'prepare_epoch2_observation_20261011.py','observe_c_off_epoch2_planned_authorized.py',
    'observe_c_off_epoch2_authorized_route.py','NORMAL_EPOCH2_OBSERVER_OWNER.json','NORMAL_EPOCH2_OBSERVER_LIVE_WITNESS.json',
    'NORMAL_EPOCH2_AUTHORIZED_OBSERVER_OWNER.json','EPOCH2_AUTH_ROUTE_REPAIR.json',
    'collect_epoch1_resources_authorized.py','collect_epoch1_resources_authorized_attempt2.py',
    'EPOCH1_RESOURCE_EXIT.json','EPOCH1_RESOURCE_ATTEMPT2_STDOUT.json','EPOCH1_RESOURCE_ATTEMPT2_EXIT.json',
    'EPOCH1_RESOURCE_ATTEMPT2_RECEIPT.json')]
names += ['normal_controls_20261010/'+p.relative_to(data).as_posix() for p in (data/'normal_first_observation').rglob('*') if p.is_file()]
names += ['prepare_c_off_epoch1_publication_20261011.py','record_c_off_epoch1_publication_20261011.py']
(root/'C_OFF_EPOCH1_PUBLIC_FILE_LIST.json').write_text(json.dumps(names,indent=2)+'\n')
head = '''"""Publish completed C-off E1 evidence, without querying the training job."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
prior = json.loads((root/'referit_gpu_source_publication.json').read_bytes())
assert prior['section']=='20.376.144' and prior['remote_handoff_sync_complete'] and prior['github_main_verified']
prior_receipt = json.loads(Path(prior['remote_sync_receipt']).read_bytes())
assert prior_receipt['remote_sync_complete'] and prior_receipt['doc_sha256']==prior['doc_sha256']
assert not (root/'c_off_epoch1_publication.json').exists()
data = root/'normal_controls_20261010'
summary=json.loads((data/'NORMAL_FIRST_OBSERVATION_SUMMARY.json').read_bytes())
plan=json.loads((data/'NORMAL_NEXT_OBSERVATION_PLAN.json').read_bytes())
owner=json.loads((data/'NORMAL_EPOCH2_AUTHORIZED_OBSERVER_OWNER.json').read_bytes())
assert summary['epoch1_delta_vs_own_E0']==[-33,-316] and summary['formal_results_unaudited']
assert owner['local_pid']==51540 and owner['first_due_cst']==plan['due_cst']
repos=[Path('C:/Users/gb')/name for name in ('.codex_mcln_g0_20260905','.codex_pvground_cs_20261002','.codex_mcln_v99_internal_20260928')]
doc='docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop=Path('C:/Users/gb/Desktop/document')/Path(doc).name
old=(repos[0]/doc).read_bytes()
assert hashlib.sha256(old).hexdigest()==prior['doc_sha256'] and b'20.376.145' not in old
assert all((repo/doc).read_bytes()==old for repo in repos) and desktop.read_bytes()==old
for index,repo in enumerate(repos):
 assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==prior['heads'][index]
 assert not subprocess.check_output(['git','-C',str(repo),'-c','core.longpaths=true','status','--porcelain']).strip()
notes=(data/'HANDOFF_C_OFF_EPOCH1_20261011.md').read_text(encoding='utf-8')
document=old+('\\n\\n'+notes.replace('\\r\\n','\\n')).replace('\\n','\\r\\n').encode('utf-8')
digest=hashlib.sha256(document).hexdigest()
prefix='refine-logs/pvground_native_joint_training_20261009/normal_c_off_epoch1_20261011/'
names=json.loads((root/'C_OFF_EPOCH1_PUBLIC_FILE_LIST.json').read_bytes())+['C_OFF_EPOCH1_PUBLIC_FILE_LIST.json','publish_c_off_epoch1_authorized_20261011.py']
files={prefix+name:(root/name).read_bytes() for name in sorted(set(names))}
files[prefix+'.gitattributes']=b'** -text whitespace=-blank-at-eol,-space-before-tab,cr-at-eol,-blank-at-eof\\n'
assert not any('STDERR' in name or '.aris' in name or 'run_mcln_authorized_20260908.py' in name for name in files)
file_hashes={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}
assert all(not (repo/prefix).exists() for repo in repos[:2])
'''
template=(root/'publish_referit_gpu_source_authorized.py').read_text(encoding='utf-8')
tail=template[template.index("remote_code = r'''import base64,hashlib,json,sys"):]
for old,new in [
    ('referit_native_gpu_preparation_20261010','normal_c_off_epoch1_20261011'),
    ('tmp_referit_gpu_source','tmp_c_off_epoch1'),
    ('REFERIT_GPU_SOURCE_PUBLICATION','C_OFF_EPOCH1_PUBLICATION'),
    ('referit_gpu_source_local_commit','c_off_epoch1_local_commit'),
    ('referit_gpu_source_publication','c_off_epoch1_publication'),
    ('REFERIT_GPU_SOURCE_ALL_','C_OFF_EPOCH1_ALL_'),
    ('20.376.144','20.376.145'),
    ('Prepare native ReferIt mixed-batch training and recovery checks','Record C-off normal joint E1 and measured next observation')]:
    assert old in tail,old
    tail=tail.replace(old,new)
begin=tail.index('    remote_sync_receipt=str(receipt_path), reviewed_result=')
end=tail.index("(root / 'c_off_epoch1_local_commit.json')",begin)
tail=tail[:begin]+'''    remote_sync_receipt=str(receipt_path), observed_result=summary,
    interim_results_unaudited=True, selected_epoch=0, selected_epoch_scope='retained_parent_not_new_training_gain',
    next_observation_cst=plan['due_cst'], observer_native_session=52851, observer_pid=51540,
    c_off_control_training_launched=True, Nr3D_or_Sr3D_training_launched=False,
    active_training_source_changed=False, publication_training_status_queries=0,
    training_restart=False, full_goal_complete=False)
'''+tail[end:]
content=head+tail
ast.parse(content,feature_version=(3,7))
path=root/'publish_c_off_epoch1_authorized_20261011.py'
assert not path.exists()
path.write_text(content,encoding='utf-8')
print(json.dumps(dict(status='C_OFF_EPOCH1_STATIC_PUBLICATION_PREPARED',files=len(names),next_due_cst=plan['due_cst'],new_training_queries=0)))
