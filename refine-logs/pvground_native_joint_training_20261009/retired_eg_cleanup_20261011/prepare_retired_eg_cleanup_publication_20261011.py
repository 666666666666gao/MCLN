"""Publish completed exact cache retirement, without any new training query."""
import ast
import datetime
import json
from pathlib import Path

root=Path(__file__).resolve().parent
data=root/'normal_controls_20261010'
archive=json.loads((data/'RETIRED_EG_ARRAY_ARCHIVE_COMPLETE.json').read_bytes())
retire=json.loads((data/'RETIRED_EG_ARRAY_RETIREMENT.json').read_bytes())
assert archive['status']=='EXACT_THREE_RETIRED_EG_ARRAYS_FULLY_ARCHIVED_NOT_DELETED'
assert retire['status']=='ONLY_THREE_FULLY_ARCHIVED_RETIRED_EG_ARRAYS_DELETED'
assert retire['logical_bytes_removed']==411318656 and retire['model_weights_deleted']==0
assert retire['original_archive_native_exit_code']==0
notes='''## §20.376.146 — 三份历史EG候选数组归档后清理

沿用用户对无用生成权重／缓存的持续清理授权；本节没有模型更新、精度结果或新的训练轮询。

三份已结束EG实验的candidates.npy分别为Nr-adapt137505920字节、Nr-transfer137505920字节、Scan-acceptance136306816字节，共411318656字节。原始文件全量SHA先在远端确认；用无损gzip流传输，没有远端临时大文件。原归档会话84511于2026年10月11日02:11:33完成，退出码0，总用时2801.93秒。传输压缩量322504481字节；没有截断候选、改变浮点数据或重跑网络。

本地C:/Users/gb/.codex/archives/pvg_retired_candidate_arrays_20261011下同时保留完整原始数组与压缩副本。02:13:49再次重读三份本地原始文件，完整字节数和SHA均与远端一致；远端再次全量核对后，于02:13:57仅删除这三个明确路径：

- /root/autodl-tmp/mcln_eg3dvg_nr3d_adapt_20260920_v3/evaluation/formal/candidates.npy
- /root/autodl-tmp/mcln_eg3dvg_nr3d_transfer_20260920_v1/formal/candidates.npy
- /root/autodl-tmp/mcln_eg3dvg_acceptance_20260920_v1/formal/candidates.npy

三份均为单链接文件。实测数据盘可用空间1457647616→1868984320字节，增加411336704字节；与逻辑文件字节数的小差异来自文件系统分配统计。本次没有删除任何模型权重、数据集、环境、活动源码或恢复状态。

系统盘这次前后均为320847872字节，没有因本次清理增加；不能将数据盘释放量写成系统盘释放量。此前仅检查与当前已安装Torch/numpy/scipy/spconv版本匹配的≥16MiB pip缓存轮包，未发现候选，因此没有删除安装缓存；这不代表所有缓存都已穷尽。

活动PV的256个候选保留完整，历史数组也在本地完整保留；清理的是服务器上的旧实验副本，没有将低分候选从模型中裁掉。最好与当前latest、官方三数据集权重、G/A/B父状态、V99、数据和运行环境继续保留。

没有因传输等待或早期估计误差重启归档，也没有执行原04:08的保守检查计划；真实完成时间早于该估计。唯一后续训练观察器仍为52851／PID51540，2026年10月11日05:00:40首查时间不变。本节0网络前向、0优化器更新、0训练状态读取，Nr/Sr尚未开训，完整目标仍ACTIVE_UNMET。
'''
(data/'HANDOFF_RETIRED_EG_CLEANUP_20261011.md').write_text(notes,encoding='utf-8')
names=['normal_controls_20261010/'+name for name in (
 'HANDOFF_RETIRED_EG_CLEANUP_20261011.md','inspect_runtime_installation_cache_authorized.py',
 'RUNTIME_INSTALLATION_CACHE_INVENTORY.json','RUNTIME_CACHE_INVENTORY_STDOUT.json','RUNTIME_CACHE_INVENTORY_EXIT.json',
 'archive_retired_eg_arrays_20261011_authorized.py','RETIRED_EG_ARRAY_ARCHIVE_OWNER.json','RETIRED_EG_ARRAY_IDENTITIES.json',
 'RETIRED_EG_ARRAY_IDENTITY_STDOUT.json','RETIRED_EG_ARRAY_IDENTITY_EXIT.json','RETIRED_EG_ARRAY_ARCHIVE_COMPLETE.json',
 'RETIRED_EG_ARRAY_ARCHIVE_NEXT_CHECK.json','retire_archived_eg_arrays_20261011_authorized.py','RETIRED_EG_LOCAL_SHA_RECHECK.json',
 'RETIRED_EG_ARRAY_RETIREMENT.json','RETIRED_EG_RETIREMENT_STDOUT.json','RETIRED_EG_RETIREMENT_EXIT.json')]
names+=['prepare_retired_eg_cleanup_publication_20261011.py','record_retired_eg_cleanup_publication_20261011.py']
(root/'RETIRED_EG_CLEANUP_PUBLIC_FILE_LIST.json').write_text(json.dumps(names,indent=2)+'\n')
head='''"""Publish completed old EG cache retirement only; no training-state read."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

root=Path(__file__).resolve().parent
prior=json.loads((root/'c_off_epoch1_publication.json').read_bytes())
assert prior['section']=='20.376.145' and prior['remote_handoff_sync_complete'] and prior['github_main_verified']
prior_receipt=json.loads(Path(prior['remote_sync_receipt']).read_bytes())
assert prior_receipt['remote_sync_complete'] and prior_receipt['doc_sha256']==prior['doc_sha256']
assert not (root/'retired_eg_cleanup_publication.json').exists()
data=root/'normal_controls_20261010'
summary=json.loads((data/'RETIRED_EG_ARRAY_RETIREMENT.json').read_bytes())
assert summary['logical_bytes_removed']==411318656 and summary['model_weights_deleted']==0
assert summary['training_status_reads']==0 and summary['neural_calls']==0
plan=json.loads((data/'NORMAL_NEXT_OBSERVATION_PLAN.json').read_bytes())
owner=json.loads((data/'NORMAL_EPOCH2_AUTHORIZED_OBSERVER_OWNER.json').read_bytes())
assert owner['local_pid']==51540 and owner['first_due_cst']==plan['due_cst']
repos=[Path('C:/Users/gb')/name for name in ('.codex_mcln_g0_20260905','.codex_pvground_cs_20261002','.codex_mcln_v99_internal_20260928')]
doc='docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop=Path('C:/Users/gb/Desktop/document')/Path(doc).name
old=(repos[0]/doc).read_bytes()
assert hashlib.sha256(old).hexdigest()==prior['doc_sha256'] and b'20.376.146' not in old
assert all((repo/doc).read_bytes()==old for repo in repos) and desktop.read_bytes()==old
for index,repo in enumerate(repos):
 assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==prior['heads'][index]
 assert not subprocess.check_output(['git','-C',str(repo),'-c','core.longpaths=true','status','--porcelain']).strip()
notes=(data/'HANDOFF_RETIRED_EG_CLEANUP_20261011.md').read_text(encoding='utf-8')
document=old+('\\n\\n'+notes.replace('\\r\\n','\\n')).replace('\\n','\\r\\n').encode('utf-8')
digest=hashlib.sha256(document).hexdigest()
prefix='refine-logs/pvground_native_joint_training_20261009/retired_eg_cleanup_20261011/'
names=json.loads((root/'RETIRED_EG_CLEANUP_PUBLIC_FILE_LIST.json').read_bytes())+['RETIRED_EG_CLEANUP_PUBLIC_FILE_LIST.json','publish_retired_eg_cleanup_authorized_20261011.py']
files={prefix+name:(root/name).read_bytes() for name in sorted(set(names))}
files[prefix+'.gitattributes']=b'** -text whitespace=-blank-at-eol,-space-before-tab,cr-at-eol,-blank-at-eof\\n'
assert not any('STDERR' in name or '.aris' in name or 'run_mcln_authorized_20260908.py' in name for name in files)
file_hashes={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}
assert all(not (repo/prefix).exists() for repo in repos[:2])
'''
template=(root/'publish_c_off_epoch1_authorized_20261011.py').read_text(encoding='utf-8')
tail=template[template.index("remote_code = r'''import base64,hashlib,json,sys"):]
for old,new in [('normal_c_off_epoch1_20261011','retired_eg_cleanup_20261011'),('tmp_c_off_epoch1','tmp_retired_eg_cleanup'),('C_OFF_EPOCH1_PUBLICATION','RETIRED_EG_CLEANUP_PUBLICATION'),('c_off_epoch1_local_commit','retired_eg_cleanup_local_commit'),('c_off_epoch1_publication','retired_eg_cleanup_publication'),('C_OFF_EPOCH1_ALL_','RETIRED_EG_CLEANUP_ALL_'),('20.376.145','20.376.146'),('Record C-off normal joint E1 and measured next observation','Archive and retire three closed EG candidate caches')]:
 assert old in tail,old
 tail=tail.replace(old,new)
begin=tail.index('    remote_sync_receipt=str(receipt_path), observed_result=summary,')
end=tail.index("(root / 'retired_eg_cleanup_local_commit.json')",begin)
tail=tail[:begin]+'''    remote_sync_receipt=str(receipt_path), cleanup_result=summary,
    selected_epoch=0, selected_epoch_scope='retained_parent_not_new_training_gain',
    next_observation_cst=plan['due_cst'], observer_native_session=52851, observer_pid=51540,
    c_off_control_training_launched=True, Nr3D_or_Sr3D_training_launched=False,
    active_training_source_changed=False, publication_training_status_queries=0,
    training_restart=False, full_goal_complete=False)
'''+tail[end:]
content=head+tail
ast.parse(content,feature_version=(3,7))
path=root/'publish_retired_eg_cleanup_authorized_20261011.py'
assert not path.exists()
path.write_text(content,encoding='utf-8')
print(json.dumps(dict(status='RETIRED_EG_CLEANUP_STATIC_PUBLICATION_PREPARED',logical_bytes_removed=411318656,files=len(names),next_normal_due_cst=json.loads((data/'NORMAL_NEXT_OBSERVATION_PLAN.json').read_bytes())['due_cst'],new_training_queries=0)))
