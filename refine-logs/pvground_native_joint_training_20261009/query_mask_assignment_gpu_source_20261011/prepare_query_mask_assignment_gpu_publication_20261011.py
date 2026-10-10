"""Prepare reviewed GPU-preflight source for publication, without executing it."""
import ast
import datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
data = root / 'query_mask_assignment_gpu_20261011'
review_path = data / 'source_review/SOURCE_REVIEW.json'
review = json.loads(review_path.read_bytes())
assert review['execution_scope'] == 'SOURCE_ONLY'
assert review['verdict'] in ('PASS', 'WARN') and not review['blocking_findings']
for path, digest in review['audited_input_hashes'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
preparation = json.loads((data / 'PREPARATION.json').read_bytes())
assert not preparation['serial_gpu_preflight_admitted'] and not preparation['actual_GPU_preflight_completed']
assert preparation['current_training_queries'] == preparation['current_training_source_mutations'] == 0
assert preparation['formal_accuracy'] is None
for mode in ('text', 'query'):
    protocol = json.loads((data / (mode + '.json')).read_bytes())
    assert protocol['matcher_mask_source'] == mode and not protocol['serial_gpu_preflight_admitted']
summary = dict(status='NATIVE_QUERY_MASK_MATCHER_GPU_PREFLIGHT_SOURCE_REVIEWED_NOT_ADMITTED',
    recorded_cst=datetime.datetime.now().astimezone().isoformat(),
    source_review_verdict=review['verdict'], source_review_blocking_findings=[],
    source_review_sha256=hashlib.sha256(review_path.read_bytes()).hexdigest(),
    source_only=True, copied_native_sources=14, native_sources_equal_to_reviewed_query_variant=True,
    planned_modes=['text', 'query'], planned_real_rows_per_mode=16, planned_updates_per_mode=2,
    existing_native_training_epoch_and_full_criterion_reused=True,
    initial_checkpoint='/root/autodl-tmp/pvground_native_joint_training_20261010/normal/logs/scanrefer/extremal_support/1791573559/best.pth',
    initial_checkpoint_sha256='ed8455ddc67e4d17018e9cf499197140e15e35db3f5daee55e0eec6846faf0c4',
    prior_measured_whole_preflight_seconds=647.0771946460009,
    prior_measured_two_update_seconds=25.904242292046547,
    approximate_per_mode_minutes=11, timing_is_estimate_not_new_measurement=True,
    future_polling='Estimate full completion; first read a few minutes before it, then240 seconds only if overdue.',
    native_GPU_preflight_completed=False, serial_gpu_preflight_admitted=False,
    new_training_started=False, queue_or_launcher_created=False,
    active_training_source_mutations=0, current_training_queries=0,
    neural_calls=0, optimizer_updates=0, formal_accuracy=None,
    review_independence='same-family', acceptance_status='provisional', actual_identity_attestation='UNATTESTED',
    next_main_observation_cst='2026-10-11T05:00:40.968644+08:00', full_goal_complete=False)
(data / 'SOURCE_READINESS_SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
notes = '''## §20.376.150 — Query Mask匹配的真实正常训练预检入口：源码准备

已将§148/149的隔离匹配改动接入现有native_c_off_preflight入口，仍仅SOURCE_ONLY。14份原生源与§148 Query版本逐字节相同；本次新增的是工程检查入口和两份模式配置，没有修改活动C-off源码、执行GPU或创建部署器/排队任务。完整PV依赖仍须从既有运行时绑定部署，14份局部源不能冒充完整仓库。

text/query两份配置唯一差异为matcher_mask_source。两者均保持同一保留E0模型状态、fresh优化器、G保留/C关闭、B8/seed2027、核心骨干LR1e-6/新增1e-5/WD0.0005/clip0.1；原生RoBERTa冻结，核心、骨干及G/A/B正常联合更新。父状态包含历史C适配，不是从头无C消融。此处仍是ScanRefer表达微调，不改称作者完整混合检测配方。

每种模式计划用真实原训练loader的两个增强batch，共16行和2次更新，调用原train_one_epoch与完整原生criterion。保留原模板已有梯度/参数更新、初始1295模型状态与保留E0逐项相等、完整Adam/scheduler/全部RNG保存及冷恢复检查。作为初始一致性参照的checkpoint继续使用受保护的C-on/E0 best.pth（ed8455dd…6faf0c4），不使用活动C-off训练终点，也不使用预检更新后的状态进入正式训练。

新增只读匹配观察器在同一实际前向输出与GT上计算另一Mask来源的匹配，并始终返回配置来源的原结果给loss。记录全部有效GT、Query分配变化、对应最终Box IoU、Own Query Mask交并数/二值成本，以及实际匹配来源的Mask成本。Own成本与Text控制实际成本分开标注；没有改变目标、候选、分数或添加损失。无Mask的早期prefix不新增比较，预计末层每步一次。

匹配模式写入checkpoint config，并在恢复后通过显式CLI参数新建criterion核对。原load_checkpoint不自动以config覆盖CLI，因此正式恢复须携带原matcher_mask_source；不能只因模型状态恢复就声称监督模式自动恢复。

两个protocol的serial_gpu_preflight_admitted均为false，入口在导入Torch之前拒绝执行。必须先等当前C-off三轮终态及判断，之后才考虑独立完整源、父权重、空闲A100/GPU锁、磁盘保存空间与真实工程预检。两个模式输入是否实际一致、成本竞争和匹配变化幅度仍未实测。

旧C-off同类实际预检总体647.08秒，两次更新25.90秒。后续每模式约11分钟仅作初始估计，查看按整体完成时间安排，逾时才每240秒继续；不能将两步计算耗时当成全部加载/恢复耗时，也不在现在创建新观察器。

限定源码审查WARN/0阻断，14源/模板及两份配置差异已核对；same-family/provisional、实际identity UNATTESTED。源码准备不是GPU通过、训练启动、新精度或三有效机制证明。新正式控制是否能复用当前text行为等价的C-off终态，要在完整终态和真实预检后决定，不预先重复启动三轮控制。

原观察器52851/PID51540仍按Oct11 05:00:40检查正常第2轮。保留最好5677/4920及必要父链/V99。没有新增ScanRefer/Nr3D/Sr3D指标，目标ACTIVE_UNMET。
'''
(data / 'HANDOFF_QUERY_MASK_ASSIGNMENT_GPU_SOURCE_20261011.md').write_text(notes, encoding='utf-8')
names = ['query_mask_assignment_gpu_20261011/' + name for name in (
    'PREPARATION.json', 'SOURCE_HASHES.json', 'init.json', 'text.json', 'query.json', 'EXPERIMENT_PLAN.md',
    'native_query_mask_matcher_preflight.py', 'source_review/SOURCE_REVIEW.md', 'source_review/SOURCE_REVIEW.json',
    'SOURCE_READINESS_SUMMARY.json', 'HANDOFF_QUERY_MASK_ASSIGNMENT_GPU_SOURCE_20261011.md')]
names += ['prepare_query_mask_assignment_gpu_20261011.py',
    'prepare_query_mask_assignment_gpu_publication_20261011.py', 'record_query_mask_assignment_gpu_publication_20261011.py']
(root / 'QUERY_MASK_ASSIGNMENT_GPU_PUBLIC_FILE_LIST.json').write_text(json.dumps(names, indent=2) + '\n')
head = '''"""Publish reviewed native matcher GPU-preflight source; no execution."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

root=Path(__file__).resolve().parent
prior=json.loads((root/'query_mask_assignment_cpu_publication.json').read_bytes())
assert prior['section']=='20.376.149' and prior['remote_handoff_sync_complete'] and prior['github_main_verified']
prior_receipt=json.loads(Path(prior['remote_sync_receipt']).read_bytes())
assert prior_receipt['remote_sync_complete'] and prior_receipt['doc_sha256']==prior['doc_sha256']
assert not (root/'query_mask_assignment_gpu_source_publication.json').exists()
data=root/'query_mask_assignment_gpu_20261011'
summary=json.loads((data/'SOURCE_READINESS_SUMMARY.json').read_bytes())
assert summary['status']=='NATIVE_QUERY_MASK_MATCHER_GPU_PREFLIGHT_SOURCE_REVIEWED_NOT_ADMITTED'
assert not summary['serial_gpu_preflight_admitted'] and not summary['native_GPU_preflight_completed']
assert summary['formal_accuracy'] is None and summary['current_training_queries']==0
audit=json.loads((data/'source_review/SOURCE_REVIEW.json').read_bytes())
assert audit['execution_scope']=='SOURCE_ONLY' and audit['verdict'] in ['PASS','WARN'] and not audit['blocking_findings']
for path,digest in audit['audited_input_hashes'].items():
 assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
plan=json.loads((root/'normal_controls_20261010/NORMAL_NEXT_OBSERVATION_PLAN.json').read_bytes())
owner=json.loads((root/'normal_controls_20261010/NORMAL_EPOCH2_AUTHORIZED_OBSERVER_OWNER.json').read_bytes())
assert owner['local_pid']==51540 and owner['first_due_cst']==plan['due_cst']
repos=[Path('C:/Users/gb')/name for name in ('.codex_mcln_g0_20260905','.codex_pvground_cs_20261002','.codex_mcln_v99_internal_20260928')]
doc='docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop=Path('C:/Users/gb/Desktop/document')/Path(doc).name
old=(repos[0]/doc).read_bytes()
assert hashlib.sha256(old).hexdigest()==prior['doc_sha256'] and b'20.376.150' not in old
assert all((repo/doc).read_bytes()==old for repo in repos) and desktop.read_bytes()==old
for index,repo in enumerate(repos):
 assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==prior['heads'][index]
 assert not subprocess.check_output(['git','-C',str(repo),'-c','core.longpaths=true','status','--porcelain']).strip()
notes=(data/'HANDOFF_QUERY_MASK_ASSIGNMENT_GPU_SOURCE_20261011.md').read_text(encoding='utf-8')
document=old+('\\n\\n'+notes.replace('\\r\\n','\\n')).replace('\\n','\\r\\n').encode('utf-8')
digest=hashlib.sha256(document).hexdigest()
prefix='refine-logs/pvground_native_joint_training_20261009/query_mask_assignment_gpu_source_20261011/'
names=json.loads((root/'QUERY_MASK_ASSIGNMENT_GPU_PUBLIC_FILE_LIST.json').read_bytes())+['QUERY_MASK_ASSIGNMENT_GPU_PUBLIC_FILE_LIST.json','publish_query_mask_assignment_gpu_authorized_20261011.py']
files={prefix+name:(root/name).read_bytes() for name in sorted(set(names))}
files[prefix+'.gitattributes']=b'** -text whitespace=-blank-at-eol,-space-before-tab,cr-at-eol,-blank-at-eof\\n'
assert not any('STDERR' in name or '.aris' in name or 'run_mcln_authorized_20260908.py' in name for name in files)
file_hashes={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}
assert all(not (repo/prefix).exists() for repo in repos[:2])
'''
template = (root / 'publish_query_mask_assignment_cpu_authorized_20261011.py').read_text(encoding='utf-8')
tail = template[template.index("remote_code = r'''import base64,hashlib,json,sys"):]
for before, after in (
    ('query_mask_assignment_cpu_20261011', 'query_mask_assignment_gpu_source_20261011'),
    ('tmp_query_mask_assignment_cpu', 'tmp_query_mask_assignment_gpu_source'),
    ('QUERY_MASK_ASSIGNMENT_CPU_PUBLICATION', 'QUERY_MASK_ASSIGNMENT_GPU_SOURCE_PUBLICATION'),
    ('query_mask_assignment_cpu_local_commit', 'query_mask_assignment_gpu_source_local_commit'),
    ('query_mask_assignment_cpu_publication', 'query_mask_assignment_gpu_source_publication'),
    ('QUERY_MASK_ASSIGNMENT_CPU_ALL_', 'QUERY_MASK_ASSIGNMENT_GPU_SOURCE_ALL_'),
    ('20.376.149', '20.376.150'),
    ('Verify native Query Mask assignment and box gradients on CPU', 'Prepare native Query Mask matching real-batch preflight')):
    assert before in tail, before
    tail = tail.replace(before, after)
content = head + tail
ast.parse(content, feature_version=(3, 7))
path = root / 'publish_query_mask_assignment_gpu_authorized_20261011.py'
assert not path.exists()
path.write_text(content, encoding='utf-8')
print(json.dumps(dict(status='QUERY_MASK_ASSIGNMENT_GPU_SOURCE_PUBLICATION_PREPARED', actual_GPU_execution=False)))
