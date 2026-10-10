"""Prepare publication only after actual isolated CPU evidence is reviewed."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
data = root / 'query_mask_assignment_20261011'
summary = json.loads((data / 'CPU_CHECK_SUMMARY.json').read_bytes())
assert summary['status'] == 'ISOLATED_NATIVE_QUERY_MASK_ASSIGNMENT_CPU_CHECK_AUDITED'
assert not summary['actual_review_blocking_findings']
assert summary['formal_accuracy'] is None and summary['current_training_queries'] == 0
assert summary['gpu_calls'] == 0 and not summary['new_training_started']
notes = '''## §20.376.149 — 候选自身Mask匹配：限定原生CPU执行与定位梯度

§148的隔离源码已完成实际CPU工程检查。原活动C-off正常训练配方及源码保持；本次没有读取它的进度、占用A100或启动额外训练。

检查在既有PV运行时执行，CUDA_VISIBLE_DEVICES为空、线程1；导入实际warm原losses与隔离modified losses，并确认utils绑定。输入是刻意构造的root-only、root+anchor及混合GT批次，B1/B2、Q256、token256、4超点映射6个原始点。没有加载真实PV模型或数据，所以不是正式精度评估。

共33次matcher调用、44个批次分配问题、66次重复合成GT配对，不是66个独立数据样本；其中6次通过原生SetCriterion.forward仅请求boxes损失并反向。默认和显式text的完整成本与warm原版逐元素相同；text下交换自身Mask不改成本，query下交换自身Mask会改成本及实际分配；query下改变公共Text Mask不改成本。去掉Mask输出时，两种模式的早期匹配成本相同。每个有效GT仍获唯一Query，未改为root-only多正例。完整成本矩阵仅在进程内比对，归档的是差异、分配与梯度摘要及退出记录。

固定框与语义，给同一GT安排两个几何成本相同的候选，是为了验证新增路径是否真实生效；不能据此估计真实场景中有多少匹配会改变或涨多少点。原生Mask成本系数仍0.0002，真实50k点的尺度及其他成本竞争仍需真实输入预检。

原生boxes-only criterion的L1/GIoU损失直接梯度只作用本次匹配Query；切换来源后，受到直接定位监督的候选随分配变化。这证明了新增路径可以改变定位责任，不证明真实候选身份更正确，也不证明整个训练目标无冲突。未匹配框的该项直接输出梯度为0，不意味着共享参数或其他任务完全不会训练它们。

本次没有执行完整compute_hungarian_loss、Mask/语义/对比损失、G修正、模型前向、优化器或GPU。不能把boxes-only检查写成正常联合训练预检通过。真实完整PV批次、显存/前后向/保存恢复与正式9508条双阈值仍未完成。

源码审查与实际证据审查均保留same-family/provisional、实际身份UNATTESTED，阻断项为0；限定工程检查完成，正式方法与三有效贡献仍未准入。它不是首次Mask参与匹配，相关前作已在§148说明。

下一步依然先等当前三轮C-off完整终态，再决定这个训练匹配改动是否值得做真实数据预检和同起点同预算正常对照。唯一原观察器52851/PID51540按Oct11 05:00:40查看第2轮，不新增排队GPU任务。保留最好5677/4920及全部必要父链；没有新ScanRefer/Nr3D/Sr3D指标，目标ACTIVE_UNMET。
'''
(data / 'HANDOFF_QUERY_MASK_ASSIGNMENT_CPU_20261011.md').write_text(notes, encoding='utf-8')
names = ['query_mask_assignment_20261011/' + name for name in (
    'CPU_CHECK_SPEC.json', 'check_query_mask_assignment_cpu.py', 'run_cpu_authorized.py', 'summarize_cpu_check.py',
    'source/models/losses.py', 'cpu_source_review/SOURCE_REVIEW.md', 'cpu_source_review/SOURCE_REVIEW.json',
    'cpu_actual_review/ACTUAL_REVIEW.md', 'cpu_actual_review/ACTUAL_REVIEW.json',
    'cpu_execution/CPU_EXECUTION.json', 'cpu_execution/QUERY_MASK_ASSIGNMENT_CPU_RESULT.json',
    'cpu_execution/CPU_STDOUT.txt', 'cpu_execution/TRANSPORT_EXIT.json',
    'CPU_CHECK_SUMMARY.json', 'HANDOFF_QUERY_MASK_ASSIGNMENT_CPU_20261011.md')]
names += ['prepare_query_mask_assignment_cpu_20261011.py',
    'prepare_query_mask_assignment_cpu_publication_20261011.py', 'record_query_mask_assignment_cpu_publication_20261011.py']
(root / 'QUERY_MASK_ASSIGNMENT_CPU_PUBLIC_FILE_LIST.json').write_text(json.dumps(names, indent=2) + '\n')
template = (root / 'publish_query_mask_assignment_authorized_20261011.py').read_text(encoding='utf-8')
head = template[:template.index("remote_code = r'''import base64,hashlib,json,sys")]
head = head.replace('"""Publish isolated training matcher source only; no GPU or training query."""',
    '"""Publish actual isolated CPU evidence; no GPU or training query."""')
head = head.replace("root/'matcher_mask_role_publication.json'", "root/'query_mask_assignment_source_publication.json'")
head = head.replace("prior['section']=='20.376.147'", "prior['section']=='20.376.148'")
head = head.replace("root/'query_mask_assignment_source_publication.json').exists()", "root/'query_mask_assignment_cpu_publication.json').exists()")
head = head.replace("data/'SOURCE_READINESS_SUMMARY.json'", "data/'CPU_CHECK_SUMMARY.json'")
head = head.replace("summary['status']=='ISOLATED_NATIVE_QUERY_MASK_ASSIGNMENT_SOURCE_REVIEWED_NOT_ADMITTED'",
    "summary['status']=='ISOLATED_NATIVE_QUERY_MASK_ASSIGNMENT_CPU_CHECK_AUDITED'")
head = head.replace("assert not summary['training_started'] and not summary['actual_CPU_check_completed']",
    "assert not summary['new_training_started'] and summary['actual_review_blocking_findings']==[] and summary['gpu_calls']==0")
head = head.replace("data/'source_review/SOURCE_REVIEW.json'", "data/'cpu_actual_review/ACTUAL_REVIEW.json'")
head = head.replace("audit['execution_scope']=='SOURCE_ONLY'", "audit['execution_scope']=='ACTUAL_CPU_ONLY'")
head = head.replace("b'20.376.148' not in old", "b'20.376.149' not in old")
head = head.replace('HANDOFF_QUERY_MASK_ASSIGNMENT_SOURCE_20261011.md', 'HANDOFF_QUERY_MASK_ASSIGNMENT_CPU_20261011.md')
tail = template[template.index("remote_code = r'''import base64,hashlib,json,sys"):]
content = head + tail
for before, after in (
    ('query_mask_assignment_source_20261011/', 'query_mask_assignment_cpu_20261011/'),
    ('query_mask_assignment_source_20261011\'','query_mask_assignment_cpu_20261011\''),
    ('QUERY_MASK_ASSIGNMENT_PUBLIC_FILE_LIST.json', 'QUERY_MASK_ASSIGNMENT_CPU_PUBLIC_FILE_LIST.json'),
    ('publish_query_mask_assignment_authorized_20261011.py', 'publish_query_mask_assignment_cpu_authorized_20261011.py'),
    ('tmp_query_mask_assignment', 'tmp_query_mask_assignment_cpu'),
    ('QUERY_MASK_ASSIGNMENT_PUBLICATION', 'QUERY_MASK_ASSIGNMENT_CPU_PUBLICATION'),
    ('query_mask_assignment_source_local_commit', 'query_mask_assignment_cpu_local_commit'),
    ('query_mask_assignment_source_publication', 'query_mask_assignment_cpu_publication'),
    ('QUERY_MASK_ASSIGNMENT_ALL_', 'QUERY_MASK_ASSIGNMENT_CPU_ALL_'),
    ("section='20.376.148'", "section='20.376.149'"),
    ('Prepare native Query Mask assignment source control', 'Verify native Query Mask assignment and box gradients on CPU')):
    content = content.replace(before, after)
# Keep the prior receipt reference pointing to the completed source publication.
content = content.replace("prior=json.loads((root/'query_mask_assignment_cpu_publication.json').read_bytes())",
    "prior=json.loads((root/'query_mask_assignment_source_publication.json').read_bytes())")
ast.parse(content, feature_version=(3, 7))
path = root / 'publish_query_mask_assignment_cpu_authorized_20261011.py'
assert not path.exists()
path.write_text(content, encoding='utf-8')
print(json.dumps(dict(status='QUERY_MASK_ASSIGNMENT_ACTUAL_CPU_PUBLICATION_PREPARED', formal_accuracy=None)))
