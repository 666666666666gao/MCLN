"""Prepare source-only publication; no CPU/GPU admission or new training."""
import ast
import datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
data = root / 'query_mask_assignment_20261011'
audit_path = data / 'source_review/SOURCE_REVIEW.json'
audit = json.loads(audit_path.read_bytes())
assert audit['execution_scope'] == 'SOURCE_ONLY'
assert audit['verdict'] in ('PASS', 'WARN') and not audit['blocking_findings']
for path, digest in audit['audited_input_hashes'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
base = json.loads((data / 'BASE_SOURCE_HASHES.json').read_bytes())
hashes = json.loads((data / 'SOURCE_HASHES.json').read_bytes())
for name, digest in hashes.items():
    assert hashlib.sha256((data / 'source' / name).read_bytes()).hexdigest() == digest
assert [name for name in base if base[name] != hashes[name]] == ['main_utils.py', 'models/losses.py']
summary = dict(status='ISOLATED_NATIVE_QUERY_MASK_ASSIGNMENT_SOURCE_REVIEWED_NOT_ADMITTED',
    recorded_cst=datetime.datetime.now().astimezone().isoformat(),
    source_review_verdict=audit['verdict'], source_review_blocking_findings=[],
    source_review_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
    changed_source_files=['main_utils.py', 'models/losses.py'], unchanged_source_files=12,
    added_model_parameters=0, model_forward_changed=False,
    original_mask_cost_metric_and_weight_retained=True, native_valid_GT_one_to_one_retained=True,
    source_only=True, actual_CPU_check_completed=False, native_GPU_preflight_completed=False,
    final_method_admitted=False, training_started=False, current_training_queries=0,
    active_training_source_mutations=0, neural_calls=0, formal_accuracy=None,
    no_new_inference_score=True, candidate_queries_retained=256,
    review_independence='same-family', acceptance_status='provisional', actual_identity_attestation='UNATTESTED',
    next_main_observation_cst='2026-10-11T05:00:40.968644+08:00', full_goal_complete=False)
(data / 'SOURCE_READINESS_SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
notes = '''## §20.376.148 — 候选自身Mask参与原生训练匹配的隔离源码

依据§147的限定路径诊断，已实现训练阶段直接Mask成本的来源对照。源码放在独立query_mask_assignment_20261011/source目录，没有修改当前C-off模型源或正在运行的配方；当前仅SOURCE_ONLY，不是新训练启动或精度结果。

十四份共同源中，十二份逐字节保留，仅main_utils.py与models/losses.py变化。新增--matcher_mask_source参数，默认text、另一选项query；原生factory将它显式传给HungarianMatcher。text仍读取公共Text Mask，query读取同候选的sp_pred_masks。其后的>0硬二值判断、超点到原始点映射、点级L1、0.0002系数、分类1/框L1为0/GIoU为2均保留。

正常PV的proposal与0head至4head没有Mask输出，其匹配路径不变；末层才具备自身Query Mask。本实现保留原生全部有效GT的一对一匹配，不按GT槽0将多目标或anchor全部改为root。原生定位、语义、对比、Mask损失及G修正保持；但分配变化可能改变各任务被监督的Query，因此不能说只影响几何，也不能保证原正确候选绝不退化。

新增模型参数为0，没有修改模型前向或last/bbs评分，保留256候选，也没有新增推理GT、候选裁剪或另一套排名。当前C仍关闭；若正式采用，该比较仍是同一已适配父状态上的增量实验，父状态已有历史C训练，不能声称从未训练C的从头消融。

Mask DINO官方matcher已使用候选Mask与GT的CE/Dice成本并与分类及框成本组合；本次不是首次Mask参与匹配，也没有复制其二维采样或sigmoid分类。此处保留PV现有三维点成员与成本形式，只隔离“候选自身支撑是否参与定位责任”。参考[官方实现](https://github.com/IDEA-Research/MaskDINO/blob/main/maskdino/modeling/matcher.py)。它目前是问题导向的训练对照，不是已经验证有效的第三个论文贡献。

新上下文源码审查WARN/0阻断，核对了两份修改、十二份未改源、argparse到factory到matcher、实际warm modules与dataset绑定及早期prefix。审查为same-family/provisional，实际模型身份和推理档位UNATTESTED；不能将静态审查写成实际CPU反向、GPU预检或正式评估通过。

下一步先进行限定原生CPU工程检查：默认/text控制、query实际使用自身Mask、多GT及无Mask早期prefix、真实原生框loss对应的直接输出梯度。合成输入只用于实现验证；当前尚未执行。之后仍须等待当前三轮C-off终态，决定是否安排真实数据GPU预检与同起点同预算正常训练；没有创建额外GPU控制器或排队任务。

若采用，明确使用同一保留E0、fresh优化器、seed2027、B8、三轮及现行学习率，9508条原生同模型评估。既看直接控制，也看是否超过起点5677/4920；只优于退化控制不足以证明新增能力。当前最好权重、父链及V99继续保护。

本节0模型前向、0数据行、0criterion执行、0优化器更新、0GPU及0当前训练状态读取；没有ScanRefer/Nr/Sr新增指标。唯一原观察器52851/PID51540仍按2026年10月11日05:00:40检查正常第2轮，三有效机制与完整Nr/Sr正式训练仍未完成，目标ACTIVE_UNMET。
'''
(data / 'HANDOFF_QUERY_MASK_ASSIGNMENT_SOURCE_20261011.md').write_text(notes, encoding='utf-8')
names = ['query_mask_assignment_20261011/' + name for name in (
    'PLAN.json', 'BASE_SOURCE_HASHES.json', 'SOURCE_HASHES.json', 'EXPERIMENT_PLAN.md',
    'source_review/SOURCE_REVIEW.md', 'source_review/SOURCE_REVIEW.json',
    'SOURCE_READINESS_SUMMARY.json', 'HANDOFF_QUERY_MASK_ASSIGNMENT_SOURCE_20261011.md')]
names += ['query_mask_assignment_20261011/source/' + name for name in hashes]
names += ['prepare_query_mask_assignment_source_20261011.py', 'prepare_query_mask_assignment_publication_20261011.py',
          'record_query_mask_assignment_publication_20261011.py']
(root / 'QUERY_MASK_ASSIGNMENT_PUBLIC_FILE_LIST.json').write_text(json.dumps(names, indent=2) + '\n')
head = '''"""Publish isolated training matcher source only; no GPU or training query."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

root=Path(__file__).resolve().parent
prior=json.loads((root/'matcher_mask_role_publication.json').read_bytes())
assert prior['section']=='20.376.147' and prior['remote_handoff_sync_complete'] and prior['github_main_verified']
prior_receipt=json.loads(Path(prior['remote_sync_receipt']).read_bytes())
assert prior_receipt['remote_sync_complete'] and prior_receipt['doc_sha256']==prior['doc_sha256']
assert not (root/'query_mask_assignment_source_publication.json').exists()
data=root/'query_mask_assignment_20261011'
summary=json.loads((data/'SOURCE_READINESS_SUMMARY.json').read_bytes())
assert summary['status']=='ISOLATED_NATIVE_QUERY_MASK_ASSIGNMENT_SOURCE_REVIEWED_NOT_ADMITTED'
assert not summary['training_started'] and not summary['actual_CPU_check_completed']
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
assert hashlib.sha256(old).hexdigest()==prior['doc_sha256'] and b'20.376.148' not in old
assert all((repo/doc).read_bytes()==old for repo in repos) and desktop.read_bytes()==old
for index,repo in enumerate(repos):
 assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==prior['heads'][index]
 assert not subprocess.check_output(['git','-C',str(repo),'-c','core.longpaths=true','status','--porcelain']).strip()
notes=(data/'HANDOFF_QUERY_MASK_ASSIGNMENT_SOURCE_20261011.md').read_text(encoding='utf-8')
document=old+('\\n\\n'+notes.replace('\\r\\n','\\n')).replace('\\n','\\r\\n').encode('utf-8')
digest=hashlib.sha256(document).hexdigest()
prefix='refine-logs/pvground_native_joint_training_20261009/query_mask_assignment_source_20261011/'
names=json.loads((root/'QUERY_MASK_ASSIGNMENT_PUBLIC_FILE_LIST.json').read_bytes())+['QUERY_MASK_ASSIGNMENT_PUBLIC_FILE_LIST.json','publish_query_mask_assignment_authorized_20261011.py']
files={prefix+name:(root/name).read_bytes() for name in sorted(set(names))}
files[prefix+'.gitattributes']=b'** -text whitespace=-blank-at-eol,-space-before-tab,cr-at-eol,-blank-at-eof\\n'
assert not any('STDERR' in name or '.aris' in name or 'run_mcln_authorized_20260908.py' in name for name in files)
file_hashes={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}
assert all(not (repo/prefix).exists() for repo in repos[:2])
'''
template = (root / 'publish_matcher_mask_role_authorized_20261011.py').read_text(encoding='utf-8')
tail = template[template.index("remote_code = r'''import base64,hashlib,json,sys"):]
for old, new in (
        ('matcher_mask_role_cpu_20261011', 'query_mask_assignment_source_20261011'),
        ('tmp_matcher_mask_role', 'tmp_query_mask_assignment'),
        ('MATCHER_MASK_ROLE_PUBLICATION', 'QUERY_MASK_ASSIGNMENT_PUBLICATION'),
        ('matcher_mask_role_local_commit', 'query_mask_assignment_source_local_commit'),
        ('matcher_mask_role_publication', 'query_mask_assignment_source_publication'),
        ('MATCHER_MASK_ROLE_ALL_', 'QUERY_MASK_ASSIGNMENT_ALL_'),
        ('20.376.147', '20.376.148'),
        ('Document native matcher Mask evidence and actual CPU behavior', 'Prepare native Query Mask assignment source control')):
    assert old in tail, old
    tail = tail.replace(old, new)
content = head + tail
ast.parse(content, feature_version=(3, 7))
path = root / 'publish_query_mask_assignment_authorized_20261011.py'
assert not path.exists()
path.write_text(content, encoding='utf-8')
print(json.dumps(dict(status='QUERY_MASK_ASSIGNMENT_SOURCE_PUBLICATION_PREPARED', files=len(names),
    actual_CPU_check_completed=False, new_training_queries=0)))
