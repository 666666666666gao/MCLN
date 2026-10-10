"""Prepare a static handoff append only after the controlled CPU check is audited."""
import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parent
data = root / 'matcher_mask_role_cpu_20261011'
summary = json.loads((data / 'ACTUAL_CHECK_SUMMARY.json').read_bytes())
assert summary['status'] == 'ACTUAL_NATIVE_MATCHER_CPU_CHECK_AUDITED'
assert summary['fixture_count'] == summary['matcher_calls'] == 9
assert summary['formal_accuracy'] is None and summary['gpu_calls'] == 0
notes = '''## §20.376.147 — 原生匹配中的公共Text Mask成本与候选自身支撑

等待正常C-off第2轮既定观察期间，完成一次限定的源码与实际原生matcher CPU检查。没有重新读取活动训练日志或GPU状态，没有改网络、criterion、优化器、数据或当前配方。

当前正常PV源码在选取Text Mask后，将其expand给全部256个Query；候选自身Query Mask另存于sp_last_pred_masks。末层criterion同时收到两路Mask，但HungarianMatcher的直接Mask成本只读取公共Text Mask，做硬二值判断、原始点映射和L1距离；cost_masks系数0.0002。实际criterion构建仍为分类1、框L1为0、GIoU为2。

因此，固定候选框和token分数时，直接Mask成本在每一个GT列内是相同常量。原生接口最多132个有效GT，候选为256，全部有效GT各匹配一次时，该项给全部完整分配增加相同的总成本，未提供区分候选自身支撑的依据。不能把这一点写成“原生没有Mask成本”。

实际CPU检查从当前warm model_source加载未修改的models/losses.py；三份主源码、两份utils导入文件以及运行环境规范均按实际原生manifest完整SHA核对，并记录真实导入路径。使用原生Torch1.10.2+cu111及SciPy assignment，CUDA隐藏、单CPU线程，不执行PV网络、criterion、真实数据或优化器。

九组受控输入覆盖单GT、两GT和混合batch，均为256个Query、256个token、四个合成点。固定框和token logits，替换候选自身Query Mask后，实际原生成本矩阵完全相同，分配不变；更换公共Text Mask或去掉Mask键时，其成本差为GT列常量，保留浮点舍入量摘要。完整矩阵仅在内存捕获，回执保留差值摘要和分配，不声称保存完整矩阵数组。

这一结果属于simulation_only工程行为检查：九次matcher调用、零模型前向、零criterion调用、零优化器更新、零GPU执行、零真实表达，formal_accuracy=null。没有新的ScanRefer、Nr3D或Sr3D指标，不能把合成GT当作正式验证，也没有证明这是正常续训掉点的原因。

必须保留间接路径：候选自己的Query Mask会通过native_mask_geometry和candidate_span_mixer改变last_center/last_pred_size，最终框再进入GIoU匹配。故固定框的检查只隔离直接Mask成本，不能说整个模型的Query Mask不影响匹配。原生匹配索引随后同时用于定位、语义及Query Mask损失。

源码与实际检查均经新上下文Codex审查；同模型家族、provisional，实际模型身份/推理档位无外部凭证，不宣称跨家族或外部独立验收。报告与完整字节摘要保留，不把执行者自己的判断改写成独立审查。

后续先完成活动C-off三轮规定预算。若需要调整几何学习责任，优先用明确的同起点、同预算对照检验候选自身支撑参与末层训练分配的增量，保留真实有效GT、其他已匹配实例与联合检测行的职责；当前没有实施这一新策略。Mask参与匹配已有前作，这次发现不构成“首次Mask匹配”或第三个有效论文模块的证明。

保留最好5677/4920及所有必要父权重；Nr/Sr正式训练尚未启动。唯一原观察器52851/PID51540仍按2026年10月11日05:00:40检查正常第2轮。完整目标继续ACTIVE_UNMET。
'''
(data / 'HANDOFF_MATCHER_MASK_ROLE_20261011.md').write_text(notes, encoding='utf-8')
names = ['matcher_mask_role_cpu_20261011/' + name for name in (
    'check_native_matcher_cpu.py', 'run_authorized.py', 'CHECK_SPEC.json', 'SOURCE_FINDING.md',
    'source_review/SOURCE_REVIEW.md', 'source_review/SOURCE_REVIEW.json',
    'actual_review/ACTUAL_REVIEW.md', 'actual_review/ACTUAL_REVIEW.json',
    'cpu_execution/CPU_EXECUTION.json', 'cpu_execution/MATCHER_CPU_RESULT.json',
    'cpu_execution/CPU_STDOUT.txt', 'cpu_execution/TRANSPORT_EXIT.json',
    'CPU_HELPER_OWNER.json', 'summarize_actual_check.py',
    'ACTUAL_CHECK_SUMMARY.json', 'HANDOFF_MATCHER_MASK_ROLE_20261011.md')]
names += ['prepare_matcher_mask_role_publication_20261011.py', 'record_matcher_mask_role_publication_20261011.py']
(root / 'MATCHER_MASK_ROLE_PUBLIC_FILE_LIST.json').write_text(json.dumps(names, indent=2) + '\n')
head = '''"""Publish the audited native matcher CPU observation, with no training query."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

root=Path(__file__).resolve().parent
prior=json.loads((root/'retired_eg_cleanup_publication.json').read_bytes())
assert prior['section']=='20.376.146' and prior['remote_handoff_sync_complete'] and prior['github_main_verified']
prior_receipt=json.loads(Path(prior['remote_sync_receipt']).read_bytes())
assert prior_receipt['remote_sync_complete'] and prior_receipt['doc_sha256']==prior['doc_sha256']
assert not (root/'matcher_mask_role_publication.json').exists()
data=root/'matcher_mask_role_cpu_20261011'
summary=json.loads((data/'ACTUAL_CHECK_SUMMARY.json').read_bytes())
assert summary['status']=='ACTUAL_NATIVE_MATCHER_CPU_CHECK_AUDITED'
assert summary['fixture_count']==summary['matcher_calls']==9 and summary['gpu_calls']==0
assert summary['formal_accuracy'] is None and summary['current_training_queries']==0
for stage in ['source_review/SOURCE_REVIEW.json','actual_review/ACTUAL_REVIEW.json']:
 audit=json.loads((data/stage).read_bytes())
 assert audit['verdict'] in ['PASS','WARN'] and not audit['blocking_findings']
 for path,digest in audit['audited_input_hashes'].items():
  assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
plan=json.loads((root/'normal_controls_20261010/NORMAL_NEXT_OBSERVATION_PLAN.json').read_bytes())
owner=json.loads((root/'normal_controls_20261010/NORMAL_EPOCH2_AUTHORIZED_OBSERVER_OWNER.json').read_bytes())
assert owner['local_pid']==51540 and owner['first_due_cst']==plan['due_cst']
repos=[Path('C:/Users/gb')/name for name in ('.codex_mcln_g0_20260905','.codex_pvground_cs_20261002','.codex_mcln_v99_internal_20260928')]
doc='docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop=Path('C:/Users/gb/Desktop/document')/Path(doc).name
old=(repos[0]/doc).read_bytes()
assert hashlib.sha256(old).hexdigest()==prior['doc_sha256'] and b'20.376.147' not in old
assert all((repo/doc).read_bytes()==old for repo in repos) and desktop.read_bytes()==old
for index,repo in enumerate(repos):
 assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==prior['heads'][index]
 assert not subprocess.check_output(['git','-C',str(repo),'-c','core.longpaths=true','status','--porcelain']).strip()
notes=(data/'HANDOFF_MATCHER_MASK_ROLE_20261011.md').read_text(encoding='utf-8')
document=old+('\\n\\n'+notes.replace('\\r\\n','\\n')).replace('\\n','\\r\\n').encode('utf-8')
digest=hashlib.sha256(document).hexdigest()
prefix='refine-logs/pvground_native_joint_training_20261009/matcher_mask_role_cpu_20261011/'
names=json.loads((root/'MATCHER_MASK_ROLE_PUBLIC_FILE_LIST.json').read_bytes())+['MATCHER_MASK_ROLE_PUBLIC_FILE_LIST.json','publish_matcher_mask_role_authorized_20261011.py']
files={prefix+name:(root/name).read_bytes() for name in sorted(set(names))}
files[prefix+'.gitattributes']=b'** -text whitespace=-blank-at-eol,-space-before-tab,cr-at-eol,-blank-at-eof\\n'
assert not any('STDERR' in name or '.aris' in name or 'run_mcln_authorized_20260908.py' in name for name in files)
file_hashes={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}
assert all(not (repo/prefix).exists() for repo in repos[:2])
'''
template = (root / 'publish_retired_eg_cleanup_authorized_20261011.py').read_text(encoding='utf-8')
tail = template[template.index("remote_code = r'''import base64,hashlib,json,sys"):]
for old, new in (
        ('retired_eg_cleanup_20261011', 'matcher_mask_role_cpu_20261011'),
        ('tmp_retired_eg_cleanup', 'tmp_matcher_mask_role'),
        ('RETIRED_EG_CLEANUP_PUBLICATION', 'MATCHER_MASK_ROLE_PUBLICATION'),
        ('retired_eg_cleanup_local_commit', 'matcher_mask_role_local_commit'),
        ('retired_eg_cleanup_publication', 'matcher_mask_role_publication'),
        ('RETIRED_EG_CLEANUP_ALL_', 'MATCHER_MASK_ROLE_ALL_'),
        ('20.376.146', '20.376.147'),
        ('Archive and retire three closed EG candidate caches', 'Document native matcher Mask evidence and actual CPU behavior'),
        ('cleanup_result=summary', 'diagnostic_result=summary')):
    assert old in tail, old
    tail = tail.replace(old, new)
content = head + tail
ast.parse(content, feature_version=(3, 7))
path = root / 'publish_matcher_mask_role_authorized_20261011.py'
assert not path.exists()
path.write_text(content, encoding='utf-8')
print(json.dumps(dict(status='MATCHER_MASK_ROLE_STATIC_PUBLICATION_PREPARED', files=len(names), new_training_queries=0)))
