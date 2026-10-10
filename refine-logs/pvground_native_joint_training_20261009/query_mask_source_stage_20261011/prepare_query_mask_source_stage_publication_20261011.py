"""Prepare publication of the actual isolated source copy, without ML execution."""
import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parent
data = root / 'query_mask_assignment_gpu_20261011'
summary = json.loads((data / 'SOURCE_STAGING_SUMMARY.json').read_bytes())
assert summary['source_file_count'] == 116
assert summary['source_bytes'] == 24660643
assert summary['GPU_preflight_executed'] is False
assert summary['normal_training_started'] is False
notes = '''## §20.376.152｜Query自身Mask匹配版本已完成独立源码部署；尚未运行GPU预检或新训练（2026-10-11）

### 本次实际完成

05:27:12.009144，通过现有授权通道执行了一次独立源码部署，返回退出码0。目的目录为`/root/autodl-tmp/pvground_query_mask_assignment_20261011/PV-Ground`，没有修改正在运行的C关闭训练目录。复制绑定清单中的116份原生源文件及已有编译扩展，覆盖14份此前已审查的变体文件；最终共24,660,643字节，每份原始、覆盖和最终文件均校验SHA256。相对绑定原生父源码，最终只改变`main_utils.py`、`models/losses.py`两份文件。

同时部署真实批次预检脚本、初始化清单及text/query两份配置，仅匹配Mask来源不同。远端以现有Python3.7做源码AST解析，没有导入项目、Torch、模型或loader；没有GPU查询、神经前向、反向、优化器更新、训练状态读取或新增控制器。没有安装包或重编译扩展，也没有复制或删除权重。

部署前的独立源码审查核对全部24份输入SHA，source correctness为PASS、总体WARN、阻断项与要求修改均为空。总体WARN保留共享运行环境、尚未执行真实模型以及审查身份未独立证实的限制；验收为same-family/provisional，不写成跨模型独立证明。本次实际传输与文件校验由部署回执记录，不由静态审查替代。

### 尚未完成及训练职责边界

这116份文件是有明确绑定的部分原生部署，需要既有共享运行环境和数据根；不是一个包含全部依赖的独立仓库。新目录的实际import来源、模型构建、编译扩展运行、完整criterion、优化器保存恢复及显存吞吐仍要通过真实预检确认。`serial_gpu_preflight_admitted`与`full_training_admitted`均保持false，因此本次部署不代表GPU预检完成或新训练启动，更没有新精度结果。

已部署变体把最后层匹配的Mask来源配置为候选自己的Query Mask。该变化属于训练criterion，模型前向、256候选、唯一原生bbs和推理输入不因此增加另一套答案。原Hungarian匹配改变后，依赖它的语义、Mask、回归与G责任也可能一起改变，不能把未来差值称为只有框监督发生变化。现有CPU合成检查只证明实现的对应关系和梯度，真实50000点成本尺度与候选变化仍未测出。

### 当前主训练与后续次序

当前C关闭的第3轮保持原样；截至05:04:55的完整结果已记于§20.376.151，不重复表格。保留最好仍为已有E0的5677／4920（59.7076%／51.7459%），不是正常续训的新收益，三个有效创新和同一完整模型Nr3D／Sr3D结果尚未成立。

本地05:29:13核对原唯一观察器PID28804仍在，创建时间05:08:04.924229，与已有35461任务一致；没有重建等待器。下一次主训练查询仍为09:07:13.156040，预计完整第3轮验证结束09:12:13；估计如有偏差，根据实际进度按240秒安排补查，不提前查询。

先收取并核验C关闭三轮终态，再决定真实预检与后续正常训练。若执行Query匹配预检，沿用原生train_one_epoch及完整criterion检查两个真实batch，核对实际匹配变化、原父1295状态、全损失、保存恢复与运行依赖。预检通过仍不是精度收益；进入训练后还要同起点、同协议、同预算比较，并同时判断是否超过保留起点及各机制是否有独立价值。当前不盲目切作者混合检测配方或Nr/Sr接口，不恢复双源排名，不丢弃低分候选，不做多seed。

本次源码部署及同步均不改变活动训练、模型选模规则或保护权重。当前完整目标保持ACTIVE_UNMET。
'''
(data / 'HANDOFF_SOURCE_STAGE_20261011.md').write_text(notes, encoding='utf-8')
names = [
    'query_mask_assignment_gpu_20261011/HANDOFF_SOURCE_STAGE_20261011.md',
    'query_mask_assignment_gpu_20261011/SOURCE_STAGE_RECEIPT.json',
    'query_mask_assignment_gpu_20261011/SOURCE_STAGE_STDOUT.json',
    'query_mask_assignment_gpu_20261011/SOURCE_STAGE_EXIT.json',
    'query_mask_assignment_gpu_20261011/SOURCE_STAGING_SUMMARY.json',
    'query_mask_assignment_gpu_20261011/SOURCE_STAGING_PLAN.md',
    'query_mask_assignment_gpu_20261011/SOURCE_STAGE_REVIEW_INPUTS.json',
    'query_mask_assignment_gpu_20261011/source_stage_review/SOURCE_REVIEW.md',
    'query_mask_assignment_gpu_20261011/source_stage_review/SOURCE_REVIEW.json',
    'query_mask_assignment_gpu_20261011/stage_native_source_authorized_20261011.py',
    'query_mask_assignment_gpu_20261011/record_source_stage_20261011.py',
    'NATIVE_SOURCE_PORT.json',
    'prepare_query_mask_source_stage_publication_20261011.py',
]
for name in names:
    assert (root / name).is_file(), name
(root / 'QUERY_MASK_SOURCE_STAGE_PUBLIC_FILE_LIST.json').write_text(json.dumps(names, indent=2) + '\n', encoding='utf-8')
head = '''"""Publish an actual isolated source staging receipt; no ML or training query."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

root=Path(__file__).resolve().parent
prior=json.loads((root/'c_off_epoch2_publication_20261011.json').read_bytes())
assert prior['section']=='20.376.151' and prior['remote_handoff_sync_complete'] and prior['github_main_verified']
prior_receipt=json.loads(Path(prior['remote_sync_receipt']).read_bytes())
assert prior_receipt['remote_sync_complete'] and prior_receipt['doc_sha256']==prior['doc_sha256']
assert not (root/'query_mask_source_stage_publication_20261011.json').exists()
data=root/'query_mask_assignment_gpu_20261011'
summary=json.loads((data/'SOURCE_STAGING_SUMMARY.json').read_bytes())
assert summary['status']=='BOUND_NATIVE_QUERY_MASK_MATCHER_SOURCE_STAGED_NOT_EXECUTED'
assert summary['GPU_preflight_executed'] is False and summary['normal_training_started'] is False
assert summary['source_stage_receipt_sha256']==hashlib.sha256((data/'SOURCE_STAGE_RECEIPT.json').read_bytes()).hexdigest()
assert summary['source_file_count']==116 and summary['source_bytes']==24660643
assert summary['changed_parent_source_files']==['main_utils.py','models/losses.py']
plan=json.loads((root/'normal_controls_20261010/NORMAL_EPOCH3_OBSERVATION_PLAN.json').read_bytes())
owner=json.loads((root/'normal_controls_20261010/NORMAL_EPOCH3_OBSERVER_OWNER.json').read_bytes())
assert owner['local_pid']==28804 and owner['first_due_cst']==plan['due_cst']==summary['next_main_observation_cst']
repos=[Path('C:/Users/gb')/name for name in ('.codex_mcln_g0_20260905','.codex_pvground_cs_20261002','.codex_mcln_v99_internal_20260928')]
doc='docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop=Path('C:/Users/gb/Desktop/document')/Path(doc).name
old=(repos[0]/doc).read_bytes()
assert hashlib.sha256(old).hexdigest()==prior['doc_sha256'] and b'20.376.152' not in old
assert all((repo/doc).read_bytes()==old for repo in repos) and desktop.read_bytes()==old
for index,repo in enumerate(repos):
 assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==prior['heads'][index]
 assert not subprocess.check_output(['git','-C',str(repo),'-c','core.longpaths=true','status','--porcelain']).strip()
notes=(data/'HANDOFF_SOURCE_STAGE_20261011.md').read_text(encoding='utf-8')
document=old+('\\n\\n'+notes.replace('\\r\\n','\\n')).replace('\\n','\\r\\n').encode('utf-8')
digest=hashlib.sha256(document).hexdigest()
prefix='refine-logs/pvground_native_joint_training_20261009/query_mask_source_stage_20261011/'
names=json.loads((root/'QUERY_MASK_SOURCE_STAGE_PUBLIC_FILE_LIST.json').read_bytes())+['QUERY_MASK_SOURCE_STAGE_PUBLIC_FILE_LIST.json','publish_query_mask_source_stage_authorized_20261011.py']
files={prefix+name:(root/name).read_bytes() for name in sorted(set(names))}
files[prefix+'.gitattributes']=b'** -text whitespace=-blank-at-eol,-space-before-tab,cr-at-eol,-blank-at-eof\\n'
assert not any('STDERR' in name or '.aris' in name or 'run_mcln_authorized_20260908.py' in name for name in files)
file_hashes={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}
assert all(not (repo/prefix).exists() for repo in repos[:2])
'''
template = (root / 'publish_c_off_epoch2_authorized_20261011.py').read_text(encoding='utf-8')
tail = template[template.index("remote_code = r'''import base64,hashlib,json,sys"):]
for before, after in (
    ('c_off_epoch2_20261011', 'query_mask_source_stage_20261011'),
    ('tmp_c_off_epoch2', 'tmp_query_mask_source_stage'),
    ('C_OFF_EPOCH2_PUBLICATION', 'QUERY_MASK_SOURCE_STAGE_PUBLICATION'),
    ('c_off_epoch2_local_commit_20261011', 'query_mask_source_stage_local_commit_20261011'),
    ('c_off_epoch2_publication_20261011', 'query_mask_source_stage_publication_20261011'),
    ('C_OFF_EPOCH2_ALL_', 'QUERY_MASK_SOURCE_STAGE_ALL_'),
    ('20.376.151', '20.376.152'),
    ('Record second C-off native joint epoch and scheduled final observation', 'Record isolated native Query Mask matcher source staging')):
    assert before in tail, before
    tail = tail.replace(before, after)
content = head + tail
ast.parse(content)
path = root / 'publish_query_mask_source_stage_authorized_20261011.py'
assert not path.exists()
path.write_text(content, encoding='utf-8')
print(json.dumps({'status': 'QUERY_MASK_SOURCE_STAGE_PUBLICATION_PREPARED', 'section': '20.376.152', 'GPU_preflight_executed': False, 'new_training_started': False, 'next_observation_cst': summary['next_main_observation_cst']}))
