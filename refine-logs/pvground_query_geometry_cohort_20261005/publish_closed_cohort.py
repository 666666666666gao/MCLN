"""Publish the closed fixed-cohort evidence and its bounded next action."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

import paramiko

root = Path(__file__).resolve().parent
study = root.parent / 'pvground_query_supported_geometry_20261005'
state_path = study / 'active_continuation_state.json'
state = json.loads(state_path.read_bytes())
previous = json.loads(Path(state['latest_publication']).read_bytes())
assert previous['section'] == '20.376.66'
assert not (root / 'terminal_publication.json').exists()
summary = json.loads((root / 'analysis/SUMMARY.json').read_bytes())
audit = json.loads((root / 'analysis/EXPERIMENT_AUDIT.json').read_bytes())
call = json.loads((root / 'analysis/TERMINAL_REVIEW_CALL.json').read_bytes())
wait = json.loads((root / 'wait.json').read_bytes())
receipt = json.loads((root / 'complete/receipt.json').read_bytes())
assert call['result_received'] and audit['verdict'] in ('PASS', 'WARN') and not audit['blocking_findings']
assert wait['observer_closed'] and not wait['controller_alive'] and wait['exitcode'] == 0
assert receipt['model_state_restored'] and receipt['optimizer_steps'] == 0 and receipt['weight_files_created'] == 0
assert summary['rows'] == 64 and summary['candidates_per_row'] == 256 and not summary['accuracy_result']
assert state['retention_executed'] and state['protected_best_hits'] == [5614, 4509]
repos = [Path(r'C:\Users\gb\.codex_mcln_g0_20260905'), Path(r'C:\Users\gb\.codex_pvground_cs_20261002'),
         Path(r'C:\Users\gb\.codex_mcln_v99_internal_20260928')]
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies = [repo / doc for repo in repos] + [Path(r'C:\Users\gb\Desktop\document') / Path(doc).name]
old = copies[0].read_bytes()
assert hashlib.sha256(old).hexdigest() == previous['handoff_sha256']
assert all(path.read_bytes() == old for path in copies) and b'## 20.376.67 ' not in old


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args]).decode().strip()


for repo, head in zip(repos, previous['heads']):
    assert git(repo, 'rev-parse', 'HEAD') == head and not git(repo, 'status', '--porcelain')
stamp = datetime.datetime.now().astimezone().isoformat()
groups = ('parent_qualified', 'qualified_inside', 'qualified_outside', 'parent_box_good', 'selected_query')
labels = ('固定父资格集合', '资格集合、目标在节点范围内', '资格集合、至少一面在范围外', '父模型原本严格合格的框', '原生固定已选Query')
lines = []
for group, label in zip(groups, labels):
    values = [summary['arms'][arm]['groups'][group] for arm in ('parent', 'control', 'query_supported')]
    lines.append('| {} | {} | {} | {} | {} | {:.6f} → {:.6f} |'.format(
        label, values[0]['candidates'], *[value['hits50'] for value in values],
        values[0]['mean_iou'], values[2]['mean_iou']))
qualified = [summary['arms'][arm]['groups']['parent_qualified'] for arm in ('parent', 'control', 'query_supported')]
transition = summary['paired']['control_to_query_supported']['parent_qualified']['0.5']
prefix = 'refine-logs/pvground_query_geometry_cohort_20261005/'
section = f'''

## 20.376.67 固定父候选的三头几何回放完成：DFL下降不等于广泛改善；下一步核对支撑与监督框坐标（{stamp}）

承接§20.376.66的正式9508结果与最佳4509权重。本节不是新增正式精度：同一64条增强fit输入、60个场景前缀，共16384个候选，父模型资格固定为自身Query Mask和融合Mask都>0.5、父最终框≤0.5、且没有任何原生末层匹配。1090个资格候选分布在52条表达中；其中889个在原节点范围内，201个至少一面在范围外，共482个范围外面。候选之间相关，GT支撑和重叠仍是训练代理，不是物理身份的无条件真值。

实际只读GPU诊断于 {wait['status']['finished_cst']} 完成，退出0；8次完整父前向、24次缓存几何头回放，原生语义头每批仅一次。三头使用同一Query、粗框、原始点、Mask和分数；父缓存框逐元素复现，最后恢复父模型全部状态，无梯度、优化更新或新增权重。实际收集24份原始产物2632409字节，含8份NPZ。CPU对IoU、面误差、位移、资格和阈值独立重算；资格及CPU/GPU阈值差异均为0。GPU DFL标量未独立CPU重算。

| 相同父集合／相同已选Query | 候选数 | 父头严格合格 | 控制头严格合格 | 策略头严格合格 | 父→策略平均IoU |
|---|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

资格集合相对控制，策略严格修复{transition['repairs']}、破坏{transition['damages']}，净{transition['net']:+d}个候选，不是新增表达命中。其GPU DFL均值为父{qualified[0]['GPU_DFL_mean']:.6f}、控制{qualified[1]['GPU_DFL_mean']:.6f}、策略{qualified[2]['GPU_DFL_mean']:.6f}；但父→策略平均IoU下降，最大单面误差中位数 {qualified[0]['median_max_face_error_m']:.6f}→{qualified[2]['median_max_face_error_m']:.6f}米，单面移动中位数 {qualified[0]['median_max_face_move_m']:.6f}→{qualified[2]['median_max_face_move_m']:.6f}米。固定已选64条严格命中均为42，10个已选资格候选均未跨过0.5。这个面板不支持“补定位责任已普遍修准候选”，也不能凭DFL下降把不足归为最终评分问题。范围外201个仍无严格合格框；范围内候选同样没有平均改善，因此不能只用节点范围解释全部不足。

审计进一步确认原生训练目标的坐标关系：实际同SHA的joint_det_dataset.py:1105–1113先从已增强scan.get_object_bbox取得框，转为中心/尺寸；仅split=train且augment=True时，再对6维乘0.95+0.1*rand。gt_masks保持对象点成员关系，不接受这项额外框噪声。全场景all_bboxes另在1152–1153独立扰动，两阶段GroupFree对象输入还存在单独augment_det路径，不能混作同一张量。本次fit及64增强诊断启用这些原生条件；正式9508在split=val且evaluate将augment/augment_det设False，6887模块留出也在创建loader前关闭它们，所以此事实不改变正式结果的GT协议。它是原生训练协议，当前未修改，尚不能证明它造成了多少下降。

下一项先做小范围CPU数据核对：在保持相同64条、随机顺序、点云和原始监督张量不变的条件下，读出加框噪声前的原生标注框，再在已保存父候选上重算资格、IoU与范围外比例。该框来自数据集标注成员和增强坐标，不能由模型输出生成。先分清Mask支撑与监督框的几何差异，再决定是否单独比较辅助定位目标或自身支撑参考；不立刻增加普通注意力、R/质量排序、教师或放大节点范围。自身支撑参考、渐进精修仍是待验证方向，当前没有新增结构通过正式验收。

fresh终态审计为{audit['verdict']}、0阻断，确定性核验通过；同族provisional、requested Astra/max、backend未attested。435个汇总标量及184份文件核对记录见 {prefix}analysis/EXPERIMENT_AUDIT.md/JSON。逐行输入、点和GT与原面板相同，但跨CUDA完整前向存在真实小幅数值漂移；三头同批缓存精确比较不等于跨进程逐位相同。只支持单seed、增强fit64的候选诊断，不支持未见场景、Nr/Sr、显著性或完整三贡献结论。

本诊断已在§20.376.66权重清理前完成。旧4506与本轮控制几何头已删除，不新建负权重归档；历史输入路径仅作为已完成诊断的来源记录。当前唯一最佳完整几何头为5614/4509，官方PV、原G为必要重建父权重，V99完整链保留。三数据集总目标仍ACTIVE_UNMET。
'''
new = old + section.encode('utf-8')
assert new.startswith(old) and new.count(b'## 20.376.67 ') == 1
names = ['spec.json', 'SOURCE_REVIEW.json', 'SOURCE_REVIEW.md', 'FINAL_SOURCE_REVIEW_CALL.json',
         'launch.json', 'resource_check.json', 'wait.json', Path(__file__).name]
for folder in ('complete', 'analysis', 'observations'):
    names.extend(str(path.relative_to(root)).replace('\\', '/') for path in sorted((root / folder).rglob('*')) if path.is_file())
assert all(not name.endswith(('.pth', '.pt')) for name in names)
payloads = {prefix + name: (root / name).read_bytes() for name in names}
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
project = '/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(project + '/' + doc, 'rb') as stream:
    assert stream.read() == old
evidence = project + '/' + prefix.rstrip('/')
assert 'complete' not in sftp.listdir(evidence)
sftp.symlink('/root/autodl-tmp/pvground_query_geometry_cohort_20261005', evidence + '/complete')
assert sftp.readlink(evidence + '/complete') == '/root/autodl-tmp/pvground_query_geometry_cohort_20261005'
for name, raw in payloads.items():
    for repo in repos[:2]:
        destination = repo / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    parent, _, basename = name.rpartition('/')
    parts = parent.split('/')
    for index in range(1, len(parts) + 1):
        folder = '/'.join(parts[:index]); upper, _, leaf = folder.rpartition('/')
        if leaf not in sftp.listdir(project + '/' + upper):
            sftp.mkdir(project + '/' + folder)
    if not name.startswith(prefix + 'complete/') or name == prefix + 'complete/INTAKE.json':
        with sftp.open(project + '/' + name, 'wb') as stream:
            stream.write(raw)
    with sftp.open(project + '/' + name, 'rb') as stream:
        assert stream.read() == raw
for path in copies:
    path.write_bytes(new)
with sftp.open(project + '/' + doc, 'wb') as stream:
    stream.write(new)
with sftp.open(project + '/' + doc, 'rb') as stream:
    assert stream.read() == new
sftp.close(); client.close()
heads = []
for index, repo in enumerate(repos):
    stage = [doc] + (list(payloads) if index < 2 else [])
    if index < 2:
        with (repo / 'MANIFEST.md').open('a', encoding='utf-8') as stream:
            stream.write('\n- ' + stamp + ' Fixed-cohort three-head geometry replay closed and audited; 64 augmented fit rows, no optimizer or new weights. Native training bbox-noise scope recorded; next CPU pre-jitter target comparison.\n')
        stage.append('MANIFEST.md')
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage], stderr=subprocess.DEVNULL)
    subprocess.check_call(['git', '-C', str(repo), 'diff', '--cached', '--check', '--', *stage])
    staged_doc = subprocess.check_output(['git', '-C', str(repo), 'show', ':' + doc])
    original_doc = subprocess.check_output(['git', '-C', str(repo), 'show', previous['heads'][index] + ':' + doc])
    assert staged_doc.startswith(original_doc) and staged_doc.count(b'## 20.376.67 ') == 1
    subprocess.check_call(['git', '-C', str(repo), 'commit', '--quiet', '-m', 'Record fixed-cohort geometry outcomes and native target-noise scope'])
    heads.append(git(repo, 'rev-parse', 'HEAD'))
    assert not git(repo, 'status', '--porcelain')
subprocess.check_call(['git', '-C', str(repos[0]), '-c', 'http.version=HTTP/1.1', 'push', 'origin', 'HEAD:main'])
assert git(repos[0], 'ls-remote', 'origin', 'refs/heads/main').split()[0] == heads[0]
digest = hashlib.sha256(new).hexdigest()
guard = Path(r'C:\Users\gb\.codex\tmp\sync_cs_handoff_remote_20260923.py')
raw = guard.read_bytes(); assert raw.count(previous['handoff_sha256'].encode()) == 1
guard.write_bytes(raw.replace(previous['handoff_sha256'].encode(), digest.encode()))
assert all(path.read_bytes() == new for path in copies)
record = dict(previous)
record.update(time_cst=stamp, section='20.376.67', heads=heads, github_main=heads[0], handoff_bytes=len(new),
              handoff_sha256=digest, four_local_and_remote_equal=True, payload_count=len(payloads),
              status='ACTUAL_CLOSED_COHORT_PUBLISHED', new_accuracy_result=False, diagnostic_GPU_executed=True,
              diagnostic_finished_cst=wait['status']['finished_cst'], diagnostic_integrity_verdict=audit['verdict'])
(root / 'terminal_publication.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
state.update(latest_publication=str(root / 'terminal_publication.json'), published_heads=heads,
             handoff_section='20.376.67', handoff_sha256=digest, status='CLOSED_COHORT_PUBLISHED_NEXT_LABEL_CHECK',
             cohort_terminal_published=True, next_action='CPU native pre-jitter target comparison on fixed64',
             current_goal_turn_classification='PROGRESS_FORMAL_RETENTION_AND_COHORT_PUBLISHED')
state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
with Path(r'C:\Users\gb\memory\2026-10-05.md').open('a', encoding='utf-8') as stream:
    stream.write('\n\nPVGround ' + stamp + ': Doc67 actual fixed64/16384 three-head diagnostic published; 1090 parent-qualified, strict parent/control/strategy0/3/5, mean IoU parent0.273640/strategy0.272432. Outside201 strict0; selected64 all42. WARN/0 blocker; target bbox jitter only augmented fit, not formal/holdout. Next CPU pre-jitter target comparison, not another attention or range-scale change. Raw4local+remote equal, main ' + heads[0] + ', best5614/4509 retained after two nonbest heads deleted. ACTIVE_UNMET.\n')
print(json.dumps({key: record[key] for key in ('time_cst', 'section', 'heads', 'handoff_bytes', 'handoff_sha256',
      'four_local_and_remote_equal', 'diagnostic_integrity_verdict', 'new_accuracy_result')}), flush=True)
