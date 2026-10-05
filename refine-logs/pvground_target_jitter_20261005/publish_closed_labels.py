"""Publish a closed bounded CPU label check, without training or checkpoint I/O."""
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
assert previous['section'] == '20.376.67'
assert not (root / 'terminal_publication.json').exists()
summary = json.loads((root / 'analysis/SUMMARY.json').read_bytes())
audit = json.loads((root / 'analysis/EXPERIMENT_AUDIT.json').read_bytes())
call = json.loads((root / 'analysis/TERMINAL_REVIEW_CALL.json').read_bytes())
wait = json.loads((root / 'wait.json').read_bytes())
receipt = json.loads((root / 'complete/receipt.json').read_bytes())
assert call['result_received'] and audit['verdict'] in ('PASS', 'WARN') and not audit['blocking_findings']
assert wait['observer_closed'] and wait['exitcode'] == 0 and receipt['status'] == 'complete'
assert summary['native_inputs_exact'] and summary['rows'] == 64 and not summary['accuracy_result']
assert receipt['model_forwards'] == receipt['weights_loaded'] == receipt['optimizer_steps'] == 0
assert state['retention_executed'] and state['protected_best_hits'] == [5614, 4509]
repos = [Path(r'C:\Users\gb\.codex_mcln_g0_20260905'), Path(r'C:\Users\gb\.codex_pvground_cs_20261002'),
         Path(r'C:\Users\gb\.codex_mcln_v99_internal_20260928')]
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies = [repo / doc for repo in repos] + [Path(r'C:\Users\gb\Desktop\document') / Path(doc).name]
old = copies[0].read_bytes()
assert hashlib.sha256(old).hexdigest() == previous['handoff_sha256']
assert all(path.read_bytes() == old for path in copies) and b'## 20.376.68 ' not in old


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args]).decode().strip()


for repo, head in zip(repos, previous['heads']):
    assert git(repo, 'rev-parse', 'HEAD') == head and not git(repo, 'status', '--porcelain')
stamp = datetime.datetime.now().astimezone().isoformat()
targets = summary['targets']
noisy = targets['native_noisy']
clean = targets['pre_jitter']
table = []
for label, target in (('原生扰动GT', noisy), ('扰动前成员标注GT', clean)):
    group = [target['arms'][arm]['fixed_noisy_qualified'] for arm in ('parent', 'control', 'query_supported')]
    table.append('| {} | {} | {} | {} | {:.6f} → {:.6f} | {} |'.format(label,
        *[item['hits50'] for item in group], group[0]['mean_iou'], group[2]['mean_iou'],
        target['fixed_noisy_cohort_outside_candidates']))
jitter = summary['target_jitter']
section = f'''

## 20.376.68 原生GT框扰动的小范围CPU核对完成；保持正式成绩与最佳权重（{stamp}）

承接§20.376.67。新增实验只重放数据，不导入模型、不加载权重、不初始化CUDA、不前向或更新优化器。原生随机调用和返回字段保持不变；从已经共同增强过的原生scan表示中的标注成员，额外取得独立6维框扰动之前的root框。这里的scan是原生50000点表示，不宣称原始全分辨率点云；框来自原生标注成员查找，不由模型Mask构造。固定64条fit表达、60个场景前缀、16384个原缓存候选；行顺序、scan ID、采样点字节、扰动后原生GT及有效root槽均精确复现。8批数据只读CPU重放实际结束于{receipt['finished_cst']}，退出0。预期以外的点／标签差异会停止核对，没有放宽匹配。

64个标注框的最大单面扰动：中位数{jitter['median_max_face_shift_m']:.6f}米，最大{jitter['max_face_shift_m']:.6f}米；超过1厘米的{jitter['rows_shift_over_1cm']}条，超过5厘米的{jitter['rows_shift_over_5cm']}条。Mask成员关系未增加此独立框扰动，但共同点增强仍保留。这里只移除最后独立GT框噪声作为离线反事实，未改变原生数据源或训练。

固定自身Query与融合Mask均>0.5且未匹配的支撑集合，共{summary['supported_unmatched_candidates']}个候选。按原生框目标，Box≤0.5资格为{noisy['qualification_candidates']}个；按扰动前成员框为{clean['qualification_candidates']}个。原资格中有{summary['qualification_removed_by_clean_target']}个因无独立框扰动而已过0.5，不再属于Box差；另有{summary['qualification_added_by_clean_target']}个新进入Box差资格。原生Hungarian匹配与缓存Mask支撑保持原值，未针对新框重算匹配。

下表始终使用原1090个资格候选，不能把两种资格集合或相关候选当成独立表达：

| 离线框目标 | 4506父头严格合格候选 | 同预算控制头 | 责任转移头 | 父头→策略头平均IoU | 固定资格中至少一面超原节点范围 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

当前正式9508的最优结果仍为5614／4509，即59.0450%／47.4232%，不是本64条离线重算；与原4506比较严格+3、宽松−2。保留策略完整10状态几何头、原G／官方PV和受保护V99链，两个非最佳头已删除的11,173,002字节不重复计算。此次新增权重和优化步均为0。

此核对证明的是独立GT框扰动对固定训练面板的资格、几何目标和范围统计的影响。它不证明原生增强是bug，不证明扰动造成原4506父头正式964条Mask好／Box差或全部方法失败；当前保留4509策略头对应961条，两者不混用。此处也未复算未知边界logit对应的DFL。原生9508开发验证和6887评估仍关闭augment／augment_det，评价目标未修改。不能把离线换目标后的命中当成模型涨点。

后续须据此结果判断是否值得单独比较辅助定位目标的坐标定义；空间参考／渐进六面结构仍为后续候选，不与监督目标变化同时加入。保持全部256候选、唯一原生last/bbs与同Query Box／Mask；不恢复普通几何回读、双源排名、V99部署侧链或无界延训。ScanRefer主目标仍未达到，Nr3D／Sr3D新结构结果仍不存在。

来源审查与终态审查均为实际调用的fresh Codex reviewer，requested Astra/max，后端未独立见证，按same-family/provisional披露。终态{audit['verdict']}，阻塞项0。原始64行框、原缓存数组引用、CPU重算、审查和回执见refine-logs/pvground_target_jitter_20261005/。
'''
new = old + section.encode('utf-8')
assert '\ufffd' not in section
prefix = 'refine-logs/pvground_target_jitter_20261005/'
payloads = {}
for file in root.rglob('*'):
    if file.is_file() and file.suffix in ('.py', '.json', '.md', '.jsonl', '.log', '.exit'):
        relative = file.relative_to(root).as_posix()
        assert not relative.startswith('__pycache__/') and not file.name.endswith(('.pth', '.pt'))
        payloads[prefix + relative] = file.read_bytes()
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
    password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
project = '/home/gb/new butd/butd_detr-main/MCLN-main'
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
sftp.close()
client.close()
heads = []
for index, repo in enumerate(repos):
    stage = [doc] + (list(payloads) if index < 2 else [])
    if index < 2:
        with (repo / 'MANIFEST.md').open('a', encoding='utf-8') as stream:
            stream.write('\n- ' + stamp + ' CPU64 native GT-jitter check closed and reviewed; exact native inputs, no model/checkpoint/optimizer. Best5614/4509 unchanged.\n')
        stage.append('MANIFEST.md')
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage], stderr=subprocess.DEVNULL)
    subprocess.check_call(['git', '-C', str(repo), 'diff', '--cached', '--check', '--', *stage])
    staged = subprocess.check_output(['git', '-C', str(repo), 'show', ':' + doc])
    original = subprocess.check_output(['git', '-C', str(repo), 'show', previous['heads'][index] + ':' + doc])
    assert staged.startswith(original) and staged.count(b'## 20.376.68 ') == 1
    subprocess.check_call(['git', '-C', str(repo), 'commit', '--quiet', '-m', 'Record bounded native target-jitter diagnostic'])
    heads.append(git(repo, 'rev-parse', 'HEAD'))
    assert not git(repo, 'status', '--porcelain')
subprocess.check_call(['git', '-C', str(repos[0]), '-c', 'http.version=HTTP/1.1', 'push', 'origin', 'HEAD:main'])
assert git(repos[0], 'ls-remote', 'origin', 'refs/heads/main').split()[0] == heads[0]
digest = hashlib.sha256(new).hexdigest()
guard = Path(r'C:\Users\gb\.codex\tmp\sync_cs_handoff_remote_20260923.py')
raw = guard.read_bytes()
assert raw.count(previous['handoff_sha256'].encode()) == 1
guard.write_bytes(raw.replace(previous['handoff_sha256'].encode(), digest.encode()))
assert all(path.read_bytes() == new for path in copies)
record = dict(previous)
record.update(time_cst=stamp, section='20.376.68', heads=heads, github_main=heads[0], handoff_bytes=len(new),
    handoff_sha256=digest, four_local_and_remote_equal=True, payload_count=len(payloads),
    status='ACTUAL_CLOSED_CPU_LABEL_CHECK_PUBLISHED', new_accuracy_result=False,
    diagnostic_CPU_executed=True, diagnostic_GPU_executed=False,
    diagnostic_finished_cst=receipt['finished_cst'], diagnostic_integrity_verdict=audit['verdict'])
(root / 'terminal_publication.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
state.update(latest_publication=str(root / 'terminal_publication.json'), published_heads=heads,
    handoff_section='20.376.68', handoff_sha256=digest, status='CLOSED_CPU_LABEL_CHECK_PUBLISHED',
    label_diagnostic_closed=True, next_action='Review measured target-jitter effect before a controlled auxiliary-target or support-reference change',
    current_goal_turn_classification='PROGRESS_CPU_TARGET_LABEL_CHECK_PUBLISHED')
state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
with Path(r'C:\Users\gb\memory\2026-10-05.md').open('a', encoding='utf-8') as stream:
    stream.write('\n\nPVGround ' + stamp + ': Doc68 CPU64 target-jitter counterfactual closed/published; exact points/native GT, no model/weights/optimizer. Noisy/clean qualifications ' + str(noisy['qualification_candidates']) + '/' + str(clean['qualification_candidates']) + ', removed/added ' + str(summary['qualification_removed_by_clean_target']) + '/' + str(summary['qualification_added_by_clean_target']) + '. Not formal accuracy or cause proof. Best5614/4509 unchanged. Main ' + heads[0] + '. ACTIVE_UNMET.\n')
print(json.dumps({key: record[key] for key in ('time_cst', 'section', 'heads', 'handoff_bytes', 'handoff_sha256',
    'four_local_and_remote_equal', 'diagnostic_integrity_verdict', 'new_accuracy_result')}), flush=True)
