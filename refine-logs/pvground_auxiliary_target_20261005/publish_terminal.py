"""Append the actual closed comparison, audit and retention to all handoffs."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import paramiko

local = Path(__file__).resolve().parent
workspace = Path('C:/Users/gb')
assert not (local / 'terminal_publication.json').exists()
state_path = local.parent / 'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state = json.loads(state_path.read_bytes())
previous = json.loads(Path(state['latest_publication']).read_bytes())
summary = json.loads((local / 'analysis/SUMMARY.json').read_bytes())
audit = json.loads((local / 'analysis/EXPERIMENT_AUDIT.json').read_bytes())
review_call = json.loads((local / 'analysis/TERMINAL_REVIEW_CALL.json').read_bytes())
retention = json.loads((local / 'weight_retention.json').read_bytes())
resources = json.loads((local / 'CLOSED_RESOURCES.json').read_bytes())
wait = json.loads((local / 'fit_wait.json').read_bytes())
assert summary['status'] == 'ACTUAL_CLOSED_ROWS_ANALYZED' and summary['training_order_exact']
assert review_call['result_received'] and audit['verdict'] in ('PASS', 'WARN') and not audit['blocking_findings']
assert wait['observer_closed'] and wait['terminal']['exitcode'] == 0 and not wait['terminal']['controller_alive']
assert retention['status'] == 'CLOSED_NONBEST_WEIGHTS_REMOVED' and resources['deleted_weights_absent']
assert not resources['gpu_compute_processes']
best = retention['retained_best']
assert best['system'] == summary['metric_best_candidate']['system']
assert best['hits'] == [summary['metric_best_candidate']['rec_hits25'], summary['metric_best_candidate']['rec_hits50']]
assert len(retention['deleted']) == 2 and retention['local_weight_archive_created'] is False
assert previous['section'] == '20.376.74'
repos = [workspace / '.codex_mcln_g0_20260905', workspace / '.codex_pvground_cs_20261002',
         workspace / '.codex_mcln_v99_internal_20260928']
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies = [repo / doc for repo in repos] + [workspace / 'Desktop/document' / Path(doc).name]
old = copies[0].read_bytes()
assert hashlib.sha256(old).hexdigest() == previous['handoff_sha256']
assert all(path.read_bytes() == old for path in copies) and b'## 20.376.75 ' not in old
for repo, head in zip(repos, previous['heads']):
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip() == head
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
stamp = datetime.datetime.now().astimezone().isoformat()
effect = summary['strategy_vs_control']['formal']['0.5']
refinement = summary['stages']['member_target']['formal']['same_query_refinement']['0.5']
versus_parent = summary['versus_protected_parent']['member_target']['0.5']
verdict = audit['verdict']
table = '\n'.join('| {system} | {rec_acc25:.4f}%〔{rec_hits25}〕 | {rec_acc50:.4f}%〔{rec_hits50}〕 |'.format(**row)
                  for row in summary['table'])
arm_lines = []
for arm in ('control', 'member_target'):
    training = summary['training'][arm]
    formal = summary['stages'][arm]['formal']
    coverage = formal['candidate_availability']['0.5']
    parent_delta = summary['versus_protected_parent'][arm]['0.5']
    arm_lines.append(f"- {arm}：额外候选学习职责累计 {training['total_extra_candidate_roles']} 次，空资格表达 {training['empty_expression_rows']} 条，额外边界目标超出当前节点范围累计 {training['total_extra_boundary_outside']} 面；额外几何loss前100／末100步均值 {training['extra_geometry_loss_first100_mean']:.8f}／{training['extra_geometry_loss_last100_mean']:.8f}。正式Mask为 {formal['mask_hits25']}／{formal['mask_hits50']}，mIoU {formal['mask_miou']:.6f}%。严格Full256离线上界 {coverage['oracle_hits'][-1]}；错误中有合格框 {coverage['errors_with_good_full256']}，无合格框 {coverage['errors_without_good_full256']}。相对4509起点严格修复 {parent_delta['repairs']}、破坏 {parent_delta['damages']}、净 {parent_delta['net']:+d}。")
volume_lines = '\n'.join(f"- GT体积第{group['volume_quartile']}四分位：{group['rows']}条，control／member_target严格命中 {group['control_hits50']}／{group['member_target_hits50']}，净 {group['strategy_vs_control']['net']:+d}。" for group in summary['formal_gt_volume_groups'])
target_text = '已有同一模型达到5615／4754开发线。' if summary['scanrefer_target_pass'] else '尚无同一模型达到5615／4754开发线。'
prefix = 'refine-logs/pvground_auxiliary_target_20261005/'
intake = json.loads((local / 'complete/INTAKE.json').read_bytes())
section = f'''

## 20.376.75 native／member辅助定位目标对照闭合：完整9508结果、审计及指标最佳权重保留（{stamp}）

承接§20.376.74。两组实际全部结束时间 {summary['actual_finished_cst']}，控制器和两组train／formal退出码均为0。每组重新加载同一4509几何头及原PV／G父权重，重新初始化优化器；29778条fit各一次且输入顺序相同，3723次更新，batch8、累积1、尾batch2、seed2027。只训练已有456102参数／10项状态的六面分布几何头；父模型、Mask、语义及全零R冻结且eval。几何头原有7446次更新，本轮终点累计11169次；原G更早的适配历史另计。

| 完整9508开发验证，同一模型原生last/bbs | Acc@0.25 | Acc@0.50 |
|---|---:|---:|
{table}

两组额外几何权重均为1，差别是辅助资格中Box≤0.5判断及额外L1／GIoU／边界分布目标使用的坐标：control使用原生独立GT框抖动后的native_gt；member_target使用同一增强点云、原生50000点表示的实际目标成员，在独立GT框抖动前计算的member_gt。这不是原始全分辨率点云包围框。此项同时改变辅助资格和辅助定位目标，不是仅替换某一个loss标签。原生GT、Hungarian匹配、native＋G＋matched DFL监督、对象输入及正式评估均不变；额外字段不传给模型。

额外资格要求候选自己的Query Mask和融合Mask都与训练root GT的IoU>0.5、最终框不足、且不属于任何原生匹配Query；资格停止梯度，每表达内平均再按实际batch平均，空集合贡献0。推理不读取GT资格、不裁剪低分候选，仍保留全部256候选、唯一原生bbs、同一Query的Box与Mask。没有新增attention、R、质量评分、教师、参考框结构或后处理。

member_target相对本轮同预算control：严格修复 {effect['repairs']}、破坏 {effect['damages']}、净 {effect['net']:+d}；最终所选Query变化 {effect['selected_query_changes']}。相对4509起点：严格修复 {versus_parent['repairs']}、破坏 {versus_parent['damages']}、净 {versus_parent['net']:+d}。member_target内部同一已选Query粗框→精修框：{refinement['coarse_hits']}→{refinement['final_hits']}，修复 {refinement['repairs']}、破坏 {refinement['damages']}；该内部变化不是相对独立baseline的增益。

{chr(10).join(arm_lines)}

按相同GT体积离线分组，不能作推理输入：
{volume_lines}

每组初始和终点6887条是作者预训练见过场景的模块留出，不能与9508正式开发验证混算。完整收集 {len(intake['files'])} 个闭合原始文件，共 {sum(entry['bytes'] for entry in intake['files'].values())} 字节；未下载权重、未重新跑模型或优化器。CPU从保存的选中Box／GT重新计算阈值命中；Mask及完整候选覆盖为保存标量的核对，不是原始点级Mask或全候选数组重放。缓存上游重复一致不证明跨CUDA完整前向逐位一致。资格的GT重叠不等于完整物理实例身份真值。

fresh终态experiment-audit为 {verdict}，阻断发现为0；请求Astra／max，新上下文审阅，后端模型身份未获独立证实，按same-family／provisional记录，不称外部或跨模型审查。报告和实际调用记录见 {prefix}analysis/EXPERIMENT_AUDIT.md、JSON及TERMINAL_REVIEW_CALL.json。单seed小变化不证明稳定显著提升、Nr3D／Sr3D泛化或完整三模块创新。历史准备文件的pending字段只表示当时状态，以本次终态报告和清理执行记录为准。

完整两组复核后按预先规定的Acc@0.50主标准保留 {best['system']}：命中 {best['hits'][0]}／{best['hits'][1]}，路径 {best['path']}，SHA256 {best['sha256']}。严格阈值与4509起点持平时保留原起点；两个新模型之间严格持平才按Acc@0.25选取。实际删除两份闭合非最佳几何头，释放 {retention['released_bytes']} 字节，不创建负权重归档。保留文件含完整10项几何状态和优化器状态，可与官方PV及原G重构，无需旧几何祖先文件。官方PV／G身份复核通过，V99保护链没有被本次清理触及。数据盘空余 {resources['data_free_bytes']} 字节，系统盘 {resources['system_free_bytes']} 字节，GPU计算进程为空。远端发布complete使用数据盘已闭合原始结果的链接，不复制大日志占用系统盘。

{target_text}目前所保留模型严格距离4754还有 {max(0,4754-best['hits'][1])} 条，距离V99的4797还有 {max(0,4797-best['hits'][1])} 条。继续PV-Ground主线；下一步依此两组相对控制、相对强起点及修复／破坏的真实结果决定辅助定位目标是否保留。若仍存在参考尺度或节点范围限制，再单独研究自身支撑参考和渐进边界；若候选几何改善却未兑现实际命中，再研究选择协调。未启动Nr3D／Sr3D新结构训练，不重复已完成的普通P2、局部概率拼接、一般回读或语义归一化。三个数据集总目标仍为ACTIVE_UNMET。
'''
new = old + section.encode('utf-8')
assert new.startswith(old) and new.count(b'## 20.376.75 ') == 1
names = ['fit_wait.json', 'weight_retention.json', 'CLOSED_RESOURCES.json', 'check_closed_resources.py', 'prepare_terminal_publication.py', 'TERMINAL_PUBLICATION_PREPARATION.json',
         'TERMINAL_PUBLICATION_SOURCE_CHECK.json', Path(__file__).name]
for directory in ('complete', 'analysis'):
    names.extend(str(path.relative_to(local)).replace('\\', '/') for path in sorted((local / directory).rglob('*')) if path.is_file())
assert all(not name.endswith(('.pth', '.pt')) and '.aris' not in name for name in names)
payloads = {prefix + name: (local / name).read_bytes() for name in names}
payloads[prefix + '.gitattributes'] = b'** -text\n'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
project = '/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(project + '/' + doc, 'rb') as stream:
    assert stream.read() == old
evidence = project + '/' + prefix.rstrip('/')
assert not any(entry.filename == 'complete' for entry in sftp.listdir_attr(evidence))
sftp.symlink('/root/autodl-tmp/pvground_auxiliary_target_20261005', evidence + '/complete')
assert sftp.readlink(evidence + '/complete') == '/root/autodl-tmp/pvground_auxiliary_target_20261005'
directories = set()
for name in payloads:
    parts = Path(name).parts[:-1]
    directories.update('/'.join(parts[:index]) for index in range(1, len(parts) + 1))
for relative in sorted(directories, key=lambda name: (name.count('/'), name)):
    parent, _, basename = relative.rpartition('/')
    if not any(entry.filename == basename for entry in sftp.listdir_attr(project + ('/' + parent if parent else ''))):
        sftp.mkdir(project + '/' + relative)
for name, raw in payloads.items():
    for repo in repos[:2]:
        destination = repo / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    remote = project + '/' + name
    if not name.startswith(prefix + 'complete/') or name == prefix + 'complete/INTAKE.json':
        with sftp.open(remote, 'wb') as stream:
            stream.write(raw)
    with sftp.open(remote, 'rb') as stream:
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
    stage = [doc]
    if index < 2:
        with (repo / 'MANIFEST.md').open('a', encoding='utf-8') as stream:
            stream.write('\n- ' + stamp + ' Actual closed native/member auxiliary target comparison published; metric-best ' + best['system'] + ' ' + str(best['hits']) + ', fresh audit ' + verdict + ' same-family/provisional. Two nonbest geometry heads removed, no archive.\n')
        stage += ['MANIFEST.md', *payloads]
    changed = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage], stderr=subprocess.DEVNULL)
    reports = [name for name in payloads if Path(name).name in ('EXPERIMENT_AUDIT.json', 'EXPERIMENT_AUDIT.md')] if index < 2 else []
    subprocess.check_call(['git', '-C', str(repo), '-c', 'core.whitespace=cr-at-eol', 'diff', '--cached', '--check', '--', *stage, *[':(exclude)' + name for name in reports]])
    subprocess.check_call(['git', '-C', str(repo), '-c', 'core.whitespace=cr-at-eol,-blank-at-eof', 'diff', '--cached', '--check', '--', *stage])
    if index < 2:
        for name, raw in payloads.items():
            assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + name]) == raw
    git_doc = subprocess.check_output(['git', '-C', str(repo), 'show', ':' + doc])
    old_git_doc = subprocess.check_output(['git', '-C', str(repo), 'show', previous['heads'][index] + ':' + doc])
    assert git_doc.startswith(old_git_doc) and git_doc.count(b'## 20.376.75 ') == 1
    subprocess.check_call(['git', '-C', str(repo), 'commit', '--quiet', '-m', 'Record closed native/member auxiliary target comparison and best retention'])
    heads.append(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip())
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
subprocess.check_call(['git', '-C', str(repos[0]), '-c', 'http.version=HTTP/1.1', 'push', 'origin', 'HEAD:main'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == heads[0]
digest = hashlib.sha256(new).hexdigest()
guard = workspace / '.codex/tmp/sync_cs_handoff_remote_20260923.py'
raw = guard.read_bytes()
assert raw.count(previous['handoff_sha256'].encode()) == 1
guard.write_bytes(raw.replace(previous['handoff_sha256'].encode(), digest.encode()))
assert all(path.read_bytes() == new for path in copies)
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), section='20.376.75', heads=heads,
              github_main=heads[0], handoff_bytes=len(new), handoff_sha256=digest, four_local_and_remote_equal=True,
              payload_count=len(payloads), new_accuracy_result=True, table=summary['table'],
              retained_best=best, deleted_nonbest_bytes=retention['released_bytes'], negative_weight_archived=False,
              integrity_verdict=verdict, review_independence='same-family', acceptance_status='provisional',
              scanrefer_target_pass=summary['scanrefer_target_pass'], full_goal_status='ACTIVE_UNMET',
              remote_complete_reuses_closed_data_disk_originals=True)
(local / 'terminal_publication.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
state.update(time_cst=record['time_cst'], status='AUXILIARY_TARGET_TERMINAL_PUBLISHED',
             latest_publication=str(local / 'terminal_publication.json'), owned_gpu_job_active=False,
             active_reviewer=None, protected_best_hits=best['hits'], strict_target_gap=max(0, 4754-best['hits'][1]),
             published_heads=heads, handoff_section='20.376.75', handoff_sha256=digest,
             auxiliary_target_fit_observer_closed=True, auxiliary_target_fit_observer_session_id=None,
             auxiliary_target_full_pair_complete=True, auxiliary_target_member_formal_result_unobserved=False,
             auxiliary_target_terminal_integrity_review_pending=False, auxiliary_target_terminal_integrity_verdict=verdict,
             auxiliary_target_retention_tools_executed=True,
             auxiliary_target_actual_fit_finished_cst=summary['actual_finished_cst'],
             auxiliary_target_formal_hits_by_arm={row['system']:[row['rec_hits25'],row['rec_hits50']] for row in summary['table']},
             current_goal_turn_classification='PROGRESS_ACTUAL_TERMINAL_AUDIT_RETENTION_PUBLISHED')
state_path.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
with (workspace / 'memory/2026-10-05.md').open('a', encoding='utf-8') as stream:
    stream.write('\nPVGround ' + record['time_cst'] + ': Native/member auxiliary target full pair closed ' + str(summary['table']) + '. Actual fresh audit ' + verdict + ' same-family/provisional; retained ' + str(best['hits']) + ', two closed nonbest heads deleted without archive. Doc75 fourlocal+remote equal, main ' + heads[0] + '. Each29778once/3723updates/B8, existing geometry head only; total11169 geometry updates. Fullgoal ACTIVE_UNMET, no new Nr/Sr result.\n')

print(json.dumps(record), flush=True)
