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
state = json.loads((local / 'active_continuation_state.json').read_bytes())
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
assert previous['section'] == '20.376.65'
repos = [workspace / '.codex_mcln_g0_20260905', workspace / '.codex_pvground_cs_20261002',
         workspace / '.codex_mcln_v99_internal_20260928']
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies = [repo / doc for repo in repos] + [workspace / 'Desktop/document' / Path(doc).name]
old = copies[0].read_bytes()
assert hashlib.sha256(old).hexdigest() == previous['handoff_sha256']
assert all(path.read_bytes() == old for path in copies) and b'## 20.376.66 ' not in old
for repo, head in zip(repos, previous['heads']):
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip() == head
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
stamp = datetime.datetime.now().astimezone().isoformat()
effect = summary['strategy_vs_control']['formal']['0.5']
refinement = summary['stages']['query_supported']['formal']['same_query_refinement']['0.5']
versus_parent = summary['versus_protected_parent']['query_supported']['0.5']
verdict = audit['verdict']
table = '\n'.join('| {system} | {rec_acc25:.4f}%〔{rec_hits25}〕 | {rec_acc50:.4f}%〔{rec_hits50}〕 |'.format(**row)
                  for row in summary['table'])
arm_lines = []
for arm in ('control', 'query_supported'):
    training = summary['training'][arm]
    formal = summary['stages'][arm]['formal']
    coverage = formal['candidate_availability']['0.5']
    arm_lines.append(f"- {arm}：累计额外候选职责 {training['total_extra_candidate_roles']}；空资格表达行 {training['empty_expression_rows']}；超出当前分布位置的额外边界目标 {training['total_extra_boundary_outside']}。额外几何项前100／末100步均值 {training['extra_geometry_loss_first100_mean']:.8f}／{training['extra_geometry_loss_last100_mean']:.8f}。控制组只记录该项，不加入优化目标。正式Mask命中 {formal['mask_hits25']}／{formal['mask_hits50']}，mIoU {formal['mask_miou']:.6f}%；严格Full256几何上界 {coverage['oracle_hits'][-1]}，错误中有合格框 {coverage['errors_with_good_full256']}、无合格框 {coverage['errors_without_good_full256']}。")
volume_lines = '\n'.join(f"- GT体积分组{group['volume_quartile']}，{group['rows']}条：控制／策略严格命中 {group['control_hits50']}／{group['query_supported_hits50']}，净变化 {group['strategy_vs_control']['net']:+d}。" for group in summary['formal_gt_volume_groups'])
target_text = '存在达到5615／4754双阈值开发线的同一模型。' if summary['scanrefer_target_pass'] else '本轮没有达到5615／4754双阈值开发线的同一模型。'
prefix = 'refine-logs/pvground_query_supported_geometry_20261005/'
intake = json.loads((local / 'complete/INTAKE.json').read_bytes())
section = f'''

## 20.376.66 Query与融合Mask共同支撑的未匹配候选几何训练完成：真实配对结果、终态审计与最佳权重保留（{stamp}）

承接§20.376.64，两组均从已验证的5616／4506几何父模型重新构建，未加载预检两步状态；新AdamW各实际更新3723次，29778条fit各一次、顺序一致，batch8、累积1，尾批2条，seed2027。各组含初始与终点6887条预训练见过场景的模块留出，以及完整9508条开发验证；两组控制器实际完成于 {summary['actual_finished_cst']}，退出码0。只更新已有456102参数／10项状态的边界精修头，官方PV、原G及零残差R全程冻结并eval，保存前父状态精确检查通过。既有几何父头3723次更新与本轮相加，终点累计几何更新7446次；原G此前的结构与适配历史仍须单独披露。

| 同一检查点、完整9508、唯一原生last/bbs | Acc@0.25〔命中〕 | Acc@0.50〔命中〕 |
|---|---:|---:|
{table}

策略仅补充未被任何原生Hungarian匹配、自己的Query Mask与最终融合Mask对root均IoU>0.5、而最终框IoU≤0.5的候选。Text Mask单独合格不能取得资格；GT资格停止梯度且仅用于训练，不证明候选真实物理身份。原匹配回归与原生损失分母保持不变，新增(10L1＋2GIoU＋面均DFL)/7先在每条表达内平均，再按实际batch平均；观察到的空集合为零。控制组相同计算与记录，新增权重为0；策略为1。没有新增网络层、第二套推理评分、教师或GT推理门控；256候选保留，只选择一个Query交付对应Box与Mask。

正式策略相对本轮控制：严格修复 {effect['repairs']}、破坏 {effect['damages']}、净 {effect['net']:+d}，最终选择Query变化 {effect['selected_query_changes']}条。策略相对保护的4506父模型：修复 {versus_parent['repairs']}、破坏 {versus_parent['damages']}、净 {versus_parent['net']:+d}。策略内部同一已选Query的粗框／精修框：严格 {refinement['coarse_hits']}→{refinement['final_hits']}，修复 {refinement['repairs']}、破坏 {refinement['damages']}；这不是相对独立baseline的增量。策略已选框最大单面位移的中位数 {summary['stages']['query_supported']['formal']['median_selected_max_face_displacement_m']:.8f}米，最大值 {summary['stages']['query_supported']['formal']['max_selected_face_displacement_m']:.8f}米。

{chr(10).join(arm_lines)}

离线按相同GT体积四分位比较，不作为推理输入：
{volume_lines}

仅收集 {len(intake['files'])}份JSON／JSONL／log／exit／源码文本、{sum(entry['bytes'] for entry in intake['files'].values())}字节；下载或新建权重归档0。CPU重新从保存的所选粗框、最终框和GT计算IoU，两阈值变化均为0；同表达坐标、点采样SHA、fit行顺序受检。Mask与Full256统计为保存标量核算，未重放原始Mask／全候选数组，也未重新运行神经模型。预检缓存同上游重放精确，不意味着跨CUDA进程完整前向逐位一致。原生真实匹配是否覆盖其他实例，以训练记录与审计的实际范围为准，不能把源码排除规则写成已观察到多实例样本。

实际收到fresh终态experiment-audit结果 {verdict}，无阻断项；同族／provisional，后端未获得工具证明，不称独立外部审查。详见{prefix}analysis/EXPERIMENT_AUDIT.md与JSON及实际TERMINAL_REVIEW_CALL。分析器保留的review_pending／cleanup_pending反映分析当时状态，后续以真实审查与weight_retention.json为准。既有单seed、预训练见过的6887与长期开发9508、GT资格与实例身份代理的边界不变；本轮不证明Nr3D／Sr3D或完整三模块novelty。

闭合与审计之后实际删除两份非最佳几何头，释放 {retention['released_bytes']}字节，无负结果权重归档。保留 {best['system']}，命中 {best['hits'][0]}／{best['hits'][1]}，路径 {best['path']}，SHA256 {best['sha256']}；它含完整10项几何状态。若新头胜出，旧4506几何文件已不再是重建依赖；官方PV与原G仍是必需父权重且实际SHA复核一致。V99文件不在删除集合。删除后数据盘剩余 {resources['data_free_bytes']}字节、系统盘 {resources['system_free_bytes']}字节，实际GPU计算进程为空。远端发布complete证据链接到已闭合数据盘原始结果，避免系统盘再复制整份逐行日志；预检独立快照仍保留。

{target_text}按Acc@0.50、再Acc@0.25与同分父模型优先的预定规则，当前保留最佳为 {best['hits'][0]}／{best['hits'][1]}，严格距4754命中还差 {max(0,4754-best['hits'][1])}。本轮结果用于判断补充合格Mask候选的几何责任是否形成真实增量；不因loss下降、几何上界或追回退化控制就宣布达到目标。继续PV-Ground、完整实例支撑与六面范围、单一原生评分；后续具体结构选择依据本轮实际修复／破坏、支撑边界误差与候选选择差距，不重复已完成的局部概率拼接、简单P2、R容量扩展或仅教师框坐标实验。Nr／Sr新结构正式训练尚未开始，三数据集目标仍ACTIVE_UNMET。
'''
new = old + section.encode('utf-8')
assert new.startswith(old) and new.count(b'## 20.376.66 ') == 1
names = ['fit_wait.json', 'weight_retention.json', 'CLOSED_RESOURCES.json', 'check_closed_resources.py',
         'TERMINAL_PUBLICATION_SOURCE_CHECK.json', Path(__file__).name]
for directory in ('complete', 'analysis'):
    names.extend(str(path.relative_to(local)).replace('\\', '/') for path in sorted((local / directory).rglob('*')) if path.is_file())
assert all(not name.endswith(('.pth', '.pt')) and '.aris' not in name for name in names)
payloads = {prefix + name: (local / name).read_bytes() for name in names}
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
project = '/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(project + '/' + doc, 'rb') as stream:
    assert stream.read() == old
evidence = project + '/' + prefix.rstrip('/')
assert not any(entry.filename == 'complete' for entry in sftp.listdir_attr(evidence))
sftp.symlink('/root/autodl-tmp/pvground_query_supported_geometry_20261005', evidence + '/complete')
assert sftp.readlink(evidence + '/complete') == '/root/autodl-tmp/pvground_query_supported_geometry_20261005'
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
            stream.write('\n- ' + stamp + ' Actual closed Query-supported geometry comparison published; metric-best ' + best['system'] + ' ' + str(best['hits']) + ', fresh audit ' + verdict + ' same-family/provisional. Two nonbest geometry heads removed, no archive.\n')
        stage += ['MANIFEST.md', *payloads]
    changed = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage], stderr=subprocess.DEVNULL)
    subprocess.check_call(['git', '-C', str(repo), 'diff', '--cached', '--check', '--', *stage])
    if index < 2:
        for name, raw in payloads.items():
            assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + name]) == raw.replace(b'\r\n', b'\n')
    git_doc = subprocess.check_output(['git', '-C', str(repo), 'show', ':' + doc])
    old_git_doc = subprocess.check_output(['git', '-C', str(repo), 'show', previous['heads'][index] + ':' + doc])
    assert git_doc.startswith(old_git_doc) and git_doc.count(b'## 20.376.66 ') == 1
    subprocess.check_call(['git', '-C', str(repo), 'commit', '--quiet', '-m', 'Record closed Query-supported geometry training and metric-best retention'])
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
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), section='20.376.66', heads=heads,
              github_main=heads[0], handoff_bytes=len(new), handoff_sha256=digest, four_local_and_remote_equal=True,
              payload_count=len(payloads), new_accuracy_result=True, table=summary['table'],
              retained_best=best, deleted_nonbest_bytes=retention['released_bytes'], negative_weight_archived=False,
              integrity_verdict=verdict, review_independence='same-family', acceptance_status='provisional',
              scanrefer_target_pass=summary['scanrefer_target_pass'], full_goal_status='ACTIVE_UNMET',
              remote_complete_reuses_closed_data_disk_originals=True)
(local / 'terminal_publication.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
state.update(time_cst=record['time_cst'], status='QUERY_GEOMETRY_TERMINAL_PUBLISHED', latest_publication=str(local / 'terminal_publication.json'),
             owned_gpu_job_active=False, fit_observer_closed=True, observer_session_id=None, active_reviewer=None,
             retention_executed=True, protected_best_hits=best['hits'], strict_target_gap=max(0, 4754-best['hits'][1]),
             published_heads=heads, handoff_section='20.376.66', handoff_sha256=digest,
             current_goal_turn_classification='PROGRESS_ACTUAL_TERMINAL_PUBLISHED')
(local / 'active_continuation_state.json').write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
with (workspace / 'memory/2026-10-05.md').open('a', encoding='utf-8') as stream:
    stream.write('\nPVGround ' + record['time_cst'] + ': Query-supported geometry full pair closed ' + str(summary['table']) + '. Same29778/order,3723 updates/head-only. Actual fresh audit ' + verdict + ' same-family/provisional; retained ' + str(best['hits']) + ', two nonbest heads deleted/no archive. Doc66 fourlocal+remote equal, main ' + heads[0] + '. Fullgoal ACTIVE_UNMET; no new Nr/Sr result.\n')
print(json.dumps(record), flush=True)
