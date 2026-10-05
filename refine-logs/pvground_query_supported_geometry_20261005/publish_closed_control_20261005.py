"""Publish the actual closed control receipt; full-pair review remains pending."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import paramiko

local = Path(__file__).resolve().parent
workspace = Path('C:/Users/gb')
assert not (local / 'closed_control_publication.json').exists()
state = json.loads((local / 'active_continuation_state.json').read_bytes())
previous = json.loads(Path(state['latest_publication']).read_bytes())
intake = json.loads((local / 'CONTROL_CLOSED_INTAKE.json').read_bytes())
assert intake['status'] == 'CLOSED_CONTROL_RECEIPTS_READ' and not intake['full_pair_complete']
assert intake['CPU_full_rows_recount_pending'] and not intake['best_weight_promoted']
for name, identity in intake['files'].items():
    raw = (local / 'closed_control_receipts' / name).read_bytes()
    assert len(raw) == identity['bytes'] and hashlib.sha256(raw).hexdigest() == identity['sha256']
fit = json.loads((local / 'closed_control_receipts/control/receipt.json').read_bytes())
metric = intake['native_bbs']
assert [metric['rec_hits25'], metric['rec_hits50']] == [5616, 4499]
assert fit['head_parameters'] == 456102 and fit['head_state_tensors'] == 10
source_check = json.loads((local / 'TERMINAL_PUBLICATION_SOURCE_CHECK.json').read_bytes())
for name, digest in source_check['source_files'].items():
    assert hashlib.sha256((local / name).read_bytes()).hexdigest() == digest
repos = [workspace / '.codex_mcln_g0_20260905', workspace / '.codex_pvground_cs_20261002', workspace / '.codex_mcln_v99_internal_20260928']
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies = [repo / doc for repo in repos] + [workspace / 'Desktop/document' / Path(doc).name]
old = copies[0].read_bytes()
assert previous['section'] == '20.376.64' and hashlib.sha256(old).hexdigest() == previous['handoff_sha256']
assert all(path.read_bytes() == old for path in copies) and b'## 20.376.65 ' not in old
for repo, head in zip(repos, previous['heads']):
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip() == head
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
stamp = datetime.datetime.now().astimezone().isoformat()
section = f'''

## 20.376.65 两组几何监督对照的控制组完成：5616／4499，策略组继续训练（{stamp}）

承接§20.376.64，本轮control已完成3723次更新、29778条fit各一次，物理／有效batch8、累积1、尾批2条，seed2027，新AdamW；仅已有456102参数／10状态的几何头参与更新，额外候选几何权重0，原native＋G与匹配DFL保持。实际训练收据记录parent与零R状态精确保持，终点重载的模型及10项优化器状态检查通过。正式9508条原生last/bbs于 {intake['formal_finished_cst']} 完成，formal与train退出码均0；11:37:46监控已观察到control/train与control/formal闭合，query_supported/train子进程599785启动。

| 同一模型、完整9508开发验证、原生last/bbs | Acc@0.25〔命中〕 | Acc@0.50〔命中〕 |
|---|---:|---:|
| 保护的几何父模型 | 59.0660%〔5616〕 | 47.3917%〔4506〕 |
| 本轮head-only续训控制 | {100*metric['rec_hits25']/9508:.4f}%〔{metric['rec_hits25']}〕 | {100*metric['rec_hits50']/9508:.4f}%〔{metric['rec_hits50']}〕 |

控制组相对父模型计数0／−7。6887条预训练见过场景的模块留出由6174／5616变为6171／5617；@0.25修复0、破坏3，@0.50修复7、破坏6。模块留出严格净+1而正式严格−7，不能将该留出当作未见场景增益。正式Mask命中 {metric['mask_hits25']}／{metric['mask_hits50']}，mIoU {metric['mask_miou']:.6f}%；Mask仍为原生输出，不能用V99数字替代。

本次仅读取5份已闭合小文本收据共 {sum(entry['bytes'] for entry in intake['files'].values())}字节，完整逐行结果尚未收集／CPU重算，没有神经前向或优化器重放，没有下载或归档权重。以上是已保存GPU完整验证收据，不是已完成两组CPU核算与终态审计的结论。完整两组的训练顺序、Box／GT阈值、修复／破坏、候选覆盖及fresh审查，仍待策略组正式闭合后统一完成。

策略组保持从同一5616／4506父模型重新初始化，只补充未匹配、自己的Query与融合Mask均支持root而最终框不合格的候选几何责任；不按低排名丢掉候选，不把全部错误候选改成正例，不恢复双源或教师推理。当前没有其正式精度，不得以控制组4499判断策略成败。

11:08:50的计划内资源检查：数据盘2258145280字节、系统盘56098816字节，GPU使用6981MiB；它是该时刻快照。控制组训练实测4245.34秒，终态6887评估日志当时5120条／713.36秒；同吞吐估计原约14:00完成整轮，控制正式实际早于估计约5.6分钟，策略组吞吐尚未测量，后续按其实际进度校准。监控会话11178仍拥有这轮controller584730，240秒远端复查；不新增GPU诊断抢占训练。

当前最佳正式结果继续保护5616／4506，未晋升控制权重，未执行删除；两组闭合、完整核算及实际审计后才使用已审过的清理脚本保留指标最佳几何头。官方PV、原G及V99所需父链继续保留，不新增负结果权重归档。源码与这份部分结果同步到GitHub、远端及四份本地交接文档；终态发布脚本顺延到§20.376.66，仅调整文档节号与来源检查，不改变已审GPU训练源码。ScanRefer双阈值开发线仍为同模型5615／4754；Nr／Sr新结构训练尚未开始，完整三数据集目标ACTIVE_UNMET。
'''
new = old + section.encode('utf-8')
assert new.startswith(old) and new.count(b'## 20.376.65 ') == 1
names = ['collect_closed_control_receipts.py', 'CONTROL_CLOSED_INTAKE.json',
         'publish_geometry_terminal_20261005.py', 'TERMINAL_PUBLICATION_SOURCE_CHECK.json', Path(__file__).name]
names += ['closed_control_receipts/' + name for name in intake['files']]
prefix = 'refine-logs/pvground_query_supported_geometry_20261005/'
payloads = {prefix + name: (local / name).read_bytes() for name in names}
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
project = '/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(project + '/' + doc, 'rb') as stream:
    assert stream.read() == old
directories = set()
for name in payloads:
    parts = Path(name).parts[:-1]
    directories.update('/'.join(parts[:index]) for index in range(1, len(parts)+1))
for relative in sorted(directories, key=lambda name: (name.count('/'), name)):
    parent, _, basename = relative.rpartition('/')
    if not any(entry.filename == basename for entry in sftp.listdir_attr(project + ('/' + parent if parent else ''))):
        sftp.mkdir(project + '/' + relative)
for name, raw in payloads.items():
    for repo in repos[:2]:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
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
    stage = [doc]
    if index < 2:
        with (repo / 'MANIFEST.md').open('a', encoding='utf-8') as stream:
            stream.write('\n- ' + stamp + ' Actual control9508 native5616/4499 closed, strategy running. GPU receipts only; full-row CPU/pair audit pending. Best5616/4506 unchanged; no cleanup. Doc65 synchronized.\n')
        stage += ['MANIFEST.md', *payloads]
    changed = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage], stderr=subprocess.DEVNULL)
    subprocess.check_call(['git', '-C', str(repo), 'diff', '--cached', '--check', '--', *stage])
    if index < 2:
        for name, raw in payloads.items():
            assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + name]) == raw.replace(b'\r\n', b'\n')
    git_doc = subprocess.check_output(['git', '-C', str(repo), 'show', ':' + doc])
    old_git = subprocess.check_output(['git', '-C', str(repo), 'show', previous['heads'][index] + ':' + doc])
    assert git_doc.startswith(old_git) and git_doc.count(b'## 20.376.65 ') == 1
    subprocess.check_call(['git', '-C', str(repo), 'commit', '--quiet', '-m', 'Record closed geometry supervision control while strategy continues'])
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
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), section='20.376.65', heads=heads,
    github_main=heads[0], handoff_bytes=len(new), handoff_sha256=digest, four_local_and_remote_equal=True,
    payload_count=len(payloads), new_accuracy_result=True, source='Closed single-arm GPU full9508 receipt',
    control_hits=[5616, 4499], protected_best_hits=[5616, 4506], full_pair_complete=False,
    CPU_full_rows_recount_pending=True, terminal_review_pending=True, cleanup_executed=False, full_goal_status='ACTIVE_UNMET')
(local / 'closed_control_publication.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
state.update(time_cst=record['time_cst'], latest_publication=str(local / 'closed_control_publication.json'),
    status='CONTROL_FORMAL_CLOSED_STRATEGY_RUNNING_PUBLISHED', observed_arm='query_supported', observed_mode='train',
    control_formal_receipts_collected=True, control_formal_hits=[5616, 4499], observed_child_pid=599785,
    latest_controller_observation_cst=intake['actual_phase_observation_cst'], current_fit_updates_unobserved=True,
    query_supported_updates_unobserved=True, formal_results_unobserved=True,
    handoff_section='20.376.65', handoff_sha256=digest, published_heads=heads,
    current_goal_turn_classification='PROGRESS_ACTUAL_CONTROL_RESULT_PUBLISHED')
(local / 'active_continuation_state.json').write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
with (workspace / 'memory/2026-10-05.md').open('a', encoding='utf-8') as stream:
    stream.write('\nPVGround ' + record['time_cst'] + ': control formal finished11:36:27 native5616/4499,47.3180 strict,parent-7; holdout6174/5616->6171/5617. Strategy running from11:37 observation. Only5 small receipts2030B,no fullrows/weights/replay. CPU/pair audit pending,best5616/4506 protected. Published doc65 fourlocal+remote equal/main ' + heads[0] + '; futureterminal publisher nowdoc66, GPU core/reviewed14 sources unchanged. GoalACTIVE_UNMET.\n')
print(json.dumps(record), flush=True)
