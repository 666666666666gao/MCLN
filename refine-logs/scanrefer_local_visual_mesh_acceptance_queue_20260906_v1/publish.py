import datetime
import hashlib
import json
import os
from pathlib import Path

import paramiko


repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
root = repo / 'refine-logs/scanrefer_local_visual_mesh_acceptance_queue_20260906_v1'
master = repo / 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop = Path('C:/Users/gb/Desktop/document/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md')
old = master.read_bytes()
assert hashlib.sha256(old).hexdigest() == 'c3e1c38090b40033c06ddd707834a7c8a0637402631976c3706918ca91dce023'
assert desktop.read_bytes() == old
plan = json.loads((root / 'plan.json').read_bytes())
launch = json.loads((root / 'launch.json').read_bytes())
assert plan['native_launcher_other_source_identical']
assert not plan['actual_old9508_negative_gate_recomputed']['advance_to_nr3d_sr3d_rec']
assert launch['worker_pid'] == 50808 and launch['existing_observer_pid'] == 50584
assert launch['gpu_forwards'] == launch['optimizer_steps'] == launch['formal_rows'] == 0
assert not (root / 'decision.json').exists()
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
when = now.strftime('%Y-%m-%d %H:%M CST')
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
runtime = '/home/gb/new butd/butd_detr-main/MCLN-main/'
with sftp.open(runtime + 'docs/' + master.name, 'rb') as stream:
    stream.prefetch(file_size=len(old))
    assert stream.read() == old
remote_root = '/root/autodl-tmp/mcln_' + root.name
sftp.mkdir(remote_root)
(root / 'publish.py').write_bytes(Path(__file__).read_bytes())
archived = {}
for name in ['continue_after_formal.py', 'audit_terminal.py', 'launch_native_conditional.py',
             'run_queue.ps1', 'plan.json', 'prepare.py', 'wait_started.json', 'launch.json', 'publish.py']:
    raw = (root / name).read_bytes()
    with sftp.open(remote_root + '/' + name, 'wx') as stream:
        stream.write(raw)
    with sftp.open(remote_root + '/' + name, 'rb') as stream:
        assert stream.read() == raw
    archived[name] = hashlib.sha256(raw).hexdigest()
addition = '\n\n### 20.99 本轮Scan正式评估后独立审计及Nr/Sr条件接续（' + when + '）\n\n'
addition += (
    '§20.98的训练、服务端终态审计/正式评估队列及本地观察器均沿用原设置。'
    '本节没有新的训练终点或正式指标；最近训练进度证据仍为23:02两臂64/2482步。'
    '没有据此改变主干、损失、学习率、更新预算、候选规则或晋级阈值。\n\n'
    '**实际衔接更新：** 旧Nr/Sr GPU预检入口绑定了已经失败的Scan formal v3。'
    '新归档入口只将这一个formal_root改为本轮`mcln_scanrefer_local_visual_mesh_official_20260906_v1`；'
    '与旧入口比较，其他源码完全相同。其完整9508行身份/点SHA、REC与Mask复算、实际正式回执SHA、'
    '独立审计SHA和GPU空闲检查继续生效。旧失败入口仍未执行。\n\n'
    '新本地接续进程PID50808于23:38:35确认存活，子进程PID37596实际执行'
    '`Wait-Process -Id 50584`；目标观察器50584也同时确认存活。执行会话10701。'
    '它等待现有收集进程自然结束，不另开一份训练观察或正式评估。服务端只保存此Windows入口的归档；'
    '实际接续worker运行在本地。\n\n'
    '后续顺序固定为：原观察器获得本轮真实formal启动回执并退出；新worker核验该回执与输入manifest；'
    '于实际formal启动后24分钟首次查询formal screen，此后240秒；screen真实退出且controller为0后，'
    '执行既有独立CPU正式审计；审计通过后按既定Scan条件决定是否调用Nr/Sr原生GPU预检。'
    '24分钟来自上一轮同源码9508配对评估实际1537.14秒加初始化的时间证据，只用于查询调度。\n\n'
    '正式审计入口绑定§20.98封存preparation及本轮实际launch，继续核对训练终点、源码、权重、'
    'mesh superpoints、9508条原生/系统输出与修复破坏。它尚未执行，因为本轮正式结果尚未生成。'
    '若Scan未达标，接续worker只保存负结果decision并停止；若达标，先启动既已准备的Nr/Sr真实输入GPU预检，'
    '不把预检或运行状态写成Nr/Sr训练完成。后续完整训练仍须读取实际预检结果。\n\n'
    '**本次已执行检查：** 三个入口完成语法编译；Windows原生等待在真实短生命周期子进程上完成；'
    '旧v3真实9508行重新计算仍为REC四项未过、ScanMask三项通过，晋级false；'
    '新native入口除formal_root外与原入口逐字一致。没有新增GPU forward、optimizer更新或权重。'
    '原正式评估及审计源码仍使用§20.98封存并已经测试的版本。\n\n'
    '接续plan SHA`' + launch['plan_sha256'] + '`；实际启动记录SHA`' + archived['launch.json'] + '`；'
    '源码及记录位于`refine-logs/' + root.name + '/`。后续不要手工重复执行本轮正式审计或native条件入口；'
    '先读取此worker的实际状态和decision。观察超时不代表GPU任务终止。\n\n'
    '磁盘清理沿用§20.97两轮累计9.529GiB；23:27实查服务器剩余10839310336 bytes（约10.095GiB），'
    '四个失败终点已退休，六个受保护文件均在。当前仍只计划保存两臂各一个终点。'
    '整体三数据集目标未达成，goal保持active。\n'
)
new = old + addition.encode('utf-8')
tracker = repo / 'refine-logs/EXPERIMENT_TRACKER.md'
lines = tracker.read_text(encoding='utf-8').splitlines()
lines[2] = 'Updated: ' + when + '. Acceptance: §20.79; Scan mesh training/endpoint queue: §20.98; formal audit/native conditional continuation: §20.99.'
insert_at = next(i for i, line in enumerate(lines) if line.startswith('| Nr/Sr warm-start interface |'))
lines.insert(insert_at, '| ScanRefer mesh formal acceptance queue | Localworker50808 and waitchild37596 verified live;waits existingcollector50584 | Actual formal terminal then independentCPU audit;only ScanPASS permits disposable Nr/Sr GPU preflight;session10701;see20.99 |')
tracker_raw = ('\n'.join(lines) + '\n\n' + when + ': Master20.99 binds future formal CPU audit and native preflight to the current mesh repeat;actual local waiting worker verified. No new formal metrics or native GPU execution.\n').encode('utf-8')
for name, raw in [('docs/' + master.name, new), ('refine-logs/EXPERIMENT_TRACKER.md', tracker_raw)]:
    with sftp.open(runtime + name, 'wb') as stream:
        stream.set_pipelined(True)
        stream.write(raw)
    with sftp.open(runtime + name, 'rb') as stream:
        stream.prefetch(file_size=len(raw))
        assert stream.read() == raw
master.write_bytes(new)
desktop.write_bytes(new)
tracker.write_bytes(tracker_raw)
assert master.read_bytes() == desktop.read_bytes() == new
proof = {'section': '20.99', 'time_cst': now.isoformat(), 'bytes': len(new),
         'sha256': hashlib.sha256(new).hexdigest(), 'three_master_copies_equal': True,
         'archived_source_sha256': archived, 'local_queue_worker_pid': 50808,
         'formal_result_available': False, 'native_gpu_preflight_launched': False}
raw = (json.dumps(proof, indent=2, sort_keys=True) + '\n').encode('utf-8')
with (root / 'handoff_sync_20_99.json').open('xb') as stream:
    stream.write(raw)
with sftp.open(remote_root + '/handoff_sync_20_99.json', 'wx') as stream:
    stream.write(raw)
sftp.close()
client.close()
with (repo / 'MANIFEST.md').open('ab') as stream:
    stream.write(('| ' + now.strftime('%Y-%m-%d %H:%M') + ' | /monitor-experiment | refine-logs/' + root.name + '/launch.json | run | §20.99 actual live formal-audit/native conditional waiter;no new quality claim |\n').encode('utf-8'))
print(json.dumps(proof))
