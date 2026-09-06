import datetime
import hashlib
import json
import os
from pathlib import Path

import paramiko


repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
root = repo / 'refine-logs/scanrefer_stage_trace_preparation_20260907_v1'
master = repo / 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop = Path('C:/Users/gb/Desktop/document/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md')
old = master.read_bytes()
assert hashlib.sha256(old).hexdigest() == '296d681fe918db1b66c1bacb69d27f55ee010f8358f5a040d385c1a4eb6c1c2d'
assert desktop.read_bytes() == old
receipt = json.loads((root / 'receipt.json').read_bytes())
binding = json.loads((root / 'dependency_binding.json').read_bytes())
assert receipt['tests'] == {'tests': 6, 'failures': 0, 'errors': 0, 'skipped': 0}
assert receipt['real_rows_traced'] == receipt['gpu_forwards'] == receipt['checkpoint_writes'] == 0
assert not binding['active_sources_modified']
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
when = now.strftime('%Y-%m-%d %H:%M CST')
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
runtime = '/home/gb/new butd/butd_detr-main/MCLN-main/'
remote = '/root/autodl-tmp/mcln_scanrefer_stage_trace_preparation_20260907_v1'
with sftp.open(runtime + 'docs/' + master.name, 'rb') as stream:
    stream.prefetch(file_size=len(old))
    assert stream.read() == old
_, output, error = client.exec_command("/root/miniconda3/envs/bdetr/bin/python -c 'import json, shutil; print(json.dumps(shutil.disk_usage(\"/root/autodl-tmp\")._asdict()))'", timeout=30)
disk = json.loads(output.read())
assert output.channel.recv_exit_status() == 0, error.read().decode()
addition = '\n\n### 20.100 独立逐级选择诊断准备及后续研究范围（' + when + '）\n\n'
addition += (
    '本节落实用户最新分析中的逐级诊断需求。当前正确mesh配对训练、终态审计、唯一9508正式评估'
    '和§20.99条件接续保持原设置；没有向在跑快照加入新结构或启动第二份训练。本次没有查询训练进度，'
    '仍按§20.98的01:13/01:14首次观察计划执行。\n\n'
    '新增`scripts/trace_scanrefer_readout_stages.py`只读取已完成JointRecReadout forward的输出：'
    '原生选择→Parent→应用geometry有效性后的Parent→Geometry→V99未裁决提议→Pareto最终选择。'
    '原生Query索引必须由原evaluator提供；复用实际部署的排序和policy函数，要求全部最终分数'
    '与该forward的runtime逐元素完全一致。保存Top16映射、合法变体、全局Query槽位、variant编号、'
    '各阶段框与Pareto预测增益；不把槽位编号视为跨forward稳定的实例身份，不让GT进入选择。\n\n'
    '**已完成的验证仅为CPU单元测试：** 原bdetr环境6项全部通过，覆盖并列分数的全局编号排序、'
    'geometry有效性引起的Parent变化、V99通过/否决、argmax不变但分数变化时拒绝、输入不变及无梯度。'
    '回执`refine-logs/scanrefer_stage_trace_preparation_20260907_v1/receipt.json`，SHA'
    '`' + hashlib.sha256((root / 'receipt.json').read_bytes()).hexdigest() + '`。'
    '9个依赖完成源码绑定：8个逐字节一致；train_dist_mod仅有此前原生factory接入差异，'
    '本诊断涉及的3个读出函数AST一致，完整差异保存在runtime_diff.txt。没有修改冻结源码。\n\n'
    '**尚未完成：** 实际场景逐级forward、逐级修复/破坏统计、152/179维归一化输入分布比较。'
    '旧rows/native_rows没有中间选择，不能从它们补造这些结果。后续独立运行须绑定实际权重、'
    'mesh版本、表达/root身份及point SHA，再在选择之后计算GT指标。当前仅已有原生→最终聚合差异，'
    '不足以把全部退化归因为旧读出不兼容。\n\n'
    '后续仍先读本轮真实Scan终态：达到§20.79底线即尽快进入Nr/Sr REC，'
    '不等待59/51，也不以Nr/Sr Mask增加门槛。若未通过，先定位逐级损失，再决定是否比较固定64点预算'
    '的中心读取与范围分组读取；边界分布/质量对齐和冻结旧读出兼容约束均未实施。'
    '8组×8只代表64个读取槽，需另报重复点及有效独立点数；空区域表示缺少观测，不能预设为错误框。\n\n'
    '本次实查服务器剩余' + str(disk['free']) + ' bytes（约' + format(disk['free'] / 1024**3, '.3f') +
    'GiB）；未新增或删除权重，继续保留当前两臂终点与既有受保护权重。整体三数据集目标仍未达成。\n'
)
new = old + addition.encode('utf-8')
tracker = repo / 'refine-logs/EXPERIMENT_TRACKER.md'
tracker_raw = tracker.read_bytes() + ('\n' + when + ': Master20.100 records isolated readout-stage trace preparation,6 original-environment CPU tests;0 real stage rows/GPU updates. Corrected-mesh training and acceptance queues unchanged.\n').encode()
payloads = {'docs/' + master.name: new, 'refine-logs/EXPERIMENT_TRACKER.md': tracker_raw}
for name in ['scripts/trace_scanrefer_readout_stages.py', 'tests/test_trace_scanrefer_readout_stages.py']:
    raw = (repo / name).read_bytes()
    with sftp.open(runtime + name, 'wx') as stream:
        stream.write(raw)
    with sftp.open(runtime + name, 'rb') as stream:
        assert stream.read() == raw
for name, raw in payloads.items():
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
for name, raw in [('README.md', (root / 'README.md').read_bytes()), ('publish.py', Path(__file__).read_bytes())]:
    (root / name).write_bytes(raw)
    with sftp.open(remote + '/' + name, 'wx') as stream:
        stream.write(raw)
proof = {'section': '20.100', 'time_cst': now.isoformat(), 'bytes': len(new),
         'sha256': hashlib.sha256(new).hexdigest(), 'three_master_copies_equal': True,
         'disk': disk, 'new_weights': 0, 'real_stage_rows': 0, 'training_polled': False}
raw = (json.dumps(proof, indent=2, sort_keys=True) + '\n').encode()
(root / 'handoff_sync_20_100.json').write_bytes(raw)
with sftp.open(remote + '/handoff_sync_20_100.json', 'wx') as stream:
    stream.write(raw)
sftp.close()
client.close()
with (repo / 'MANIFEST.md').open('ab') as stream:
    stream.write(('| ' + now.strftime('%Y-%m-%d %H:%M') + ' | direct implementation | refine-logs/' + root.name + '/receipt.json | preparation | Six CPU stage-trace contract tests;no real metric or current-run change |\n').encode())
print(json.dumps(proof))
