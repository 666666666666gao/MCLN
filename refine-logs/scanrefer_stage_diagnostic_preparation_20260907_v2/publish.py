import datetime
import hashlib
import json
import os
from pathlib import Path

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
root = repo / 'refine-logs/scanrefer_stage_diagnostic_preparation_20260907_v2'
master = repo / 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop = Path('C:/Users/gb/Desktop/document/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md')
old = master.read_bytes()
assert hashlib.sha256(old).hexdigest() == '88cfa3e0c1a715f265883550026e6d50b3aad31e0b4904d4fd584ed7038958c9'
assert desktop.read_bytes() == old
receipt = json.loads((root / 'receipt.json').read_bytes())
imports = json.loads((root / 'import_receipt.json').read_bytes())
assert receipt['tests'] == {'tests': 13, 'failures': 0, 'errors': 0, 'skipped': 0}
assert receipt['real_rows_traced'] == receipt['gpu_forwards'] == 0
assert imports['status'] == 'runtime_imports_pass' and len(imports['modules']) == 10
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
when = now.strftime('%Y-%m-%d %H:%M CST')
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
runtime = '/home/gb/new butd/butd_detr-main/MCLN-main/'
remote = '/root/autodl-tmp/mcln_scanrefer_stage_diagnostic_preparation_20260907_v2'
with sftp.open(runtime + 'docs/' + master.name, 'rb') as stream:
    stream.prefetch(file_size=len(old))
    assert stream.read() == old
addition = '\n\n### 20.101 真实逐级诊断的独立入口准备（' + when + '）\n\n'
addition += (
    '在§20.100纯读出helper之后，新增`diagnose_scanrefer_readout_stages.py`独立入口，'
    '从既有正式evaluator复制相同forward/选择流程，并复用其训练终点验证及指标函数。'
    '原`evaluate_scanrefer_local_visual_official.py`逐字未改，当前训练、formal及条件接续快照未改；'
    '本节没有GPU运行或新指标。\n\n'
    '新入口必须绑定已经完成、controller退出0且独立审计通过的正式参考结果及7个实际文件SHA，'
    '沿用其训练起点/终点、data_root、mesh文件表、loader和随机种子。逐行记录表达/root/point身份、'
    '6个阶段的Query槽位/variant/框/IoU、Top16候选与合法性、Pareto提议及增益；GT只在选择之后'
    '计算IoU。Parent152维和Geometry179维的实际归一化输入另记录合法候选计数、均值、RMS、极值。'
    '两臂合法候选总体可能不同，统计差异不能单独证明归一化或兼容性是根因。\n\n'
    '输出固定为独立`diagnostic_result/`，标记`formal_rows=0`、`diagnostic_rows=9508`、'
    '`used_for_promotion=false`。逐级修复/破坏使用原严格IoU阈值，并单列诊断重跑与参考正式结果'
    '之间的Query、变体和IoU差异；不能把重跑差异悄悄当成原正式指标。输入point身份不一致时拒绝汇总。'
    '同Query槽位而框不同，仍不等于已经证明同一语义实例。\n\n'
    '**实际准备验证：** 原bdetr环境13项CPU测试和direct-file CLI检查通过；10个运行依赖从'
    '隔离scripts及冻结model_source正确导入，CUDA隐藏。V1仅完成module-style CLI；V2显式固定'
    '运行时scripts搜索范围，避免从可变工作区导入。测试包括合法特征掩码、原forward不变、严格阈值'
    '修复/破坏、JSON排序往返、point/root不一致拒绝、参考预测漂移记录。没有实际场景forward。\n\n'
    '准备目录`refine-logs/scanrefer_stage_diagnostic_preparation_20260907_v2/`；测试回执SHA`'
    + hashlib.sha256((root / 'receipt.json').read_bytes()).hexdigest() + '`；运行依赖导入回执SHA`'
    + hashlib.sha256((root / 'import_receipt.json').read_bytes()).hexdigest() + '`。'
    '脚本来源、实际配置要求与未完成边界见该目录README。\n\n'
    '**下一步仍由当前Scan结果决定。** 此诊断尚未生成绑定当前结果的运行manifest，也未排队或启动；'
    '需要先取得现有formal及独立审计结果。Scan通过即优先接续Nr/Sr；若未通过，再用本入口定位逐级损失。'
    '没有新增权重或重复训练，本次也没有提前查询GPU训练进度。三数据集目标继续active。\n'
)
new = old + addition.encode()
tracker = repo / 'refine-logs/EXPERIMENT_TRACKER.md'
lines = tracker.read_text(encoding='utf-8').splitlines()
lines[2] = 'Updated: ' + when + '. Acceptance: §20.79; active Scan queues: §20.98/20.99; isolated diagnostic preparation: §20.100/20.101.'
tracker_raw = ('\n'.join(lines) + '\n\n' + when + ': Master20.101 records isolated real-forward diagnostic entry preparation,13 CPU tests and10 bound runtime imports. No GPU diagnostic launched;current Scan and acceptance queues unchanged.\n').encode()
for name in ['scripts/diagnose_scanrefer_readout_stages.py', 'scripts/scanrefer_stage_diagnostics.py',
             'tests/test_scanrefer_stage_diagnostics.py']:
    raw = (repo / name).read_bytes()
    with sftp.open(runtime + name, 'wx') as stream:
        stream.write(raw)
    with sftp.open(runtime + name, 'rb') as stream:
        assert stream.read() == raw
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
for name, raw in [('README.md', (root / 'README.md').read_bytes()), ('publish.py', Path(__file__).read_bytes())]:
    (root / name).write_bytes(raw)
    with sftp.open(remote + '/' + name, 'wx') as stream:
        stream.write(raw)
proof = {'section': '20.101', 'time_cst': now.isoformat(), 'bytes': len(new),
         'sha256': hashlib.sha256(new).hexdigest(), 'three_master_copies_equal': True,
         'diagnostic_job_launched': False, 'real_stage_rows': 0, 'training_polled': False}
raw = (json.dumps(proof, indent=2, sort_keys=True) + '\n').encode()
(root / 'handoff_sync_20_101.json').write_bytes(raw)
with sftp.open(remote + '/handoff_sync_20_101.json', 'wx') as stream:
    stream.write(raw)
sftp.close()
client.close()
with (repo / 'MANIFEST.md').open('ab') as stream:
    stream.write(('| ' + now.strftime('%Y-%m-%d %H:%M') + ' | direct implementation | refine-logs/' + root.name + '/receipt.json | preparation | Isolated stage diagnostic,13 CPU tests and10 runtime imports;0 scene forwards |\n').encode())
print(json.dumps(proof))
