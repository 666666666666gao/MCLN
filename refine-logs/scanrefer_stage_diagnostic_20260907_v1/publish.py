import datetime
import hashlib
import json
import os
from pathlib import Path

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
logs = repo / 'refine-logs'
pair = logs / 'scanrefer_local_visual_mesh_pair_20260906_v1'
formal = logs / 'scanrefer_local_visual_mesh_official_20260906_v1'
queue = logs / 'scanrefer_local_visual_mesh_acceptance_queue_20260906_v1'
trace = logs / 'scanrefer_stage_diagnostic_20260907_v1'
master = repo / 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop = Path('C:/Users/gb/Desktop/document/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md')
old = master.read_bytes()
assert hashlib.sha256(old).hexdigest() == 'c95fb921c7d2845ff5c864b92aafd948ebca95245c672c2ca627b19ed691da2f'
assert desktop.read_bytes() == old
training = json.loads((pair / 'receipt.json').read_bytes())
archive = json.loads((pair / 'terminal_archive_verification.json').read_bytes())
receipt = json.loads((formal / 'result/receipt.json').read_bytes())
audit = json.loads((formal / 'result/independent_audit.json').read_bytes())
decision = json.loads((queue / 'decision.json').read_bytes())
launch = json.loads((trace / 'launch.json').read_bytes())
probe = json.loads((trace / 'progress_20260907_020126.json').read_bytes())
schedule = json.loads((trace / 'terminal_observation_schedule.json').read_bytes())
assert audit['integrity_pass'] and not decision['promotion']['advance_to_nr3d_sr3d_rec']
assert audit['receipt_sha256'] == decision['formal_receipt_sha256'] == hashlib.sha256((formal / 'result/receipt.json').read_bytes()).hexdigest()
assert decision['formal_audit_sha256'] == hashlib.sha256((formal / 'result/independent_audit.json').read_bytes()).hexdigest()
assert (queue / 'controller.exit').read_text().strip() == '0'
assert probe['screen_live'] and probe['progress']['SCANREFER STAGE DIAGNOSTIC']['rows'] == 1536
assert launch['formal_rows'] == 0 and schedule['observer_pid'] == 49108
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
when = now.strftime('%Y-%m-%d %H:%M CST')
addition = '\n\n### 20.102 正确mesh训练及正式评估负结果；逐级诊断实测启动（' + when + '）\n\n'
addition += (
    '**本轮训练已经完整结束。** 两臂各2482次更新，29778条fit、6887条模块holdout；'
    '训练于01:16:33完成，退出0，独立CPU审计通过。该holdout仍是预训练主干见过的训练场景。'
    '相同正确mesh起点REC为6684/6426，终点control为6679/6449，local为6670/6405；'
    'local相对起点−14/−21，相对control−9/−44。固定终点仍按原计划进行唯一正式评估，未据开发集选择epoch。\n\n'
    '6887条完整终态和fit点身份记录已归档；命中数本地复算完全一致。Windows本地mIoU复算与'
    '原环境记录最大差4.55e−13pp，采用既有evaluator的1e−8浮点比较精度；未修改模型、损失或指标阈值。'
    '训练回执SHA`c0248a1928306ae73565a5d54c8eb1b56d68de01ca77b3aad1998138bdd075ef`；'
    '独立训练审计SHA`bb270b9ec04259ec20a587a0c9833f2c657abce74fc8f52bf9eeb5469fdc55c3`。\n\n'
    '**9508条正式评估于01:44:34结束，独立审计及条件队列于01:45:41完成，均退出0。** '
    '正确mesh的312个val文件、输入身份、权重及冻结状态核验通过，但质量晋级失败：\n\n'
    '| 同次正式评估 | REC hits@0.25 / @0.50 | REC百分比@0.25 / @0.50 | Mask hits@0.25 / @0.50 | Mask mIoU |\n'
    '|---|---:|---:|---:|---:|\n')
for name, label in [('protected_v99', '受保护V99控制'), ('local_v99', '正确mesh局部视觉终点')]:
    metric = receipt['metrics'][name]
    addition += ('| ' + label + ' | {rec_hits025}/{rec_hits050} | '.format(**metric)
                 + '{:.5f}% / {:.5f}%'.format(metric['rec_hits025'] * 100. / 9508, metric['rec_hits050'] * 100. / 9508)
                 + ' | {mask_hits025}/{mask_hits050} | {mask_miou:.8f}% |\n'.format(**metric))
addition += (
    '\nlocal相对同次保护控制：REC@0.25修复37、破坏64，净−27；@0.50修复73、破坏148，净−75。'
    'Mask两阈值净+6/+1，mIoU+0.009927pp，三项ScanMask底线通过；REC相对历史5572/4797及'
    '同次5570/4797均未过。历史受保护版本没有替换，也未将Mask单项拼入历史最好行。\n\n'
    '同次原生REC：保护5514/4409，local5510/4426，净−4/+17。原生严格阈值略升、完整系统下降的'
    '现象在正确数据重训后仍出现；这支持继续逐级定位，但尚不能证明冻结读出不兼容是唯一原因。'
    '正式审计SHA`' + decision['formal_audit_sha256'] + '`；正式回执SHA`'
    + decision['formal_receipt_sha256'] + '`，位于`refine-logs/' + formal.name + '/result/`。'
    '按§20.79，Nr/Sr预检和训练均未启动。原观察/正式审计队列已正常完成，不要重启。\n\n'
    '**独立逐级诊断已实际启动。** 01:51:47，screen`45890.mcln_scanrefer_stage_diagnostic_v1`，'
    '绑定本次正式结果、两个固定权重、同一正确mesh和原始输入配置；无训练、无新增权重，'
    '`formal_rows=0`、`used_for_promotion=false`。输入manifest SHA`'
    + launch['manifest_sha256'] + '`。\n\n'
    '02:01:26实际观察到1536/9508条，两臂逐级选择已通过每个forward的最终分数完全一致校验；'
    'Parent/Geometry的归一化特征读取也已在真实输入上执行。完整阶段汇总、参考结果一致性和'
    '独立终态审计仍待完成，不能将1536条进度当正式新指标。首批12条6.21秒含启动影响，'
    '没有用其82分钟外推安排后续；按1536条260.54秒的记录，保守剩余约1352秒。\n\n'
    '本地观察器PID49108、会话6419已启动；首查`' + schedule['first_check_cst'] + '`，之后240秒，'
    '结束后只收集诊断输出，不重复训练或启动Nr/Sr。目录`refine-logs/' + trace.name + '/`。'
    '下一步先读取实际逐级损失和归一化统计，再决定范围读取或兼容约束，不恢复旧Gate/Pair扫参。\n\n'
    '01:45实查磁盘剩余' + str(archive['disk']['free']) + ' bytes（约'
    + format(archive['disk']['free'] / 1024**3, '.3f') + 'GiB），6个受保护文件均在；'
    '本轮只生成两臂终点共1238954544 bytes，未下载权重。local终点正在用于诊断，继续保留；'
    '既有累计约9.529GiB清理记录不变。三数据集目标未完成，goal保持active。\n'
)
new = old + addition.encode()
tracker = logs / 'EXPERIMENT_TRACKER.md'
lines = tracker.read_text(encoding='utf-8').splitlines()
lines[2] = 'Updated: ' + when + '. Scan mesh repeat/formal sealed FAIL;actual stage diagnostic1536/9508 and observer49108: §20.102. Acceptance remains §20.79.'
lines.insert(6, '| ScanRefer corrected-mesh final and stage diagnostic | Formal5543/4722 vs protected5570/4797;integrityPASS,RECgateFAIL | Nr/Sr not launched;fixed-weight stage diagnostic45890 observed1536rows;collector49108/session6419 first02:20:58 then240s |')
tracker_raw = ('\n'.join(lines) + '\n\n' + when + ': Master20.102 seals actual Scan mesh training/formal negative result and records running independent stage diagnostic;no native training or new method success.\n').encode()
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
runtime = '/home/gb/new butd/butd_detr-main/MCLN-main/'
remote_trace = '/root/autodl-tmp/mcln_scanrefer_stage_diagnostic_20260907_v1'
with sftp.open(runtime + 'docs/' + master.name, 'rb') as stream:
    stream.prefetch(file_size=len(old))
    assert stream.read() == old
for name in ['probe_from_local.py', 'observe_terminal.py', 'terminal_observation_schedule.json',
             'progress_20260907_015701.json', 'progress_20260907_020126.json']:
    raw = (trace / name).read_bytes()
    with sftp.open(remote_trace + '/' + name, 'wx') as stream:
        stream.write(raw)
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
proof = {'section': '20.102', 'time_cst': now.isoformat(), 'bytes': len(new),
         'sha256': hashlib.sha256(new).hexdigest(), 'three_master_copies_equal': True,
         'formal_receipt_sha256': decision['formal_receipt_sha256'],
         'formal_audit_sha256': decision['formal_audit_sha256'], 'scan_promoted': False,
         'diagnostic_manifest_sha256': launch['manifest_sha256'],
         'diagnostic_observed_rows': 1536, 'diagnostic_observer_pid': 49108, 'goal_complete': False}
raw = (json.dumps(proof, indent=2, sort_keys=True) + '\n').encode()
(trace / 'handoff_sync_20_102.json').write_bytes(raw)
with sftp.open(remote_trace + '/handoff_sync_20_102.json', 'wx') as stream:
    stream.write(raw)
archive_code = Path(__file__).read_bytes()
(trace / 'publish.py').write_bytes(archive_code)
with sftp.open(remote_trace + '/publish.py', 'wx') as stream:
    stream.write(archive_code)
sftp.close()
client.close()
with (repo / 'MANIFEST.md').open('ab') as stream:
    stream.write(('| ' + now.strftime('%Y-%m-%d %H:%M') + ' | direct continuation | refine-logs/' + trace.name + '/launch.json | diagnostic | Actual Scan negative formal result, independent stage diagnostic1536rows;no new training |\n').encode())
print(json.dumps(proof))
