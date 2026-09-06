import datetime
import hashlib
import json
import os
from pathlib import Path

import paramiko


repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
relative = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
master = repo / relative
desktop = Path('C:/Users/gb/Desktop/document/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md')
old = master.read_bytes()
assert hashlib.sha256(old).hexdigest() == '86ab8e118a0127df580f1c61ba2c9ff5cdc09ff9e53e48792316ec7e68d74110'
assert desktop.read_bytes() == old
formal = repo / 'refine-logs/scanrefer_local_visual_official_20260906_v1'
result = formal / 'result'
receipt = json.loads((result / 'receipt.json').read_bytes())
audit = json.loads((result / 'independent_audit.json').read_bytes())
assert receipt['status'] == 'complete' and receipt['formal_rows'] == 9508
assert audit['integrity_pass'] and audit['formal_rows'] == 9508
assert audit['receipt_sha256'] == hashlib.sha256((result / 'receipt.json').read_bytes()).hexdigest()
assert audit['promotion'] == receipt['promotion']
assert (formal / 'controller.exit').read_text().strip() == '0'
decision_note = (formal / 'next_action.txt').read_text(encoding='utf-8')
values = audit['metrics']
native = audit['native_rec_metrics']
comparison = audit['system_local_minus_protected']
native_comparison = audit['native_local_minus_protected']
promotion = audit['promotion']['advance_to_nr3d_sr3d_rec']
outcome = 'PASS' if promotion else 'FAIL'
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
when = now.strftime('%Y-%m-%d %H:%M CST')


def percent(hits):
    return '%.6f%%' % (hits * 100. / 9508)


table = []
for arm, label in [('protected_v99', '受保护V99配对控制'), ('local_v99', '候选局部读取＋原V99')]:
    value = values[arm]
    table.append('| %s | %d / %d | %s / %s | %d / %d | %.8f%% |' % (
        label, value['rec_hits025'], value['rec_hits050'],
        percent(value['rec_hits025']), percent(value['rec_hits050']),
        value['mask_hits025'], value['mask_hits050'], value['mask_miou']))
native_table = []
for arm, label in [('protected_v99', '原核心直接输出'), ('local_v99', '局部读取核心直接输出')]:
    value = native[arm]
    native_table.append('| %s | %d / %d | %s / %s |' % (
        label, value['rec_hits025'], value['rec_hits050'], percent(value['rec_hits025']), percent(value['rec_hits050'])))
effects = '\n'.join('| %s | %d | %d | %+d |' % (
    key, value['repair'], value['damage'], value['net']) for key, value in comparison['effects'].items())
checks = '\n'.join('- `%s`: **%s**。' % (key, 'PASS' if value else 'FAIL')
                   for key, value in audit['promotion']['checks'].items())
ci = comparison['bootstrap']['intervals_95_percent_pp']
addition = '\n\n### 20.96 Scan入口误用旧superpoint路径：9508条结果、根因与修正复核（' + when + '）\n\n'
addition += ('本节记录固定端点的实际9508条验证结果。事后查明入口使用旧mixed superpoint输入，'
             '不满足历史最好所用meshsp协议，不能作为该协议的正式晋级证据。'
             '训练已于20.95完成，本次优化器更新和权重写入均为0；'
             '两个模型读取相同9508条表达和相同采样点。controller退出0；实际终态时间为`'
             + receipt['time_cst'] + '`，评估用时%.2f秒，Torch峰值%.2fMiB。\n\n' % (
                 receipt['elapsed_seconds'], receipt['max_gpu_mib']))
addition += '| 完整系统 | REC hits@0.25 / @0.50 | REC Acc@0.25 / @0.50 | Mask hits@0.25 / @0.50 | Mask mIoU |\n'
addition += '|---|---:|---:|---:|---:|\n' + '\n'.join(table) + '\n\n'
addition += ('原生REC由同一个forward在V99前单独计算，未拼入其他版本Mask：\n\n'
             '| 原生网络 | REC hits@0.25 / @0.50 | REC Acc@0.25 / @0.50 |\n'
             '|---|---:|---:|\n' + '\n'.join(native_table) + '\n\n')
addition += ('局部完整系统相对配对受保护系统逐行变化：\n\n'
             '| 指标 | 修复 | 破坏 | 净变化 |\n|---|---:|---:|---:|\n' + effects + '\n\n')
addition += '原生REC净变化为%+d/%+d，单独保存在审计的`native_local_minus_protected`。\n' % (
    native_comparison['effects']['025']['net'], native_comparison['effects']['050']['net'])
addition += ('按%d个物理空间、2000次seed0 bootstrap，完整系统REC差异95%%区间：'
             '@0.25[%.6f,%.6f]pp；@0.50[%.6f,%.6f]pp。区间用于诊断，不新增晋级门槛。\n\n' % (
                 comparison['physical_spaces'], ci['rec025'][0], ci['rec025'][1], ci['rec050'][0], ci['rec050'][1]))
addition += ('旧输入上的数值门判断：**' + outcome + '**；正确数据版本的晋级尚待重评。Scan REC必须同时达到历史5572/4797与本次配对控制；'
             'Scan Mask必须达到论文58.70/50.70/44.72%。59/51为争取目标，Nr/Sr Mask不设门槛。\n\n'
             + checks + '\n\n' + decision_note + '\n\n')
addition += ('独立CPU审计重新计算了完整系统与原生REC、逐行输入身份、修复/破坏和晋级条件；'
             '复核614个源码文件、正式入口、训练审计和本次local及四份保护artifact的SHA。'
             '本次正式运行内部还校验全部模型state和V99参数未变化。'
             '它不补充缺失的逐行GT实例身份，因此IoU区间转移不能单独解释为选错实例或框回归原因。\n\n')
addition += ('正式receipt SHA`' + audit['receipt_sha256'] + '`；独立审计SHA`'
             + hashlib.sha256((result / 'independent_audit.json').read_bytes()).hexdigest() + '`。'
             '全部逐行记录与protocol位于`refine-logs/scanrefer_local_visual_official_20260906_v1/result/`。'
             '验收器已在原Python3.7/Torch1.10.2环境通过23项单元测试；该测试记录本身不是正式成绩。\n')
new = old + addition.encode('utf-8')
tracker = repo / 'refine-logs/EXPERIMENT_TRACKER.md'
lines = tracker.read_text(encoding='utf-8').splitlines()
lines[2] = 'Updated: ' + when + '. Acceptance: §20.79; actual fixed Scan formal result: §20.96.'
for index, line in enumerate(lines):
    if line.startswith('| ScanRefer local visual official |'):
        value = values['local_v99']
        lines[index] = ('| ScanRefer local visual official | Complete9508 paired rows;actual independent integrity PASS;'
                        'local REC%d/%d;native%d/%d | Old mixed-SP input,numeric gate %s;not protected mesh protocol;v3 corrected reevaluation running;see20.96 |' % (
                            value['rec_hits025'], value['rec_hits050'], native['local_v99']['rec_hits025'],
                            native['local_v99']['rec_hits050'], outcome))
    if line.startswith('| ScanRefer candidate local visual |'):
        lines[index] = ('| ScanRefer candidate local visual | Complete2482 updates/arm+6887 terminal;integrity PASS;'
                        'development REC negative;old-root9508 endpoint complete | Old data-root limitation;corrected v3 result pending;see20.96 |')
tracker_raw = ('\n'.join(lines) + '\n\n' + when + ': Master20.96 records actual fixed9508 Scan result and independent audit. '
               'Promotion ' + outcome + ';three-benchmark objective remains active.\n').encode()
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
runtime = '/home/gb/new butd/butd_detr-main/MCLN-main/'
remote_run = '/root/autodl-tmp/mcln_scanrefer_local_visual_official_20260906_v1/'
for name in ['result/receipt.json', 'result/independent_audit.json']:
    with sftp.open(remote_run + name, 'rb') as stream:
        assert stream.read() == (formal / name).read_bytes()
with sftp.open(runtime + relative, 'rb') as stream:
    stream.prefetch(file_size=len(old))
    assert stream.read() == old
with sftp.open(runtime + relative, 'wb') as stream:
    stream.set_pipelined(True)
    stream.write(new)
with sftp.open(runtime + relative, 'rb') as stream:
    stream.prefetch(file_size=len(new))
    assert stream.read() == new
with sftp.open(runtime + 'refine-logs/EXPERIMENT_TRACKER.md', 'wb') as stream:
    stream.write(tracker_raw)
with sftp.open(remote_run + 'next_action.txt', 'wx') as stream:
    stream.write(decision_note.encode())
corrected = repo / 'refine-logs/scanrefer_local_visual_official_20260906_v3'
observer = Path('C:/Users/gb/.codex/tmp/observe_mcln_local_visual_official_20260906_v3.py').read_bytes()
watch = json.loads((corrected / 'observation_schedule.json').read_bytes())
assert watch['process_live'] and watch['remote_pid'] == 40897
assert hashlib.sha256(observer).hexdigest() == watch['script_sha256']
with sftp.open('/root/autodl-tmp/mcln_scanrefer_local_visual_official_20260906_v3/observe_official.py', 'wx') as stream:
    stream.write(observer)
(corrected / 'observe_official.py').write_bytes(observer)
sftp.close()
client.close()
master.write_bytes(new)
desktop.write_bytes(new)
assert master.read_bytes() == desktop.read_bytes() == new
tracker.write_bytes(tracker_raw)
(repo / ('refine-logs/EXPERIMENT_TRACKER_' + now.strftime('%Y%m%d_%H%M%S') + '.md')).write_bytes(tracker_raw)
proof = {'section': '20.96', 'time_cst': now.isoformat(), 'three_master_copies_equal': True,
         'bytes': len(new), 'sha256': hashlib.sha256(new).hexdigest(), 'formal_promotion': promotion}
(formal / 'handoff_sync_20_96.json').write_bytes((json.dumps(proof, indent=2) + '\n').encode())
(formal / 'publish_terminal.py').write_bytes(Path(__file__).read_bytes())
with (repo / 'MANIFEST.md').open('ab') as stream:
    stream.write(('| ' + now.strftime('%Y-%m-%d %H:%M') + ' | /run-experiment | '
                  'refine-logs/scanrefer_local_visual_official_20260906_v1/result/independent_audit.json | result | '
                  '§20.96 actual9508 paired Scan result,independent integrity PASS;fixed promotion ' + outcome + ' |\n').encode())
print(json.dumps(proof))
