import datetime
import hashlib
import json
import os
from pathlib import Path

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
local = repo / 'refine-logs/range_source_audit_20260907_v1'
remote = '/root/autodl-tmp/mcln_range_source_audit_20260907_v1'
runtime = '/home/gb/new butd/butd_detr-main/MCLN-main/'
master = repo / 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop = Path('C:/Users/gb/Desktop/document') / master.name
old = master.read_bytes()
assert hashlib.sha256(old).hexdigest() == '08dc5ee2a7f89952e53c4caff579fe9cee881d96e4bd8be9e0486774fb021d85'
assert desktop.read_bytes() == old
receipt_raw = (local / 'source_receipt.json').read_bytes()
receipt = json.loads(receipt_raw)
assert len(receipt['sources']) == 7 and not receipt['training_config_changed']
note_name = 'docs/MCLN_RANGE_READER_SOURCE_AUDIT_2026-09-07.md'
note = (repo / note_name).read_bytes()
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
with sftp.open(runtime + 'docs/' + master.name, 'rb') as stream:
    stream.prefetch(file_size=len(old))
    assert stream.read() == old
with sftp.open(runtime + receipt['local_module']['path'], 'rb') as stream:
    assert hashlib.sha256(stream.read()).hexdigest() == receipt['local_module']['sha256']
_, output, error = client.exec_command("/root/miniconda3/envs/bdetr/bin/python -c 'import json,shutil; print(json.dumps(dict(zip((\"total\",\"used\",\"free\"),shutil.disk_usage(\"/root/autodl-tmp\")))))'", timeout=30)
disk = json.loads(output.read())
assert output.channel.recv_exit_status() == 0, error.read().decode()
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
when = now.strftime('%Y-%m-%d %H:%M CST')
addition = '\n\n### 20.108 范围读取来源核对；保持原训练配对（' + when + '）\n\n'
addition += ('等待既定训练期间，核对并锁定PV-RCNN/OpenPCDet及Box-DETR官方源码的两个commit、7个文件SHA与行定位。'
    '新增说明`docs/MCLN_RANGE_READER_SOURCE_AUDIT_2026-09-07.md`，未复制第三方完整源码，没有新增训练或改动正在运行的模型、预算及晋级门。\n\n'
    '**方法边界修正。** RoI网格和空邻域识别均有现成上游实现，不能作为本轮独立创新。本轮八个支撑位置在5厘米半尺寸下限未生效时，等于2×2×2 RoI网格中心；不是六个边界面。'
    '两臂都使用同一分区读出，只比较中心近邻分配与每象限最多8个点的范围分配，64是槽位上限、并非相同有效点数。语言通过已有多模态Query间接影响读取，当前没有新增完整token接口或身份—范围解耦监督。'
    'Box-DETR使用动态的每头二维代理点，因此本轮也不等于其完整移植。上述为代码关系与代数推断，不是性能结果或完整查新结论。\n\n'
    '**实际运行继续。** 04:12:09实查screen47112、47242、48128均存活；队列最近04:08:13记录每臂704/2482更新，elapsed2028.49秒，估计fit剩余5123.09秒。'
    '仍预计约05:30结束fit，随后完成终态模块评估和固定正式评估。原生Nr/Sr GPU预检尚未启动；只由既定48128在Scan通过后接续，不能重复启动。'
    '当前没有新增正式分数，受保护结果保持不变，目标继续active。\n\n')
addition += ('本次实际磁盘剩余' + str(disk['free']) + ' bytes（' + format(disk['free'] / 1024**3, '.3f')
    + 'GiB）；来源说明不产生权重，之前累计清理约9.529GiB未计作本轮新增。来源回执SHA`'
    + hashlib.sha256(receipt_raw).hexdigest() + '`。\n')
new = old + addition.encode()
tracker = repo / 'refine-logs/EXPERIMENT_TRACKER.md'
lines = tracker.read_text(encoding='utf-8').splitlines()
lines[2] = 'Updated: ' + when + '. §20.108: pinned range-reader primary-source audit; Scan pair 704/2482 last04:08, three screens live04:12; no runtime change or new metrics.'
tracker_raw = ('\n'.join(lines) + '\n').encode()
sftp.mkdir(remote)
for name, raw in [('source_receipt.json', receipt_raw), ('source_audit.md', note),
                  ('record_from_local.py', (local / 'record_from_local.py').read_bytes())]:
    with sftp.open(remote + '/' + name, 'wx') as stream:
        stream.write(raw)
    with sftp.open(remote + '/' + name, 'rb') as stream:
        assert stream.read() == raw
for name, raw in [(note_name, note), ('docs/' + master.name, new), ('refine-logs/EXPERIMENT_TRACKER.md', tracker_raw)]:
    with sftp.open(runtime + name, 'wb') as stream:
        stream.set_pipelined(True)
        stream.write(raw)
    with sftp.open(runtime + name, 'rb') as stream:
        stream.prefetch(file_size=len(raw))
        assert stream.read() == raw
master.write_bytes(new)
desktop.write_bytes(new)
tracker.write_bytes(tracker_raw)
proof = {'time_cst': now.isoformat(), 'section': '20.108', 'bytes': len(new),
    'sha256': hashlib.sha256(new).hexdigest(), 'three_master_copies_equal': master.read_bytes() == desktop.read_bytes() == new,
    'source_receipt_sha256': hashlib.sha256(receipt_raw).hexdigest(), 'source_audit_sha256': hashlib.sha256(note).hexdigest(),
    'disk': disk, 'runtime_module_matches_audited_sha': True, 'runtime_changed': False, 'goal_complete': False}
for name, raw in [('handoff_sync_20_108.json', (json.dumps(proof, indent=2, sort_keys=True) + '\n').encode()),
                  ('publish_from_local.py', Path(__file__).read_bytes())]:
    (local / name).write_bytes(raw)
    with sftp.open(remote + '/' + name, 'wx') as stream:
        stream.write(raw)
sftp.close()
client.close()
print(json.dumps(proof), flush=True)
