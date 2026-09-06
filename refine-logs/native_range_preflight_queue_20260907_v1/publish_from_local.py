import datetime
import hashlib
import json
import os
from pathlib import Path

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
local = repo / 'refine-logs/native_range_preflight_queue_20260907_v1'
remote = '/root/autodl-tmp/mcln_native_range_preflight_queue_20260907_v1'
master = repo / 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop = Path('C:/Users/gb/Desktop/document') / master.name
old = master.read_bytes()
assert hashlib.sha256(old).hexdigest() == '2606ff81741e6799d9c932fe1b736ddaf1eb986222506ce5951dc2720dbbb95f'
assert desktop.read_bytes() == old
armed_raw = (local / 'armed_receipt.json').read_bytes()
armed = json.loads(armed_raw)
assert armed['status'] == 'armed_waiting_for_scan' and armed['tests_passed'] == 3
assert not armed['native_preflight_started'] and not armed['native_training_started']
launch = json.loads((local / 'launch.json').read_bytes())
assert launch['manifest_sha256'] == hashlib.sha256((local / 'input_manifest.json').read_bytes()).hexdigest()
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
runtime = '/home/gb/new butd/butd_detr-main/MCLN-main/'
with sftp.open(runtime + 'docs/' + master.name, 'rb') as stream:
    stream.prefetch(file_size=len(old))
    assert stream.read() == old
_, output, error = client.exec_command('ps -p 47112,47242,48128 -o pid,stat,etime,args', timeout=30)
processes = output.read().decode()
assert output.channel.recv_exit_status() == 0, error.read().decode()
assert all(str(pid) in processes for pid in [47112, 47242, 48128])
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
when = now.strftime('%Y-%m-%d %H:%M CST')
addition = '\n\n### 20.107 ScanRefer通过后自动接续原生范围GPU预检（' + when + '）\n\n'
addition += ('本轮将§20.105已准备的GPU预检从手动入口改为实际后台条件接续。'
    '03:48:54启动screen`48128.mcln_native_range_preflight_queue_v1`，03:51:06核验其存活、3项原环境CPU条件测试通过，'
    '第一条观察明确读取原posttraining screen47242的实际进程。测试使用明确标注的合成临时文件，只验证“不改选center”“无需达到59/51”“两臂输入身份必须一致”，'
    '不能作为任何实验精度。三份screen47112/47242/48128在本次文档写入前再次核实存活。\n\n'
    '**唯一实际接续链。** 47112完成当前Scan训练及终态模块评估；47242执行原定独立审计、固定三组9508正式评估及正式审计；'
    '48128每240秒等待47242完整退出，随后核对上游manifest、正式receipt/audit SHA、9508逐条指标及输入对应，'
    '只对预先指定extent的原有晋级条件做判断。未通过则写`scanrefer_not_promoted`并结束，不分配Nr/Sr GPU；'
    '通过则使用同一GPU锁运行已有范围原生预检，每数据集16条真实训练输入、2次临时更新，结果须再次核验。'
    '它不会直接宣称Nr/Sr训练完成，也没有新建正式训练任务。\n\n'
    '该接续沿用已通过CPU加载检查的618文件原生快照、锁定Nr3D预训练权重、正确mesh数据与原生butd_cls协议；'
    '已有ScanRefer训练参数、损失、候选、更新步数及运行源码均未改。GPU预检目前尚未开始，其目录本次检查仍不存在。'
    '若启动成功，回执记录实际子进程PID与父queue PID；预检失败保留exit与日志，不自动重试或跳过。'
    '当前源码移除被替代的`launch_native_candidate_range_preflight.py`手动启动入口，历史版本仍在§20.105的归档中；'
    '后续应检查48128，不能再运行旧手动入口造成重复接续。\n\n'
    '最近上游观察03:48:12为每臂256/2482更新，elapsed739.37秒、估计剩余6429.03秒，'
    '实际吞吐仍与§20.106接近，fit约05:30结束的估计不变。03:51:06磁盘剩余' + str(armed['disk']['free']) + ' bytes（'
    + format(armed['disk']['free'] / 1024**3, '.3f') + 'GiB），本轮没有新增权重文件。正式成绩未更新，目标仍active。\n\n')
addition += '接续manifest SHA`' + launch['manifest_sha256'] + '`；armed回执SHA`' + hashlib.sha256(armed_raw).hexdigest() + '`。'
addition += '目录`/root/autodl-tmp/mcln_native_range_preflight_queue_20260907_v1`；后续预检目标目录仍是`/root/autodl-tmp/mcln_native_range_preflight_20260907_v1`。\n'
new = old + addition.encode()
tracker = repo / 'refine-logs/EXPERIMENT_TRACKER.md'
lines = tracker.read_text(encoding='utf-8').splitlines()
lines[2] = 'Updated: ' + when + '. §20.107:conditional native range GPU preflight queue48128 armed,3testsPASS;waits original Scan formal queue47242;no native GPU started.'
lines.insert(6, '| Native range conditional GPU preflight | Queue48128 live;3condition testsPASS | Waits fixed Scan formal promotion+audit;negative stops;positive runs16rows+2steps each;no native training started |')
tracker_raw = ('\n'.join(lines) + '\n').encode()
for name in ['scripts/queue_native_candidate_range_preflight.py', 'tests/test_queue_native_candidate_range_preflight.py']:
    raw = (repo / name).read_bytes()
    assert raw == (local / name).read_bytes()
    with sftp.open(runtime + name, 'wb') as stream:
        stream.write(raw)
    with sftp.open(runtime + name, 'rb') as stream:
        assert stream.read() == raw
retired = runtime + 'scripts/launch_native_candidate_range_preflight.py'
with sftp.open(retired, 'rb') as stream:
    assert stream.read() == (repo / 'refine-logs/native_range_preparation_20260907_v2/launch_conditional.py').read_bytes()
sftp.remove(retired)
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
proof = {'time_cst': now.isoformat(), 'section': '20.107', 'bytes': len(new), 'sha256': hashlib.sha256(new).hexdigest(),
    'three_master_copies_equal': master.read_bytes() == desktop.read_bytes() == new,
    'training_screen': 47112, 'formal_queue_screen': 47242, 'native_preflight_queue_screen': 48128,
    'native_preflight_started': False, 'nr3d_sr3d_training_started': False, 'goal_complete': False}
for name, raw in [('handoff_sync_20_107.json', (json.dumps(proof, indent=2, sort_keys=True) + '\n').encode()),
                  ('publish_from_local.py', Path(__file__).read_bytes()),
                  ('verify_from_local.py', Path('C:/Users/gb/.codex/tmp/verify_mcln_native_range_queue_20260907.py').read_bytes())]:
    (local / name).write_bytes(raw)
    with sftp.open(remote + '/' + name, 'wx') as stream:
        stream.write(raw)
sftp.close()
client.close()
print(json.dumps(proof), flush=True)
