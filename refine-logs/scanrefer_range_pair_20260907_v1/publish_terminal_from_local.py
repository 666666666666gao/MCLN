import datetime
import hashlib
import json
import os
from pathlib import Path

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
local = repo / 'refine-logs/scanrefer_range_pair_20260907_v1'
formal = repo / 'refine-logs/scanrefer_range_official_20260907_v1'
remote = '/root/autodl-tmp/mcln_scanrefer_range_pair_20260907_v1'
remote_formal = '/root/autodl-tmp/mcln_scanrefer_range_official_20260907_v1'
archive = '/root/autodl-tmp/mcln_scanrefer_range_terminal_check_20260907_v1'
runtime = '/home/gb/new butd/butd_detr-main/MCLN-main/'
master = repo / 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop = Path('C:/Users/gb/Desktop/document') / master.name
old = master.read_bytes()
assert hashlib.sha256(old).hexdigest() == 'd2f19313cb28f85f62638f93ba699ccd23bb1c76cdca1f97f0098c111799c77e'
assert desktop.read_bytes() == old
check_raw = (local / 'terminal_collection_check.json').read_bytes()
check = json.loads(check_raw)
audit_raw = (local / 'independent_audit.json').read_bytes()
audit = json.loads(audit_raw)
assert hashlib.sha256(audit_raw).hexdigest() == check['training_independent_audit_sha256']
assert hashlib.sha256((local / 'receipt.json').read_bytes()).hexdigest() == audit['receipt_sha256'] == check['training_receipt_sha256']
assert audit['integrity_pass'] and not audit['development_dual_rec_nonregression']
assert audit['metrics'] == check['module_metrics']
assert all(item['optimizer_steps'] == 2482 and item['optimizer_parameter_tensors'] == 76 for item in audit['checkpoints'].values())
observation_path = sorted(formal.glob('observation_*.json'))[-1]
observation = json.loads(observation_path.read_bytes())
assert observation['formal_children']['returncode'] == 0
assert observation['latest_progress']['total'] == 9508
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
assert (now - datetime.datetime.fromisoformat(observation['time_cst'])).total_seconds() < 900
when = now.strftime('%Y-%m-%d %H:%M CST')
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
with sftp.open(runtime + 'docs/' + master.name, 'rb') as stream:
    stream.prefetch(file_size=len(old))
    assert stream.read() == old
for name in ['receipt.json', 'independent_audit.json', 'controller.exit']:
    with sftp.open(remote + '/' + name, 'rb') as stream:
        assert stream.read() == (local / name).read_bytes()
with sftp.open(remote_formal + '/input_manifest.json', 'rb') as stream:
    raw = stream.read()
    assert raw == (formal / 'input_manifest.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == check['formal_manifest_sha256']
_, output, error = client.exec_command('ps -p 50454,50455,47245,48128 -o pid,ppid,stat,etime,args', timeout=30)
processes = output.read().decode()
assert output.channel.recv_exit_status() == 0, error.read().decode()
assert all(any(line.split()[0] == str(pid) for line in processes.splitlines()[1:]) for pid in [50454,50455,47245,48128])
assert 'controller.exit' not in sftp.listdir(remote_formal)
addition = '\n\n### 20.110 范围配对完整终态负结果；独立审计通过并启动固定正式三组评估（' + when + '）\n\n'
addition += ('原47112训练任务已正常结束，controller.exit=0。两臂各完成2482次更新和完整6887条terminal评估；'
    '独立CPU审计确认fit遍历及配对点身份一致、冻结核心参数/缓冲区和Parent/Geometry/V99权重及元数据未变，'
    '每臂76个有优化器状态的参数张量均为2482步。两个固定终点SHA与§20.109一致，未重新训练或更换epoch。\n\n'
    '| 6887条模块留出阶段 | REC hits@0.25/@0.50 | Mask hits@0.25/@0.50 | Mask mIoU |\n'
    '|---|---:|---:|---:|\n'
    '| 零更新起点，两臂相同 | 6684/6426 | 6511/6097 | 77.8108607870% |\n'
    '| center终点，control | 6674/6416 | 6510/6087 | 77.7892916000% |\n'
    '| extent终点，local | 6672/6413 | 6510/6087 | 77.7868386765% |\n\n'
    '**本版本未通过模块REC双阈值不退化。** extent相对起点为−12/−13：@0.25修复16、破坏28；@0.50修复61、破坏74。'
    'extent相对center为−2/−3：修复/破坏分别6/8、16/19。按106个physical space聚类的2000次bootstrap，'
    'extent−center的95%区间为@0.25 [−0.1670,+0.1074]pp、@0.50 [−0.2766,+0.1924]pp，均跨0；区间只作诊断，不能把未过线改写为等效或晋级。'
    '此集合仍是主干见过的训练场景；不是9508条正式指标，也没有可靠的选中实例身份标签，不能从IoU变化直接宣布选错实例或仅框边界变差。'
    '结果未支持本次固定八区域、总64槽位读取优于匹配中心控制，不能外推为所有范围读取方法无效。\n\n'
    '**固定正式评估已实际接续。** 独立完整性审计PASS后，原47242队列于06:08:32 CST启动flock PID50454，'
    '实际Python子进程50455；正式输入manifest绑定训练receipt、audit及两个终点。'
    '三组仍为protected_v99、center_v99、local_v99(extent)，唯一预定晋级候选仍是local_v99，'
    '不根据模块结果临时换center。原协议允许完整性通过后执行一次固定正式比较，模块负结果如实保留。'
    '06:16:44实查已输出首12/9508行，正式完整result/receipt尚未产生；本次文档写入前再查正式父子进程及两个接续队列均存活。'
    '正式脚本对同一批点同步保存原生与系统REC，结束后原队列独立审计修复/破坏、IoU区间转移和系统相对原生的收益变化。\n\n'
    'Nr/Sr接续48128仍等待固定正式结果；只在ScanRefer历史REC5572/4797及同次保护不退化、ScanMask三底线均通过后运行原生GPU预检。'
    '目前Nr/Sr训练未开始，实际训练预算尚未锁定；不把已准备的CPU代码或条件队列当作训练成绩。'
    '范围读取尚无有效正证据，当前不叠加边界分布、质量头或兼容Loss，等待唯一正式结果。\n\n')
addition += ('06:16:44磁盘剩余' + str(observation['disk']['free']) + ' bytes（' + format(observation['disk']['free']/1024**3, '.3f')
    + 'GiB）；本轮未删当前终点或任何受保护权重。训练receipt SHA`' + check['training_receipt_sha256']
    + '`；独立审计SHA`' + check['training_independent_audit_sha256'] + '`；正式manifest SHA`' + check['formal_manifest_sha256'] + '`。\n')
new = old + addition.encode()
tracker = repo / 'refine-logs/EXPERIMENT_TRACKER.md'
lines = tracker.read_text(encoding='utf-8').splitlines()
lines[2] = 'Updated: ' + when + '. §20.110: range2482-step endpoints+6887 terminal complete;integrityPASS,extent-baseline REC-12/-13 and extent-center-2/-3;fixed9508 formal three-arm eval live.'
for index,line in enumerate(lines):
    if line.startswith('| ScanRefer matched center/extent range reading |'):
        lines[index] = '| ScanRefer matched center/extent range reading | Complete2482updates/arm+6887terminal;independent integrityPASS,qualityFAIL | extent REC6672/6413 vs baseline6684/6426 and center6674/6416;fixed formal PID50454/50455 live;no promotion yet |'
tracker_raw = ('\n'.join(lines) + '\n').encode()
for name,raw in [('docs/' + master.name,new),('refine-logs/EXPERIMENT_TRACKER.md',tracker_raw)]:
    with sftp.open(runtime + name,'wb') as stream:
        stream.set_pipelined(True)
        stream.write(raw)
    with sftp.open(runtime + name,'rb') as stream:
        stream.prefetch(file_size=len(raw))
        assert stream.read() == raw
master.write_bytes(new)
desktop.write_bytes(new)
tracker.write_bytes(tracker_raw)
proof = {'time_cst':now.isoformat(),'section':'20.110','bytes':len(new),'sha256':hashlib.sha256(new).hexdigest(),
    'three_master_copies_equal':master.read_bytes()==desktop.read_bytes()==new,'processes':processes,
    'terminal_collection_check_sha256':hashlib.sha256(check_raw).hexdigest(),
    'training_receipt_sha256':check['training_receipt_sha256'],'training_independent_audit_sha256':check['training_independent_audit_sha256'],
    'formal_manifest_sha256':check['formal_manifest_sha256'],'formal_observation_sha256':hashlib.sha256(observation_path.read_bytes()).hexdigest(),
    'formal_observation_file':observation_path.name,'formal_results_present':False,'goal_complete':False}
proof_raw = (json.dumps(proof,indent=2,sort_keys=True)+'\n').encode()
sftp.mkdir(archive)
for name,raw in [('terminal_collection_check.json',check_raw),('handoff_sync_20_110.json',proof_raw),
                 ('publish_terminal_from_local.py',Path(__file__).read_bytes()),
                 ('collect_terminal_from_local.py',(local/'collect_terminal_from_local.py').read_bytes())]:
    (local/name).write_bytes(raw)
    with sftp.open(archive+'/'+name,'wx') as stream:
        stream.write(raw)
    with sftp.open(archive+'/'+name,'rb') as stream:
        assert stream.read()==raw
observer_raw = Path('C:/Users/gb/.codex/tmp/observe_mcln_range_formal_20260907.py').read_bytes()
(formal/'observe_formal_from_local.py').write_bytes(observer_raw)
with sftp.open(archive+'/observe_formal_from_local.py','wx') as stream:
    stream.write(observer_raw)
with sftp.open(archive+'/observe_formal_from_local.py','rb') as stream:
    assert stream.read()==observer_raw
sftp.close()
client.close()
print(json.dumps(proof),flush=True)
