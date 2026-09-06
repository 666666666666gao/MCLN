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
assert hashlib.sha256(old).hexdigest() == '383deaf684974e04ab3b893a3f8f86adb7d13b7b931fff487f3585464e9f9f68'
assert desktop.read_bytes() == old
formal = repo / 'refine-logs/scanrefer_local_visual_official_20260906_v3'
audit = json.loads((formal / 'result/independent_audit.json').read_bytes())
result = json.loads((formal / 'result/receipt.json').read_bytes())
assert audit['schema'] == 'mcln-scanrefer-local-visual-official-audit-v2'
assert audit['integrity_pass'] and audit['formal_rows'] == result['formal_rows'] == 9508
assert audit['receipt_sha256'] == hashlib.sha256((formal / 'result/receipt.json').read_bytes()).hexdigest()
assert not audit['promotion']['advance_to_nr3d_sr3d_rec']
assert (formal / 'controller.exit').read_text().strip() == '0'
probe = repo / 'refine-logs/scanrefer_local_visual_mesh_preflight_20260906_v1'
probe_receipt = json.loads((probe / 'receipt.json').read_bytes())
assert probe_receipt['status'] == 'pass' and probe_receipt['disposable_optimizer_steps'] == 2
assert probe_receipt['data_root'] == '/root/autodl-tmp/DATA_ROOT_mcln_meshsp/'
assert (probe / 'controller.exit').read_text().strip() == '0'
pair = repo / 'refine-logs/scanrefer_local_visual_mesh_pair_20260906_v1'
launch = json.loads((pair / 'launch.json').read_bytes())
startup = json.loads((pair / 'startup.json').read_bytes())
schedule = json.loads((pair / 'first_update_observation_schedule.json').read_bytes())
assert startup['process_live'] and startup['remote_training_pid'] == schedule['remote_training_pid']
cleanup_dir = repo / 'refine-logs/weight_cleanup_20260906_v2'
cleanup = json.loads((cleanup_dir / 'receipt.json').read_bytes())
assert cleanup['status'] == 'complete' and len(cleanup['deleted']) == 2
assert len(cleanup['protected_sha256_before_and_after_verified']) == 6
cpu_dir = repo / 'refine-logs/native_local_preflight_preparation_20260906_v2'
cpu = json.loads((cpu_dir / 'receipt.json').read_bytes())
assert cpu['gpu_forwards'] == cpu['optimizer_steps'] == cpu['checkpoint_writes'] == 0
assert cpu['old_root_point_samples_verified_equal'] == 64
test_dir = repo / 'refine-logs/scanrefer_local_visual_mesh_audit_preparation_20260906_v1'
test = json.loads((test_dir / 'receipt.json').read_bytes())
assert test['exit_code'] == 0 and '12 passed' in test['stdout']
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
when = now.strftime('%Y-%m-%d %H:%M CST')
rows = []
for arm, label in [('protected_v99', '受保护 V99 控制'), ('local_v99', '旧数据训练的局部视觉终点')]:
    value = audit['metrics'][arm]
    rows.append('| %s | %d / %d | %.6f%% / %.6f%% | %d / %d | %.8f%% |' % (
        label, value['rec_hits025'], value['rec_hits050'], value['rec_hits025'] * 100 / 9508,
        value['rec_hits050'] * 100 / 9508, value['mask_hits025'], value['mask_hits050'], value['mask_miou']))
addition = '\n\n### 20.97 正确 mesh 数据正式复核仍未晋级；从受保护起点重复训练配对（' + when + '）\n\n'
addition += ('§20.96 的修正正式评估 v3 已于 `' + result['time_cst'] + '` 实际完成，controller 退出0。'
    '此次使用 DATA_ROOT_mcln_meshsp/，完整9508条、两臂相同采样点；训练更新与权重写入均为0。\n\n'
    '| 系统 | REC hits@0.25 / @0.50 | REC Acc@0.25 / @0.50 | Mask hits@0.25 / @0.50 | Mask mIoU |\n'
    '|---|---:|---:|---:|---:|\n' + '\n'.join(rows) + '\n\n')
addition += ('局部模型相对同数据控制：REC@0.25修复42/破坏63，净−21；REC@0.50修复76/破坏129，净−53；'
    'Mask净+1/−14。三个Scan Mask论文底线通过，两个REC相对历史和相对控制的四项检查均失败，'
    '**不能晋级Nr3D/Sr3D训练**。受保护控制5570/4797较历史5572/4797仍少2/0；没有精确恢复历史源码，'
    '不将这2个命中归因于新方法，也不在缺少证据时归因于浮点噪声。\n\n'
    '同一forward在V99之前的原生REC为控制5514/4409、local5512/4433，净−2/+24。'
    '完整系统的严格阈值退化与原生变化不同，表明需继续研究表征更新和冻结读出的关系；'
    '但当前未完整区分实例身份错误与同实例低质量框，不把IoU转移直接写成实例消歧结论。\n\n'
    '实际独立CPU审计重新计算全量指标、修复/破坏和晋级条件，核对逐行/采样点身份、614文件冻结源码、'
    '评估后312个mesh验证superpoint及4个受保护系统artifact和local终点的文件SHA。'
    '评估器自身同时检查内存模型状态未变；CPU审计不是第二次GPU复算。\n\n'
    '- 正式receipt SHA：`' + audit['receipt_sha256'] + '`。\n'
    '- 独立审计 SHA：`' + hashlib.sha256((formal / 'result/independent_audit.json').read_bytes()).hexdigest() + '`。\n'
    '- 全量结果：`refine-logs/scanrefer_local_visual_official_20260906_v3/result/`。\n\n')
addition += ('**下一步已固定并开始执行：只修正训练输入，重复原配对。** 计划为'
    '`docs/SCANREFER_LOCAL_VISUAL_MESH_REPEAT_2026-09-06.md`，启动前SHA为`' + json.loads((pair / 'input_manifest.json').read_bytes())['plan_sha256'] + '`。'
    '仍采用相同614文件模型源码、E71预训练、原局部读取结构、GT原生损失和冻结V99；'
    '两臂各一遍29778条fit/2482更新，batch12、末批6；6887条模块holdout只作固定对照，'
    '不选择epoch。没有同步加入新的评分、匹配或损失改动。\n\n')
addition += ('修正数据的16条真实GPU预检于`' + probe_receipt['time_cst'] + '`通过：'
    '零初始化原生/V99输出一致，2次一次性更新后局部点编码及Q/K/V梯度有效，冻结参数/buffer/读出不变，完整权重写入0。'
    '预检receipt SHA：`' + hashlib.sha256((probe / 'receipt.json').read_bytes()).hexdigest() + '`。'
    '正式训练重新实例化受保护权重，没有继承预检的两次更新。\n\n'
    '配对训练启动时间`' + launch['time_cst'] + '`，screen `' + launch['screen_session'][0] + '`；'
    '目录`/root/autodl-tmp/mcln_scanrefer_local_visual_mesh_pair_20260906_v1`。此处只记录已核实的启动，'
    '不是训练终点或新指标；首批更新与ETA由独立进度观察补充。训练/预检receipt明确记录data_root与superpoint清单SHA，'
    '终态审计检查命令中的实际data_root并重查1201/312文件。更新后的审计与数据契约在原环境完成12项CPU检查。\n\n')
addition += ('最近实际进程观察`' + startup['time_cst'] + '`：训练PID`' + str(startup['remote_training_pid'])
    + '`存活，正在原生文本解析，尚未观察到优化器更新。一次性观察器本地PID`'
    + str(schedule['local_observer_pid']) + '`已启动，首次观察`' + schedule['first_check_cst']
    + '`，此后240秒，达到64步或进程实际终止后退出。该时间依据上一轮baseline1790.91秒和本轮约7分钟输入初始化估计，'
    '不是完成承诺；不得因观察超时重启训练。\n\n')
addition += ('**Nr/Sr准备保留，训练尚未启动。** 修正数据的CPU输入预检于`' + cpu['time_cst'] + '`完成：'
    '32条固定记录、两种增强状态合计64份真实点样本；64份点SHA及已记录字段与旧预检一致。'
    '原始表达/目标身份由原生构造器核对；没有宣称逐元素比较全部token map。新的superpoint身份另外记录。'
    'GPU前向/优化器更新/权重写入均0。CPU回执SHA：`' + hashlib.sha256((cpu_dir / 'receipt.json').read_bytes()).hexdigest() + '`。'
    '对应GPU条件启动器已经准备，但正确Scan v3未通过，因此没有执行，也没有新Nr/Sr REC结果。\n\n'
    '**磁盘与保留：** 22:10:21启动前实查服务器剩余9606447104 bytes（约8.95GiB），GPU仅1MiB、无计算进程。'
    '此前已释放8.375GiB的清理记录继续有效；本轮复用原预训练文件，不新增完整副本，训练只写两臂各一个终点。'
    '保留日志、逐行结果和受保护最好；没有删除正在被使用的权重。Scan达到历史/配对REC不退化及Scan Mask底线即转Nr/Sr，'
    '不等待59/51，Nr/Sr Mask不设门。三数据集目标仍未完成，goal保持active。\n')
addition += ('随后于`' + cleanup['time_cst'] + '`按用户磁盘清理要求，删除旧配对的'
    '`control_local_visual_state.pt`与`local_local_visual_state.pt`：两者训练审计已通过、固定终点结果已封存，'
    '正确数据的formal v3确认local不达标，旧进程已终止，新轮artifact输入不依赖这两个文件。'
    '删除前复核大小、SHA和单链接身份，删除后复核六个受保护权重SHA一致；日志、逐行结果及旧审计保留。'
    '本次释放1238966272 allocated bytes（约1.154GiB），实查剩余10843734016 bytes（约10.10GiB）。'
    '证据`refine-logs/weight_cleanup_20260906_v2/receipt.json`，SHA`'
    + hashlib.sha256((cleanup_dir / 'receipt.json').read_bytes()).hexdigest() + '`。旧权重已按此记录退休，'
    '以后不可再声称其文件仍可重载；历史审计是删除前实际完成的证据。\n')
new = old + addition.encode('utf-8')
tracker = repo / 'refine-logs/EXPERIMENT_TRACKER.md'
lines = tracker.read_text(encoding='utf-8').splitlines()
lines[2] = 'Updated: ' + when + '. Acceptance: §20.79; corrected-data Scan result and repeat: §20.97.'
for index, line in enumerate(lines):
    if line.startswith('| ScanRefer local visual official |'):
        lines[index] = '| ScanRefer local visual official | Correct mesh v3 complete9508;independent audit PASS;local5549/4744 vs protected5570/4797 | REC promotion FAIL;ScanMask floors PASS;see20.97 |'
    if line.startswith('| ScanRefer candidate local visual |'):
        lines[index] = '| ScanRefer candidate local visual | Old-root endpoint sealed negative;correct-mesh16-row preflight PASS;new fixed2482/arm pair launched from E71 | Terminal and new formal metrics pending;see20.97 |'
    if line.startswith('| ScanRefer local visual endpoint audit |'):
        lines[index] = '| ScanRefer local visual endpoint audit | Old2482-step states audited before retirement;two failed weights deleted22:21 with logs/rows/proof retained | New mesh audit entry12CPUtests PASS;new trained endpoint pending;see20.97 |'
tracker_raw = ('\n'.join(lines) + '\n\n' + when + ': Master20.97 records corrected-mesh9508 failure,actual GPU preflight PASS and same-setting Scan training repeat launch. Nr/Sr not launched;goal active.\n').encode()
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
runtime = '/home/gb/new butd/butd_detr-main/MCLN-main/'
for directory in [formal, probe, pair, cpu_dir, test_dir, cleanup_dir]:
    remote_run = '/root/autodl-tmp/mcln_' + directory.name + '/'
    target_name = 'launch.json' if directory == pair else 'result/independent_audit.json' if directory == formal else 'receipt.json'
    with sftp.open(remote_run + target_name, 'rb') as stream:
        assert stream.read() == (directory / target_name).read_bytes()
with sftp.open(runtime + relative, 'rb') as stream:
    stream.prefetch(file_size=len(old))
    assert stream.read() == old
for name, raw in [(relative, new), ('refine-logs/EXPERIMENT_TRACKER.md', tracker_raw)]:
    with sftp.open(runtime + name, 'wb') as stream:
        stream.set_pipelined(True)
        stream.write(raw)
    with sftp.open(runtime + name, 'rb') as stream:
        stream.prefetch(file_size=len(raw))
        assert stream.read() == raw
for name in ['docs/SCANREFER_LOCAL_VISUAL_MESH_REPEAT_2026-09-06.md', 'docs/NR_SR_NATIVE_DATA_PREFLIGHT_2026-09-06.md',
             'scripts/run_scanrefer_local_visual_pair.py', 'scripts/audit_scanrefer_local_visual_pair.py',
             'scripts/scanrefer_data_contract.py', 'scripts/run_native_candidate_local_preflight.py']:
    raw = (repo / name).read_bytes()
    with sftp.open(runtime + name, 'wb') as stream:
        stream.write(raw)
    with sftp.open(runtime + name, 'rb') as stream:
        assert stream.read() == raw
sftp.close()
client.close()
master.write_bytes(new)
desktop.write_bytes(new)
assert master.read_bytes() == desktop.read_bytes() == new
tracker.write_bytes(tracker_raw)
(repo / ('refine-logs/EXPERIMENT_TRACKER_' + now.strftime('%Y%m%d_%H%M%S') + '.md')).write_bytes(tracker_raw)
proof = {'section': '20.97', 'time_cst': now.isoformat(), 'three_master_copies_equal': True,
    'bytes': len(new), 'sha256': hashlib.sha256(new).hexdigest(), 'formal_promotion': False,
    'correct_mesh_preflight': 'pass', 'correct_mesh_pair': 'launched; terminal pending'}
(pair / 'handoff_sync_20_97.json').write_bytes((json.dumps(proof, indent=2) + '\n').encode())
(pair / 'publish_launch.py').write_bytes(Path(__file__).read_bytes())
with (repo / 'MANIFEST.md').open('ab') as stream:
    stream.write(('| ' + now.strftime('%Y-%m-%d %H:%M') + ' | /run-experiment | docs/SCANREFER_LOCAL_VISUAL_MESH_REPEAT_2026-09-06.md | run | §20.97 corrected Scan formal FAIL;actual mesh GPU preflight PASS;fixed pair launched |\n').encode())
print(json.dumps(proof))
