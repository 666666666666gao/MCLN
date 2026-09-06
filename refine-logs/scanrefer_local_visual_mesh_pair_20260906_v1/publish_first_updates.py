import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
master = repo / 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop = Path('C:/Users/gb/Desktop/document/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md')
old = master.read_bytes()
assert hashlib.sha256(old).hexdigest() == 'fa3d351bb9393d56f717da3a38248868e21a2c4179b9b2c98f35503e6c06e6d2'
assert desktop.read_bytes() == old
pair = repo / 'refine-logs/scanrefer_local_visual_mesh_pair_20260906_v1'
post = repo / 'refine-logs/scanrefer_local_visual_mesh_posttraining_20260906_v1'
checked = json.loads((pair / 'baseline_verification.json').read_bytes())
assert checked['status'] == 'pass' and checked['rows'] == 6887
assert checked['baseline_arms_all_rows_equal'] and checked['old_new_row_identity_and_points_equal']
progress = json.loads((pair / 'progress_20260906_230205.json').read_bytes())
assert progress['process_live'] and progress['progress']['SCANREFER LOCAL VISUAL TRAIN']['step'] == 64
queue = json.loads((post / 'queue_launch.json').read_bytes())
schedule = json.loads((post / 'observation_schedule.json').read_bytes())
assert queue['first_check_cst'] == checked['next_check_cst']
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
when = now.strftime('%Y-%m-%d %H:%M CST')
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
remote_pair = '/root/autodl-tmp/mcln_' + pair.name
remote_post = '/root/autodl-tmp/mcln_' + post.name
for root, folder, name in [(remote_pair, pair, 'baseline_verification.json'), (remote_post, post, 'queue_launch.json')]:
    with sftp.open(root + '/' + name, 'rb') as stream:
        assert stream.read() == (folder / name).read_bytes()
_, out, err = client.exec_command('/root/miniconda3/envs/bdetr/bin/python -c ' + shlex.quote("import json,shutil;print(json.dumps({'free_bytes':shutil.disk_usage('/root/autodl-tmp').free}))"), timeout=30)
disk = json.loads(out.read())
assert out.channel.recv_exit_status() == 0, err.read().decode()
values = checked['correct_mesh_metrics']
previous = checked['old_root_metrics']
addition = '\n\n### 20.98 mesh训练初始6887条核对通过、两臂64步与终态接续队列（' + when + '）\n\n'
addition += ('§20.97 的原训练PID42648继续运行。23:02:05的实际观察确认两臂各完成64/2482次更新；'
    'control/local原生GT loss分别10.266443/10.223102，梯度范数为有限值。此处是早期训练状态，'
    '不是训练终点、留出增益或正式9508成绩。累计172.08秒，平均每对更新2.68881秒。\n\n'
    '**观察器错误已隔离。** 原本地观察器在22:58处理`EVAL COMPLETE`时，较短的`EVAL`前缀也匹配该行，'
    '引发JSONDecodeError并退出1。22:59:39重新核对同一训练PID仍存活、训练controller没有退出；'
    '只将观察器匹配改为`label + "{"`，复用先前已修复的读取方式。实际捕获的完成/更新日志解析通过，'
    'v2观察器于23:02读到64步后正常退出。训练没有被重启或修改。\n\n'
    '**完整初始评估的独立核对：** 两臂6887条所有已保存字段完全一致；与旧数据运行相比，'
    'row_id、scan_id、physical_space、point SHA逐条一致。106个holdout物理场景与456个fit物理场景无交集，'
    '但冻结预训练主干以前见过这些训练场景，因此不能把本表当新场景泛化成绩。\n\n'
    '| 相同E71＋V99起点、模块holdout | REC hits@0.25 / @0.50 | Mask hits@0.25 / @0.50 | Mask mIoU |\n'
    '|---|---:|---:|---:|\n'
    '| 旧mixed superpoint数据 | %d / %d | %d / %d | %.8f%% |\n' % (previous['rec_hits025'], previous['rec_hits050'], previous['mask_hits025'], previous['mask_hits050'], previous['mask_miou'])
    + '| 正确mesh superpoint数据 | %d / %d | %d / %d | %.8f%% |\n\n' % (values['rec_hits025'], values['rec_hits050'], values['mask_hits025'], values['mask_hits050'], values['mask_miou']))
addition += ('数据修正的初始REC修复/破坏为24/28与179/76，净−4/+103；Mask为45/49与368/100，'
    '净−4/+268。它说明输入版本确实影响输出，尤其严格阈值和Mask；尚无新训练终点，不能计作局部模块收益。'
    '训练仍执行已固定的一遍更新预算，不据此重新选择超参数或正式验证阈值。\n\n'
    '初始核对回执：`refine-logs/scanrefer_local_visual_mesh_pair_20260906_v1/baseline_verification.json`，SHA`'
    + hashlib.sha256((pair / 'baseline_verification.json').read_bytes()).hexdigest() + '`；完整baseline_rows SHA`'
    + checked['source_sha256']['baseline_rows.json'] + '`。\n\n')
addition += ('**实测时间与已启动接续任务：** 初始6887评估耗时1833.90秒；据64步的实测速度，'
    '训练及终态6887评估预计于`' + checked['estimated_terminal_completion_cst'][0] + '`至`'
    + checked['estimated_terminal_completion_cst'][1] + '`结束。此估计假设训练速度和终态评估耗时保持接近，'
    '不是完成承诺。\n\n'
    '- 服务端接续任务：screen `' + queue['screen_session'][0] + '`，实际等待worker PID`' + queue['worker'][0] + '`。\n'
    '- 首次检查：`' + queue['first_check_cst'] + '`，之后240秒；等待原训练screen42642真正退出，随后要求controller退出0。\n'
    '- 之后执行已准备的独立CPU终点审计，检查两臂实际2482步、冻结状态、数据/源码SHA及逐行结果；通过后自动启动唯一local终点的9508条正式配对复核。\n'
    '- 这条接续链不以开发集分数选择epoch，也不启动Nr/Sr。新的正式目录将是`/root/autodl-tmp/mcln_scanrefer_local_visual_mesh_official_20260906_v1`；本节时尚未创建正式结果。\n'
    '- 本地观察器PID`' + str(schedule['local_observer_pid']) + '`，首次收集`' + schedule['first_check_cst']
    + '`，以后240秒，遇到实际formal启动或接续进程真实终止后退出。\n\n')
addition += ('实际接续启动回执：`refine-logs/scanrefer_local_visual_mesh_posttraining_20260906_v1/queue_launch.json`，SHA`'
    + hashlib.sha256((post / 'queue_launch.json').read_bytes()).hexdigest() + '`。原环境已编译10个封存入口，'
    '生成的Bash控制器语法和精确退出码输出检查通过。正式评估的6个源码文件与已完成v3逐字节一致，'
    '复用其实际30项测试证据；训练审计入口复用§20.97实际12项测试。准备、编译和排队都不是实际终态审计或正式评估完成。\n\n'
    '接续任务已排队，后续不要再单独启动第二份终态审计/正式评估。若观察会话超时，先检查同一任务句柄和回执，'
    '不能据此重启训练。新formal完成后仍需读取实际独立正式审计和Scan验收结果；Nr/Sr条件入口须绑定这次新的通过回执，'
    '目前绑定旧v3负结果的入口继续不执行。历史最好、Scan REC/Mask门槛及Nr/Sr Mask豁免保持§20.79。\n\n'
    '本次文档同步时实查磁盘剩余%d bytes（约%.3fGiB）。仍只计划两臂各一个终点；本节没有新增权重清理。'
    '整体三数据集目标未达成，goal保持active。\n' % (disk['free_bytes'], disk['free_bytes'] / 1024**3))
new = old + addition.encode('utf-8')
tracker = repo / 'refine-logs/EXPERIMENT_TRACKER.md'
lines = tracker.read_text(encoding='utf-8').splitlines()
lines[2] = 'Updated: ' + when + '. Acceptance: §20.79; Scan mesh actual64 updates and post-training queue: §20.98.'
for index, line in enumerate(lines):
    if line.startswith('| ScanRefer candidate local visual |'):
        lines[index] = '| ScanRefer candidate local visual | Correct-mesh6887 baseline independently verified;originalPID42648 at64/2482 per arm23:02;new initial REC6684/6426 | Terminal estimate09-07 01:18-01:21;fixed budget continues;see20.98 |'
lines.insert(31, '| ScanRefer mesh post-training queue | Live waiting worker43358;CPU audit then fixed9508 formal launch after actual successful training exit | Firstcheck09-07 01:13:08 then240s;localobserver50584 at01:14:08;no Nr/Sr activation;see20.98 |')
tracker_raw = ('\n'.join(lines) + '\n\n' + when + ': Master20.98 records actual64-step Scan progress,independent6887 paired baseline PASS and verified live endpoint-audit/formal-launch queue. No new formal metric;goal active.\n').encode()
runtime = '/home/gb/new butd/butd_detr-main/MCLN-main/'
with sftp.open(runtime + 'docs/' + master.name, 'rb') as stream:
    stream.prefetch(file_size=len(old))
    assert stream.read() == old
for name, raw in [('docs/' + master.name, new), ('refine-logs/EXPERIMENT_TRACKER.md', tracker_raw)]:
    with sftp.open(runtime + name, 'wb') as stream:
        stream.set_pipelined(True)
        stream.write(raw)
    with sftp.open(runtime + name, 'rb') as stream:
        stream.prefetch(file_size=len(raw))
        assert stream.read() == raw
for name in ['observe_first_updates_v2.py', 'first_update_observation_schedule_v2.json', 'progress_20260906_230205.json']:
    with sftp.open(remote_pair + '/' + name, 'wx') as stream:
        stream.write((pair / name).read_bytes())
observer_raw = Path('C:/Users/gb/.codex/tmp/observe_mcln_mesh_posttraining_20260906.py').read_bytes()
assert hashlib.sha256(observer_raw).hexdigest() == schedule['script_sha256']
for name, raw in [('observe_posttraining.py', observer_raw), ('observation_schedule.json', (post / 'observation_schedule.json').read_bytes())]:
    with sftp.open(remote_post + '/' + name, 'wx') as stream:
        stream.write(raw)
    (post / name).write_bytes(raw)
sftp.close()
client.close()
master.write_bytes(new)
desktop.write_bytes(new)
tracker.write_bytes(tracker_raw)
assert master.read_bytes() == desktop.read_bytes() == new
(repo / ('refine-logs/EXPERIMENT_TRACKER_' + now.strftime('%Y%m%d_%H%M%S') + '.md')).write_bytes(tracker_raw)
proof = {'section': '20.98', 'time_cst': now.isoformat(), 'bytes': len(new),
    'sha256': hashlib.sha256(new).hexdigest(), 'three_master_copies_equal': True,
    'actual_observed_steps_per_arm': 64, 'post_training_queue_live_verified': True,
    'formal_rows_this_stage': 0, 'disk_free_bytes': disk['free_bytes']}
(pair / 'handoff_sync_20_98.json').write_bytes((json.dumps(proof, indent=2) + '\n').encode())
(pair / 'publish_first_updates.py').write_bytes(Path(__file__).read_bytes())
with (repo / 'MANIFEST.md').open('ab') as stream:
    stream.write(('| ' + now.strftime('%Y-%m-%d %H:%M') + ' | /run-experiment | refine-logs/scanrefer_local_visual_mesh_pair_20260906_v1/baseline_verification.json | run | §20.98 actual64 updates;paired6887 baseline PASS;post-training queue live |\n').encode())
print(json.dumps(proof))
