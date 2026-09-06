import datetime
import hashlib
import json
import os
from pathlib import Path

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
trace = repo / 'refine-logs/scanrefer_stage_diagnostic_20260907_v1'
result = trace / 'diagnostic_result'
master = repo / 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop = Path('C:/Users/gb/Desktop/document') / master.name
old = master.read_bytes()
assert hashlib.sha256(old).hexdigest() == '8e68cbc32b717c11b5fad49457383cffd2f06b5c42625dfd518b167f762e8b75'
assert desktop.read_bytes() == old
audit = json.loads((result / 'independent_audit.json').read_bytes())
analysis = json.loads((result / 'evidence_breakdown.json').read_bytes())
receipt = json.loads((result / 'receipt.json').read_bytes())
assert audit['integrity_pass'] and audit['formal_rows'] == 0
assert analysis['audit_sha256'] == hashlib.sha256((result / 'independent_audit.json').read_bytes()).hexdigest()
assert (trace / 'controller.exit').read_text().strip() == '0'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
runtime = '/home/gb/new butd/butd_detr-main/MCLN-main/'
remote = '/root/autodl-tmp/mcln_scanrefer_stage_diagnostic_20260907_v1'
with sftp.open(runtime + 'docs/' + master.name, 'rb') as stream:
    stream.prefetch(file_size=len(old))
    assert stream.read() == old
source = '/root/autodl-tmp/mcln_scanrefer_local_visual_preflight_20260906_v2/model_source/'
for name, expected in analysis['feature_schema_source_sha256'].items():
    with sftp.open(source + name, 'rb') as stream:
        assert hashlib.sha256(stream.read()).hexdigest() == expected
pair = json.loads((repo / 'refine-logs/scanrefer_local_visual_mesh_pair_20260906_v1/input_manifest.json').read_bytes())
protected = {name: item['path'] for name, item in pair['artifacts'].items()}
base = '/root/autodl-tmp/DATA_ROOT/output/network_v99_baseline_gt/nr3d/'
protected['nr_protected'] = base + 'control/official_rec_monitor/official_best_rec025_epoch_57_0p56652741.pth'
protected['nr_resume_e57'] = base + 'audit/nr3d_mcln_joint_butdcls_v99_relation_cf_conservative_anchor_density_v2_audit_e58_b100_b16x1_w4p2_one_shot/resume_e57.pth'
expected_sizes = json.loads((repo / 'refine-logs/scanrefer_local_visual_mesh_pair_20260906_v1/terminal_archive_verification.json').read_bytes())['protected_file_sizes']
sizes = {name: sftp.stat(path).st_size for name, path in protected.items()}
assert sizes == expected_sizes
command = "import json,shutil,subprocess; d=shutil.disk_usage('/root/autodl-tmp'); p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],stdout=subprocess.PIPE,check=True); print(json.dumps({'disk':d._asdict(),'gpu_compute_processes':p.stdout.decode().strip()}))"
_, stdout, stderr = client.exec_command('/root/miniconda3/envs/bdetr/bin/python - <<\'PY\'\n' + command + '\nPY', timeout=30)
state_raw = stdout.read()
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
state = json.loads(state_raw)
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
state.update({'time_cst': now.isoformat(), 'protected_file_sizes': sizes, 'new_weights_deleted': 0,
              'previous_cleanup_reclaimed_gib': 9.529224, 'all_current_gpu_work_completed': not state['gpu_compute_processes']})
(trace / 'terminal_resource_state.json').write_text(json.dumps(state, indent=2, sort_keys=True) + '\n', encoding='utf-8')
when = now.strftime('%Y-%m-%d %H:%M CST')
text = '\n\n### 20.103 逐级诊断完成：Parent宽松阈值退化与Geometry严格阈值收益减少（' + when + '）\n\n'
text += ('**实际诊断已于02:20:37结束，9508条、两固定权重、controller=0。** 观察器6419/PID49108于02:21:18完成收集并退出0；当前没有本项目训练/诊断GPU任务运行，不要重启旧观察器。'
         '这是既定正式评估之后的独立诊断：formal_rows=0，optimizer_updates=0，checkpoint_writes=0，不能晋级或替代§20.102正式数值。\n\n'
         '**独立CPU审计通过。** 两臂共114096个阶段框及304256个Top-16框相对root GT的IoU用NumPy独立重算，与保存值最大差均为0；'
         '所有阶段命中、修复/破坏、Query/变体映射、输入身份、152/179维有效特征计数及文件SHA核对通过。'
         '运行时还逐forward核对最终分数完全一致，模型和读出状态未变化。\n\n'
         '| 本次诊断阶段 | 保护REC hits@0.25/@0.50 | local REC hits@0.25/@0.50 | local−保护 |\n'
         '|---|---:|---:|---:|\n')
stages = ['native', 'parent', 'parent_after_geometry_validity', 'geometry', 'v99_proposal', 'v99_final']
for stage in stages:
    a, b = [audit['arms'][arm]['metrics'][stage] for arm in ['protected_v99', 'local_v99']]
    text += '| ' + stage + ' | ' + str(a['hits025']) + '/' + str(a['hits050']) + ' | ' + str(b['hits025']) + '/' + str(b['hits050']) + ' | ' + str(b['hits025']-a['hits025']) + '/' + str(b['hits050']-a['hits050']) + ' |\n'
text += ('\n本次合法Top-16 oracle保护6764/5968，local6806/6050，增加42/82；没有证据将此次完整系统下降归因为Top-16覆盖减少。'
         'Parent有效性转接没有改变任何选择。@0.25的差距在Parent扩大，@0.50在Geometry由+6转成−89。'
         '未加Pareto限制的V99 proposal只是诊断中间态，不能据正式验证集结果直接删规则或改部署。\n\n'
         '**Geometry损失进一步分解。** 在各自已经选定的同一个Query内，以原始框替代所选变体作为诊断：保护4484→4816（+332），local4485→4727（+242），'
         '因此89个严格阈值差距可算术分解为所选Query原框+1、该Query变体相对原框收益−90。原始Parent到Geometry所选Query原框的收益为+64/+59。'
         '最终V99同样是保护4388→4795（+407），local4435→4723（+288）。这说明原框的改进没有补回几何变体收益；'
         '尚不能进一步区分候选变体质量、变体选择和跨臂样本组成的因果贡献，也不能把Query槽变化说成实例变化。\n\n'
         '**真实归一化特征变化。** 两臂Parent有效输入各152128个，Geometry为1044873/1040832。Parent152维均值差的RMS为0.28590；'
         '最大均值变化位于query_proj_42（+1.08186个既有归一化单位）。Geometry的source_mask_to_regressed_volume_ratio均值差+0.70632、RMS差+19.30311。'
         '这些是各臂实际有效候选总体的描述统计，不是相同候选上的因果对照，不能直接据此调归一化、裁剪或阈值。\n\n'
         '**重跑并非逐行完全一致。** 相同点SHA和固定权重下，保护native有66个Query索引变化、最终89个变体位置变化；local为63/107。'
         '保护最终REC相对既定正式结果0/−2，local为0/+1；本次原生为5514/4409和5511/4428。'
         '具体漂移原因尚未定位，不能自动归因于浮点误差或将原正式回执改写。本次诊断最终−27/−72，§20.102正式仍是−27/−75；两者必须分开报告。\n\n'
         '**证据归档。** 原始stage_rows.json为104832441 bytes，GitHub存储无损gzip 22566787 bytes，解压SHA与原件完全一致，'
         '原件留在远程和本地工作目录；完整收集与压缩回执位于`refine-logs/scanrefer_stage_diagnostic_20260907_v1/`。'
         '复查时先将stage_rows.json.gz解压为同目录stage_rows.json，再用Python+NumPy运行`scripts/audit_scanrefer_stage_diagnostic.py --directory <诊断目录> --reference <§20.102正式目录>`；'
         '输出采用排他创建，复核应在独立副本中保留既有审计文件。模型推理没有引入新依赖。\n\n'
         '本次诊断回执SHA`' + audit['receipt_sha256'] + '`，独立审计SHA`' + analysis['audit_sha256'] + '`。'
         '新脚本只是只读诊断，不是新模型贡献。接下来从同一E71预训练起点准备固定64点的中心读取/范围读取最小对照；'
         '先在训练数据上核对区域覆盖、空间布局和梯度，再锁定ScanRefer训练预算。不得根据本次9508逐场景结果调规则，暂不同时加入质量分布、兼容损失或新Gate。'
         '范围读取需要证明本身有效；旧读出与几何变体接口仍须在终态单列检查，不能用原生提升替代完整系统门槛。此新对照尚未启动。\n\n')
text += ('磁盘本次实查剩余' + str(state['disk']['free']) + ' bytes（' + format(state['disk']['free']/1024**3, '.3f') + 'GiB），6个受保护文件大小一致。'
         '已失败旧权重的累计清理仍约9.529GiB；本轮两臂终点暂保留作配对复核，只保留必要终点，不增加逐步权重。'
         '历史ScanRefer V99保持5572/4797，Nr/Sr未启动；三数据集目标尚未完成，goal保持active。\n')
new = old + text.encode()
tracker = repo / 'refine-logs/EXPERIMENT_TRACKER.md'
lines = tracker.read_text(encoding='utf-8').splitlines()
lines[2] = 'Updated: ' + when + '. Corrected-mesh formal FAIL;full stage diagnostic and independent audit complete: §20.103. No GPU job active;next fixed64-point extent preflight not launched.'
lines[6] = '| ScanRefer corrected-mesh final and stage diagnostic | Formal5543/4722 vs5570/4797 FAIL;diagnostic9508 complete,auditPASS | Parent widens@0.25 loss;Geometry turns@0.50 +6 to−89;Top16 oracle+42/+82;next fixed64-point extent control;Nr/Sr not launched |'
tracker_raw = ('\n'.join(lines) + '\n\n' + when + ': Stage terminal audited;formal remains sealed;extent preflight next,not yet started.\n').encode()
uploads = {
    'diagnostic_result/independent_audit.json': result / 'independent_audit.json',
    'diagnostic_result/evidence_breakdown.json': result / 'evidence_breakdown.json',
    'diagnostic_result/stage_rows_archive.json': result / 'stage_rows_archive.json',
    'scripts/audit_scanrefer_stage_diagnostic.py': repo / 'scripts/audit_scanrefer_stage_diagnostic.py',
    'scripts/analyze_scanrefer_stage_evidence.py': repo / 'scripts/analyze_scanrefer_stage_evidence.py',
    'terminal_resource_state.json': trace / 'terminal_resource_state.json',
    'progress_20260907_022059.json': trace / 'progress_20260907_022059.json',
}
for name, path in uploads.items():
    raw = path.read_bytes()
    with sftp.open(remote + '/' + name, 'wx') as stream:
        stream.write(raw)
    with sftp.open(remote + '/' + name, 'rb') as stream:
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
proof = {'time_cst': now.isoformat(), 'section': '20.103', 'sha256': hashlib.sha256(new).hexdigest(), 'bytes': len(new),
         'three_master_copies_equal': True, 'stage_audit_sha256': analysis['audit_sha256'],
         'diagnostic_rows': 9508, 'formal_rows': 0, 'scan_promoted': False, 'goal_complete': False,
         'observer_completed': True, 'new_training_started': False}
for name, raw in [('handoff_sync_20_103.json', (json.dumps(proof, indent=2, sort_keys=True)+'\n').encode()),
                  ('publish_terminal.py', Path(__file__).read_bytes())]:
    (trace / name).write_bytes(raw)
    with sftp.open(remote + '/' + name, 'wx') as stream:
        stream.write(raw)
sftp.close()
client.close()
with (repo / 'MANIFEST.md').open('ab') as stream:
    stream.write(('| ' + when + ' | direct continuation | refine-logs/' + trace.name + '/diagnostic_result/independent_audit.json | diagnostic | Actual9508 stages audited,formalFAIL preserved;fixed64-point extent preflight next |\n').encode())
print(json.dumps(proof))
print(json.dumps(state))
