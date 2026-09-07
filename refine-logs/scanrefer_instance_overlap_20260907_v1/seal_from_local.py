import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
trial = repo / 'refine-logs/scanrefer_native_box_transfer_pair_20260907_v1'
analysis = repo / 'refine-logs/scanrefer_instance_overlap_20260907_v1'
queue = repo / 'refine-logs/scanrefer_native_box_transfer_posttraining_20260907_v1'
receipt = json.loads((trial / 'receipt.json').read_bytes())
audit = json.loads((analysis / 'independent_recount.json').read_bytes())
decision = json.loads((queue / 'decision.json').read_bytes())
assert receipt['steps_per_arm'] == 2482 and receipt['status'] == 'complete'
assert not receipt['eligible_for_fixed_terminal_formal_evaluation']
assert audit['status'] == 'pass' and audit['geometry_stage_records_recomputed'] == 152128
assert decision['formal_evaluation_count'] == 0 and not decision['nr3d_sr3d_training_started']
master = repo / 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop = Path('C:/Users/gb/Desktop/document') / master.name
old = master.read_bytes()
assert hashlib.sha256(old).hexdigest() == '73c25a2fe6d174166f658e7c84f34045466453086d4e31fc4f57fa2f38310b00'
assert desktop.read_bytes() == old
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()

report = '''# ScanRefer 原生教师框转移终态与历史几何归因

## 1. 当前固定教师框转移：完整结束，未晋级

两臂各2482次更新、29778条fit表达；6887条模块留出来自预训练主干见过的训练场景，不能称为完整系统新场景泛化。实际训练16:38:50结束，独立审计16:39:03通过，条件队列16:40:04正常退出。只改16项center/size参数，完整1144项重建核对通过；其余核心、BN缓冲、语义输出、原始Mask、教师和读出不变。

| 6887条模块留出 | 完整系统REC@0.25/@0.50 | 原生REC@0.25/@0.50 | Mask@0.25/@0.50 | mIoU |
|---|---:|---:|---:|---:|
| 共同起点 | 6684/6426 | 6572/5955 | 6511/6097 | 77.810860787% |
| gt_only | 6679/6440 | 6557/5979 | 6511/6097 | 77.810860787% |
| gt_teacher_box | 6678/6441 | 6556/5976 | 6511/6097 | 77.810860787% |

1. 候选相对起点系统−6/+15，原生−16/+21；相对GT-only系统−1/+1、原生−1/−3。没有证据表明这个教师框辅助优于直接GT回归；严格阈值局部提升也没有保护宽松阈值。
2. 系统相对起点修复/破坏为1/7、27/12；相对GT-only为1/2、4/3。没有将训练loss、教师连续IoU收益或不变Mask作为晋级证据。
3. 固定候选封存，不改辅助权重、LR或epoch，不用GT-only替代候选。正式评估0、Nr/Sr训练0。已有原生初始化导出准备不赋予该失败终点使用资格。

两份终点各4045205 bytes，保留在原远端实验目录；无新增完整模型副本。17.81MB fit_point_batches.json保留远端和本地，Git只存精确SHA/大小/位置指针。

## 2. 历史正确mesh局部视觉：下降主要包含几何修复机会流失

以下复用已完成的9508-row逐级诊断，与上面教师框实验不同。CPU从141个场景的实际对象框和旧预测重算152128项几何记录，6项测试及本地独立重算通过；新增GPU/优化/正式forward均为0。

最大GT框重叠仅是几何对应代理。使用全部已标注对象（包含嵌套/结构类），最大值并列1e-6单列；不把Query编号、IoU>0.25或最大GT重叠直接当作真实语义身份。

| V99 final @0.50失败几何代理 | 保护 | 局部 |
|---|---:|---:|
| root为唯一最大重叠对象 | 833 | 900 |
| 另一同原始标签对象为唯一最大 | 2685 | 2674 |
| 另一不同标签对象为唯一最大 | 1194 | 1207 |
| 最大重叠并列 | 0 | 0 |
| 无重叠 | 1 | 4 |
| 失败总数 | 4713 | 4785 |
| 命中总数 | 4795 | 4723 |

1. Final配对修复75、破坏147，净−72；两臂均root唯一最大重叠的子集中修复54、破坏114，净−60。它支持几何范围质量是重要组成，不能证明实例语义身份保持不变。833→900的总体分类变化与配对−60不是同一统计口径。
2. 对各臂自己选定的Geometry Query，比较其原框与所选变体：

| 同一Query原框→所选变体，@0.50 | 原框命中 | 变体命中 | 修复 | 破坏 | 净收益 |
|---|---:|---:|---:|---:|---:|
| 保护 | 4484 | 4816 | 419 | 87 | +332 |
| 局部 | 4485 | 4727 | 303 | 61 | +242 |

变体净收益少90 = 修复少116 − 破坏少26。因此不能写成“新变体制造了更多破坏”；这里主要是原本可获得的修复变少。两臂选择的Query集合不同，尚不能区分完整变体覆盖下降与冻结读出选择失配。
3. 旧stage_rows只保留实际选中变体和原生Top16框，缺少每个Query的完整七变体集合，无法从该缓存算出全变体oracle。当前诊断不支持调整验证集分位数/阈值，也不产生新方法精度。

## 3. 下一步问题与执行边界

停止继续局部64点/固定范围/教师框辅助的同机制扫描。下一项应明确训练实际部署答案的身份与几何，而不是只改GT Hungarian匹配的回归头：先在已有fit数据与源码中核对最终Query—Variant决策的训练作用范围，再选一个能让实际几何输出与分数共同学习的机制。此前joint/frozen_gt已完成，单改detach、加自由IoU头或复制教师排名不算新方案。

保持同一预训练起点、mesh输入、ScanRefer优先；正式Scan不低于保护V99及Scan Mask底线后立即接Nr/Sr REC，不等待59/51。Nr/Sr Mask仍不设门。最终结果仍须三数据集验证；当前正式Pareto集未刷新。

## 4. 证据

- 原生框转移：`refine-logs/scanrefer_native_box_transfer_pair_20260907_v1/`（terminal_rows、receipt、independent_audit）。
- 条件接续决定：`refine-logs/scanrefer_native_box_transfer_posttraining_20260907_v1/decision.json`。
- 历史几何归因：`refine-logs/scanrefer_instance_overlap_20260907_v1/`（带几何数据的overlap_rows.json.gz、summary、独立重算和来源）。
- 诊断源码：`scripts/analyze_scanrefer_instance_overlap.py`；6项有意义的几何与映射测试：`tests/test_instance_overlap_diagnostic.py`。
'''
report_path = 'docs/SCANREFER_BOX_TRANSFER_AND_INSTANCE_OVERLAP_RESULT_2026-09-07.md'
addition = '\n\n### 20.128 原生教师框转移终态封存（' + now + '）\n\n'
addition += '两臂实际各2482更新，16:38:50终态、16:39:03独立审计、16:40:04队列结束，均exit0。6887条主干曾见模块留出：起点系统6684/6426、原生6572/5955；GT-only系统6679/6440、原生6557/5979；教师框候选系统6678/6441、原生6556/5976。Mask仍6511/6097、mIoU77.810860787%。候选系统相对起点−6/+15、对GT-only−1/+1；原生分别−16/+21、−1/−3。固定REC筛选失败，正式0、Nr/Sr训练0，不替换候选、不扫描辅助权重/LR/epoch。只改16项回归参数，1144项完整重建及冻结输出审计通过；两份终点各4045205 bytes。\n\n'
addition += '训练receipt SHA `a3b91dd91fe7a72106c468c9b928c2f8789661799ab40ad765156ad85996b075`；终态rows SHA `0e318fa605e795f19b960eb203c7d5634b0127d528e1d620d526fa012a9810aa`；独立审计SHA `cc62d9fe2ac64c349880f97a0538828b0d2800ba3fe176ad99efa612ad05a68a`。本地再次从逐行结果重算系统/原生/Mask一致。大批次输入清单原件保留远端与本地，Git只存SHA指针。已准备的小框头原生导出仅限未来正式晋级终点，当前失败候选不导出。\n\n'
addition += '### 20.129 历史局部视觉几何归因：变体收益主要损失在修复减少（' + now + '）\n\n'
addition += 'CPU复用原9508-row逐级记录及141个真实场景的GT对象几何，重算2臂×8阶段共152128项。本地独立算术与映射核验、6项测试通过，新增GPU/优化/正式forward均0。最大GT重叠只是几何代理，嵌套对象与结构类也参与，不能据此证明真实语义身份。\n\n'
addition += 'V99 final@0.50保护/局部命中4795/4723，配对修复75、破坏147；两臂均root唯一最大重叠子集54/114、净−60。各臂选定Geometry Query的原框→变体：保护4484→4816，修复419/破坏87、净+332；局部4485→4727，修复303/破坏61、净+242。故变体收益少90来自修复少116且破坏也少26，不能写成“变体增加破坏”。两臂所选Query不同，旧缓存没有全七变体，仍不能区分变体oracle覆盖下降与冻结读出失配。\n\n'
addition += 'summary SHA `d7db6f5f0a2da414dbe65769b8e3b240182534b401aec2430f1ec18ddcead1df`，独立重算记录16:49:13 PASS。完整表和边界见`' + report_path + '`。下一步从实际Query—Variant输出与训练目标的对应入手，不复跑已失败固定范围读取或同教师辅助扫描。ScanRefer先过保护线即接Nr/Sr REC的总目标不变，当前没有新正式精度或新训练启动。\n'
new = old + addition.encode('utf-8')
tracker_path = repo / 'refine-logs/EXPERIMENT_TRACKER.md'
lines = tracker_path.read_text(encoding='utf-8').splitlines()
lines[2] = 'Updated: ' + now + '. Sections20.128-20.129: teacher-box pair COMPLETED/REC FAIL, queue ended with formal0; historical geometry attribution independently recounted; no current training.'
replacements = {
    '| Native box transfer automatic continuation |': '| Native box transfer automatic continuation | Completed16:40:04, controller0 | Fixed candidate module REC FAIL; formal0, Nr/Sr0; no control substitution |',
    '| Native teacher-box transfer |': '| Native teacher-box transfer | Complete2482/arm,6887rows; integrityPASS, RECFAIL | System candidate6678/6441 vs6684/6426 and6679/6440; native6556/5976 vs6572/5955 and6557/5979; fixed version sealed |',
    '| ScanRefer corrected-mesh final and stage diagnostic |': '| ScanRefer corrected-mesh final and stage diagnostic | Formal REC FAIL; archived diagnostic independently extended | Same-query Geometry benefit332→242:116 fewer repairs and26 fewer breaks; overlap proxy is not semantic identity; fixed range successor also already failed |',
    '| ScanRefer candidate local visual |': '| ScanRefer candidate local visual | Correct-mesh pair and full formal completed; endpoints retired with evidence retained | REC FAIL5543/4722 vs5570/4797; see20.102-20.103 and20.127; no process remains pending |',
    '| ScanRefer mesh post-training queue |': '| ScanRefer mesh post-training queue | Historical queue completed | Correct-mesh formal REC FAIL; no Nr/Sr launch; old live observations superseded by20.102-20.103 |',
    '| ScanRefer mesh formal acceptance queue |': '| ScanRefer mesh formal acceptance queue | Historical acceptance completed | Scan failed; native GPU preflight not activated; no pending wait handle |',
}
for i, line in enumerate(lines):
    for prefix, replacement in replacements.items():
        if line.startswith(prefix):
            lines[i] = replacement
lines.insert(6, '| Historical instance-overlap attribution | CPU complete;141 scenes,152128 stage records;6tests and independent recount PASS | Final50 net−72; root-max proxy subgroup−60; lost variant repairs dominate, no new method/formal result |')
tracker_raw = ('\n'.join(lines) + '\n').encode()

client = paramiko.SSHClient(); client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
runtime = '/home/gb/new butd/butd_detr-main/MCLN-main/'
with sftp.open(runtime + 'docs/' + master.name, 'rb') as stream:
    stream.prefetch(file_size=len(old)); assert stream.read() == old
for local in [trial/'receipt.json', trial/'independent_audit.json', queue/'decision.json', analysis/'summary.json']:
    remote = '/root/autodl-tmp/mcln_' + local.parent.name + '/' + local.name
    with sftp.open(remote, 'rb') as stream:
        stream.prefetch(file_size=local.stat().st_size); assert stream.read() == local.read_bytes()
code = """import json,os,shutil,socket,subprocess
print(json.dumps({'hostname':socket.gethostname(),'uid':os.getuid(),'processes':{str(p):os.path.exists('/proc/'+str(p)) for p in [58023,58296,60139]},'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits']).decode().strip(),'free_bytes':shutil.disk_usage('/root/autodl-tmp').free}))
"""
_, out, err = client.exec_command('/root/miniconda3/envs/bdetr/bin/python -c ' + shlex.quote(code), timeout=60)
observed = json.loads(out.read().decode()); assert out.channel.recv_exit_status() == 0, err.read().decode()
assert not any(observed['processes'].values()), observed

payloads = {report_path: report.encode(), 'docs/'+master.name: new, 'refine-logs/EXPERIMENT_TRACKER.md': tracker_raw,
            'scripts/analyze_scanrefer_instance_overlap.py': (repo/'scripts/analyze_scanrefer_instance_overlap.py').read_bytes(),
            'tests/test_instance_overlap_diagnostic.py': (repo/'tests/test_instance_overlap_diagnostic.py').read_bytes()}
for name, raw in payloads.items():
    with sftp.open(runtime + name, 'wb') as stream:
        stream.set_pipelined(True); stream.write(raw)
    with sftp.open(runtime + name, 'rb') as stream:
        stream.prefetch(file_size=len(raw)); assert stream.read() == raw
    (repo/name).write_bytes(raw)
desktop.write_bytes(new)

pointer = {'file':'fit_point_batches.json', 'bytes':(trial/'fit_point_batches.json').stat().st_size,
           'sha256':hashlib.sha256((trial/'fit_point_batches.json').read_bytes()).hexdigest(),
           'local_path':str(trial/'fit_point_batches.json'),
           'remote_path':'/root/autodl-tmp/mcln_scanrefer_native_box_transfer_pair_20260907_v1/fit_point_batches.json',
           'reason':'Raw17.81MB input-batch manifest preserved outside source Git; no evidence deleted.'}
assert pointer['sha256'] == receipt['fit_batches_sha256']
extra = [(trial, 'fit_point_batches_archive.json', (json.dumps(pointer, indent=2)+'\n').encode()),
         (analysis, 'independent_recount_from_local.py', Path('C:/Users/gb/.codex/tmp/audit_mcln_overlap_and_box_terminal_20260907.py').read_bytes()),
         (analysis, 'independent_recount.json', (analysis/'independent_recount.json').read_bytes())]
proof = {'time_cst':now,'master_sha256':hashlib.sha256(new).hexdigest(),'master_bytes':len(new),
         'three_master_copies_equal':True,'observation':observed,'new_formal_rows':0,'goal_complete':False}
extra += [(analysis, 'handoff_sync.json', (json.dumps(proof,indent=2)+'\n').encode()),
          (analysis, 'seal_from_local.py', Path(__file__).read_bytes())]
for directory, name, raw in extra:
    (directory/name).write_bytes(raw)
    remote = '/root/autodl-tmp/mcln_' + directory.name + '/' + name
    with sftp.open(remote, 'wb') as stream:
        stream.set_pipelined(True); stream.write(raw)
    with sftp.open(remote, 'rb') as stream:
        stream.prefetch(file_size=len(raw)); assert stream.read() == raw
sftp.close(); client.close()
with (repo/'.gitignore').open('a',encoding='utf-8',newline='\n') as stream:
    stream.write('/refine-logs/scanrefer_native_box_transfer_pair_20260907_v1/fit_point_batches.json\n')
with (repo/'.gitattributes').open('a',encoding='utf-8',newline='\n') as stream:
    stream.write('refine-logs/scanrefer_instance_overlap_20260907_v1/** -text\n')
print(json.dumps(proof),flush=True)
