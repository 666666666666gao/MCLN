"""Prepare static progress and archive-availability notes; no remote calls."""
import ast
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parent
study = root / 'deployment_geometry_training_20261012'
data = root / 'pv_initial_and_dedup_publication_20261014'
assert not data.exists()
data.mkdir()
e0 = json.loads((study / 'LOWER_SPAN_LR_E0_COMPLETE_RECEIPT_20261014.json').read_bytes())
retired = json.loads((study / 'LOCAL_EQUAL_E0_WEIGHT_DEDUP_RETIREMENT_20261014.json').read_bytes())
assert e0['current_arm_E0_complete_metrics']['hits025'] == 5677
assert e0['current_arm_E0_complete_metrics']['hits050'] == 4920
assert e0['actual_process_witness_cst'] == '2026-10-14T03:20:49.600024+08:00'
assert retired['released_bytes'] == 1229851934 and retired['unique_best_model_preserved']
captured = study / 'OBS_lower_span_lr_e0_completion_files'
diag, = captured.rglob('native_diagnostic_epoch000.json')
metrics, = captured.rglob('native_metrics.jsonl')
sources = {
    'LOWER_SPAN_LR_E0_COMPLETE_RECEIPT.json': study / 'LOWER_SPAN_LR_E0_COMPLETE_RECEIPT_20261014.json',
    'ACTUAL_E0_OBSERVATION_INTAKE.json': study / 'OBS_lower_span_lr_e0_completion_INTAKE.json',
    'NATIVE_DIAGNOSTIC_EPOCH000.json': diag,
    'NATIVE_METRICS.jsonl': metrics,
    'NORMAL_LOWER_B_LR_PROTOCOL.json': captured / 'protocol.json',
    'NEXT_E1_OBSERVER_OWNER.json': study / 'OBS_lower_span_lr_e1_near_complete_OWNER.json',
    'LOCAL_WEIGHT_DUPLICATE_VERIFICATION.json': study / 'LOCAL_EQUAL_E0_WEIGHT_DEDUP_VERIFICATION_20261014.json',
    'LOCAL_WEIGHT_DUPLICATE_RETIREMENT.json': study / 'LOCAL_EQUAL_E0_WEIGHT_DEDUP_RETIREMENT_20261014.json',
    'verify_equal_local_E0_weights.py': study / 'verify_equal_local_E0_weights_20261014.py',
    'record_complete_initial.py': study / 'record_span_lr_complete_initial_20261014.py',
    'record_local_duplicate_retirement.py': study / 'record_equal_E0_local_retirement_20261014.py',
    'SAVED_C_ENDPOINT_TRANSITIONS.json': study / 'selected_endpoint_changes_20261014' / 'RESULT.json',
    'SAVED_C_ENDPOINT_REPORT.md': study / 'selected_endpoint_changes_20261014' / 'REPORT.md',
    'analyze_saved_C_endpoints.py': study / 'analyze_selected_endpoint_changes_20261014.py',
}
hashes = {}
for name, source in sources.items():
    raw = source.read_bytes()
    (data / name).write_bytes(raw)
    hashes[name] = {'source': str(source), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
summary = dict(status='LOWER_B_LR_COMPLETE_E0_VERIFIED_REAL_FIRST_EPOCH_STARTED',
    observed_cst=e0['actual_process_witness_cst'],
    retained_best_hits=[5677, 4920], goal_hits=[5705, 4945],
    E0=e0['current_arm_E0_complete_metrics'], first_train_batch_observed=35,
    train_batches_per_epoch=4583, current_trained_formal_metrics=None,
    current_normal_controller_pid=432868, current_normal_child_pid=432871,
    source_stage_changed_during_running_experiment=False,
    learning_rates={'core': 1e-6, 'backbone': 1e-6, 'A': 1e-5, 'B': 1e-6},
    observer_pid=26676, observer_native_session=11018, observer_functions_cell='457',
    next_observation_cst='2026-10-14T08:15:00+08:00',
    no_remote_progress_queries_before_next_due=True,
    initial_counts_match_historical_E0=True, initial_bitwise_identity_established=False,
    selected_raw_gate_outside_count=6, selected_axis_count=28524,
    raw_gate_counts_not_gradient_or_clamp_causal_proof=True,
    original_E0_receipt_GPU_fields_scope='Historical startup02:47:41; noGPU query at03:20',
    local_duplicate_weight_files_retired=2, local_released_bytes=1229851934,
    remote_released_bytes_this_annex=0,
    preserved_model=retired['protected_model'], original_non_model_payloads_preserved=True,
    retired_containers_not_currently_materialized=True,
    rehydrate_original_metadata_before_historical_checkpoint_load=True,
    original_ZIP_byte_reconstruction_not_claimed=True, cold_native_resume_verified=False,
    MCLN_formal_training_launched=True, MCLN_formal_training_complete=True,
    independent_three_module_gains_established=False,
    Nr3D_or_Sr3D_new_NN_validation_or_training_launched=False,
    metric_promoted=False, full_goal_complete=False)
(data / 'CLOSEOUT.json').write_bytes((json.dumps(summary, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
text = '''本节记录两个实际变化：低B学习率正常训练的完整E0已结束，并清理了本地相同模型状态的重复容器。没有新增已训练精度，不提前读取第1轮进度，不改活动网络、优化器、训练源码或单卡队列。

### 实际正常训练

03:20:49（北京时间）原控制器432868与原生子进程432871均存活；完整9508条`last/bbs`起点为5677／4920，即59.7076%／51.7459%。训练日志随后到第1轮35／4583个batch。该计数是日志进度，不是冷加载优化器状态审计。E0与历史强起点的命中数相同；尚未验证全输出逐位一致，也不能称为训练新增收益。

本轮只将B基础学习率从1e−5降为1e−6；核心和骨干1e−6、A为1e−5，seed2027、batch8、有效batch8、最多8轮、完整Acc@0.5连续两轮逐轮下降停止，持平或回升重置。当前没有E1完整9508条成绩。下一次唯一观察器26676／native11018／functions457定在08:15；依据同正常流程此前约3.53秒每训练batch及约34分钟正式验证，预计第1轮完整结果08:00—08:30。时间未到不查询远端NN进度。

本节E0诊断来自实际当前前向。三轴原系数低于0的全部候选轴计数为198203／211220／167258，高于1均为0；已选候选仅6／28524轴超出裁剪范围。它不证明裁剪导致训练下降，也没有测任务梯度。E0完整候选覆盖8970／8234属于日志标量；未重新读取全部256候选坐标。原E0完整回执中继承的显存及设备字段来自02:47:41启动观察，03:20没有重新查询GPU。

### 已完成C实验的保存数据分析

分析只使用旧C的28524条保存输出及此前CPU重算IoU，零NN、零更新、零远程查询。严格结果E0/E1/E2为4920／4752／4623；同轮同一已选Query的Mask端点命中为4881／4809／4725，最终框对Mask端点净变化为+39／−57／−102。E2相对E0净损失297条；按表达行GT体积分组，较小前两组净损失250条。E0原正确、E2错误的640条中，215条在E2仍有同Query的合格端点。

这些是保存输出的几何比较，不独立证明选择错物理实例、B学习率或某个梯度路径是主因。四分位仍有重复目标和场景相关性，不能作为独立样本显著性证明。本轮不会据此临时改变活动配置或推理规则。

### 当前权重可用性更新

使用已有CPU Torch实际加载三份完整权重，各1295个模型状态张量的dtype、shape、字节摘要及顺序完全一致。仅删除本地旧控制和旧C各614925967字节的`reconstructed/best.pth`完整容器，共1229851934字节；本节没有释放新的远端空间。唯一保留的共享完整模型为`C:/Users/gb/.codex/archives/pvg_native_normal_20261010/best.pth`，SHA256为`ed8455ddc67e4d17018e9cf499197140e15e35db3f5daee55e0eec6846faf0c4`。

两项实验各自的`original_non_model_state.pth`（包括原优化器／调度器等非模型状态及原模型metadata）和归档回执继续保留。旧历史回执中的完整容器路径描述的是当时状态，现在不再直接存在；如重用历史检查点，须先将共享模型状态与该实验原始非模型状态重新组合。保存内容可重建，不声称恢复原远端ZIP字节，也未验证冷启动原生训练恢复。六份可变续接状态已明确记录此可用性映射。作者、G、A/B、V99、E71、活动训练依赖、数据集和唯一日志不变。

### 完整目标仍未达到

同一完整模型在ScanRefer至少5705／4945命中（60%／52%），三个有效机制的直接证据，以及同最终完整方法的Nr3D／Sr3D作者起点评估和独立训练，仍未完成。当前调低B学习率是优化假设，不能作为新的论文创新。各数据集协议、score与对象输入限制延续§20.376.175及对应来源报告。
'''
(data / 'CLOSEOUT.md').write_bytes(text.encode('utf-8'))
handoff = Path(tempfile.gettempdir()) / 'pvground_low_B_lr_continuation_20261014.md'
handoff_text = '''# PV-Ground continuation

Continue the same complete goal: ScanRefer9508 >=5705/4945 hits, three directly effective mechanisms, then matching-author-protocol Nr3D/Sr3D baseline evaluation and independent full-model training. Single seed2027 and sole A100 serial. No V99 dual ranking.

## Live ownership

Original normal controller432868/child432871. Saved actual E0 and first-train evidence: `deployment_geometry_training_20261012/LOWER_SPAN_LR_E0_COMPLETE_RECEIPT_20261014.json`. E0 matches5677/4920; no trained new formal score yet. Do not poll remote NN before08:15 BJT. Sole original observer26676/native11018 is joined by responsive functions cell457; resume only that cell, never a duplicate observer or parallel native11018 query. If observation fails, inspect the original process after its due node; do not restart NN due to observer timeout/failure.

## Authoritative artifacts

Study directory: `C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012`. Read CURRENT_PROGRESS.json current_normal_* and current_low_span_lr_normal_record; old C/control fields are history. Source/runtime of the active model unchanged. Acc@0.5 two successive epochwise declines stops; equality/rise resets. Do not change the ongoing one-variable training.

Current local weight availability: LOCAL_EQUAL_E0_WEIGHT_DEDUP_RETIREMENT_20261014.json plus local_equal_E0_weight_current_availability in six continuation states. Two old control/C full containers are now retired. Shared complete model remains pvg_native_normal_20261010/best.pth; each original_non_model_state.pth and original receipt remain. Rehydrate the old experiment metadata before loading a historical checkpoint. Original ZIP bytes/cold native resume not verified. No metric promotion; local-only space released.

Saved old-C geometry analysis: selected_endpoint_changes_20261014/RESULT.json and REPORT.md. These do not prove scorer identity errors or clamp/gradient causal failure. Do not rerun completed diagnostics, engineering pairs, or negative R/P2/quality experiments to manufacture activity while waiting. Necessary new actions should depend on the real E1 result.

## Suggested skills

- monitor-experiment: consume the authorized near-completion observer, without early queries.
- analyze-results: compare complete same-model metrics and saved repairs/breaks.
- experiment-audit: verify any new performance claim against actual inputs and outputs; preserve review scope and route limitations.
'''
handoff.write_bytes(handoff_text.encode('utf-8'))
shutil.copyfile(handoff, data / 'NEXT_SESSION.md')
publisher = (root / 'publish_pv_terminal_span_lr_authorized_20261014.py').read_text(encoding='utf-8')
publisher = publisher.replace('pv_strategy_epoch1_entry_publication_20261013.json', 'pv_terminal_span_lr_publication_20261014.json')
publisher = publisher.replace("prior['section']=='20.376.174'", "prior['section']=='20.376.175'")
publisher = publisher.replace('pv_terminal_span_lr_publication_20261014', 'pv_initial_and_dedup_publication_20261014')
# The new publisher must still read the completed previous publication.
publisher = publisher.replace("prior=json.loads((root/'pv_initial_and_dedup_publication_20261014.json').read_bytes())", "prior=json.loads((root/'pv_terminal_span_lr_publication_20261014.json').read_bytes())")
publisher = publisher.replace('pv_terminal_span_lr_20261014', 'pv_initial_and_dedup_20261014')
publisher = publisher.replace('PV_TERMINAL_SPAN_LR', 'PV_INITIAL_AND_DEDUP')
publisher = publisher.replace('Publish reviewed saved PV diagnosis', 'Publish saved PV evidence and verified archive availability')
publisher = publisher.replace('.tmp_pv_terminal_span_lr', '.tmp_pv_initial_and_dedup')
publisher = publisher.replace("b'20.376.175' not in old", "b'20.376.176' not in old")
publisher = publisher.replace('## §20.376.175｜PV策略终态与边界融合学习率正常训练（2026-10-14）', '## §20.376.176｜低B学习率完整起点与重复权重去重（2026-10-14）')
publisher = publisher.replace("section='20.376.175'", "section='20.376.176'")
publisher = publisher.replace('Record completed PV strategy and isolated span LR normal training', 'Record complete PV E0 and equal model checkpoint deduplication')
publisher = publisher.replace('pv_terminal_span_lr_local_commit_20261013.json', 'pv_initial_and_dedup_local_commit_20261014.json')
publisher = publisher.replace('selected_epoch=2,', 'selected_epoch=0,')
publisher = publisher.replace('PV_C_complete_E2_5588_4623_lower_span_lr_normal_started_new_metrics_pending', 'Lower_B_LR_complete_E0_5677_4920_real_first_epoch_started')
publisher = publisher.replace('observer_native_session=None,', "observer_native_session=summary['observer_native_session'],")
publisher = publisher.replace('new lower_span_lr controller and native child verified at saved startup witness; C closed', 'lower_span_lr controller and child verified03:20:49; real first epoch35batches; no new trained complete metric')
ast.parse(publisher)
path = root / 'publish_pv_initial_and_dedup_authorized_20261014.py'
assert not path.exists()
path.write_bytes(publisher.encode('utf-8'))
(data / 'SOURCE_BINDINGS.json').write_bytes((json.dumps(hashes, indent=2) + '\n').encode())
print(json.dumps({'prepared_files': len(list(data.iterdir())), 'temporary_handoff': str(handoff),
    'new_publisher': str(path), 'remote_calls': 0, 'NN_calls': 0}))
