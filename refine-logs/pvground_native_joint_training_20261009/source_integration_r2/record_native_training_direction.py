"""Record the user's normal-training requirement without touching active jobs."""
import datetime
import json
from pathlib import Path

root = Path(__file__).resolve().parent
home = Path('C:/Users/gb')
now = datetime.datetime.now().astimezone().isoformat()
direction = dict(
    recorded_cst=now,
    user_instruction='尽量做在正常训练中，尽量不要后处理那些',
    implementation='PVGround.forward plus native normal epoch trainer and unified criterion',
    original_trainable_core_jointly_finetuned=True,
    original_native_roberta_freeze_preserved=True,
    evaluation_only_box_replacement=False,
    second_deployed_ranking=False,
    complete_model_optimizer_scheduler_checkpoint_required=True,
    hard_predicted_mask_reference_is_discrete=True,
    source_root=str(root / 'source'),
    preparation=str(root / 'METRIC_PREPARATION.json'),
    source_review_status='R1_SEAL_PENDING_RESUME_DIRECTORY_DEFECT_FOUND',
    source_review_task='/root/pvg_native_joint_source_20261009',
    native_joint_training_started=False,
    actual_gpu_preflight_completed=False,
    runtime_source_data_binding_completed=False,
    storage_admission_completed=False,
    current_frozen_parent_pair_is_mechanism_control=True,
    current_running_pair_modified=False,
    scanrefer_target_minimum_hits=[5658, 4850],
    seed=2027,
    multiseed=False,
    three_effective_contributions_confirmed=False,
    full_goal_complete=False)
(root / 'NORMAL_TRAINING_DIRECTION.json').write_text(
    json.dumps(direction, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
goal_path = home / '.codex/tmp/pvground_referit_mask_reference_20261006/current_research_goals.json'
state_path = home / '.codex/tmp/pvground_query_supported_geometry_20261005/active_continuation_state.json'
for path in (goal_path, state_path):
    record = json.loads(path.read_bytes())
    record['normal_training_requirement'] = direction
    record['full_goal_complete'] = False
    if path == state_path:
        record['current_turn_class'] = 'NATIVE_NORMAL_TRAINING_SOURCE_IMPLEMENTATION_PROGRESS'
        record['latest_verified_wait'] = dict(
            proof=str(home / '.codex/tmp/pvground_extremal_span_evidence_20261009/runner_v1/fit_local_wait_20261009_191416.json'),
            native_session_id=22083, exec_cell_id='347', native_exit_code=0,
            started_cst='2026-10-09T19:14:16.4420067+08:00',
            ended_cst='2026-10-09T19:19:16.4720538+08:00',
            elapsed_seconds=300.0278488, owner_exited=False,
            original_observer_pid=40924, remote_queries=0)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
note = (
    '\n\nPV-Ground ' + now + ': 用户要求后续尽量在正常训练框架中实现、尽量不要后处理。'
    '已准备独立原生PVGround.forward/统一criterion/正常epoch训练入口，保留原版RoBERTa冻结，'
    '其余原生可训练主体与新模块联合微调；完整模型/Adam/scheduler/RNG保存。'
    '当前只是源码与AST检查，fresh源码R1审查正在封存，发现恢复实验目录与best路径的真实缺陷待修。'
    '尚未GPU联合预检、部署或正常训练，运行中的冻结父模型span对照未改动。'
    '原观察器40924的19:14:16–19:19:16本地300秒等待已closed0、没有远端查询，首查仍23:20:53后240秒。'
    '目标5658/4850、三项有效贡献、之后对应作者初始化独立Nr/Sr仍ACTIVE_UNMET；单seed2027。'
    '源码目录 ' + str(root / 'source') + '。\n')
for path in (home / 'memory/2026-10-09.md', home / 'MEMORY.md'):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(note)
print(json.dumps(dict(status='USER_DIRECTION_RECORDED_SOURCE_ONLY', time_cst=now,
    native_joint_training_started=False, remote_queries=0, full_goal_complete=False)))
