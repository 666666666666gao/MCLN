"""Record the actual closed CPU attempt, preserving the sealed source tracker."""
import datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent / 'referit_native_preparation_20261010'
audit = json.loads((root / 'source_review/EXPERIMENT_CODE_REVIEW_R2.json').read_bytes())
for name, digest in audit['audited_input_hashes'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest.removeprefix('sha256:')
cpu = root / 'cpu_execution'
execution = json.loads((cpu / 'CPU_EXECUTION.json').read_bytes())
result = json.loads((cpu / 'CPU_PARSED_RESULT.json').read_bytes())
assert execution['exit_code'] == 0 and execution['source_audit_sha256'] == hashlib.sha256((root / 'source_review/EXPERIMENT_CODE_REVIEW_R2.json').read_bytes()).hexdigest()
assert result['real_dataset_rows'] == 0 and result['full_PV_factory_checked'] is False
tracker = root / 'refine-logs/EXPERIMENT_TRACKER.md'
snapshot = root / 'source_review/R2_CLOSED_TRACKER_INPUT.md'
assert not snapshot.exists()
raw = tracker.read_bytes()
snapshot.write_bytes(raw)
alias = {str(tracker):dict(snapshot_path=str(snapshot), sha256=hashlib.sha256(raw).hexdigest(),
    reason='Source R2 reviewed the pre-execution tracker; this exact input is archived before live status update')}
(root / 'source_review/R2_SEALED_TRACKER_SNAPSHOT.json').write_text(json.dumps(alias, indent=2) + '\n', encoding='utf-8')
text = tracker.read_text(encoding='utf-8')
assert text.count('SOURCE_PREPARED') == 2 and text.count('CPU loss／梯度检查 | NOT_RUN') == 1
text = text.replace('SOURCE_PREPARED', 'SOURCE_R2_REVIEWED')
text = text.replace('14 文件 AST；6 文件必要变更，等待独立复核', '14 文件 AST；6 文件必要变更，R2 独立源码复核 0 阻塞')
text = text.replace('CPU loss／梯度检查 | NOT_RUN | 不计为真实数据或准确率',
    'CPU loss／梯度检查 | EXECUTED_OUTCOME_AUDIT_PENDING | 隔离子进程退出 0；合成目标／梯度面板，真实数据行数 0')
tracker.write_text(text, encoding='utf-8')
report = dict(status='REFERIT_SYNTHETIC_TARGET_CPU_EXECUTED_OUTCOME_AUDIT_PENDING',
    time_cst=datetime.datetime.now().astimezone().isoformat(),
    execution=str(cpu / 'CPU_EXECUTION.json'), result=str(cpu / 'CPU_PARSED_RESULT.json'),
    source_review=str(root / 'source_review/EXPERIMENT_CODE_REVIEW_R2.json'),
    execution_finished_cst=execution['finished_cst'], actual_reviewer_outcome_pending=True,
    real_dataset_rows=0, author_states_or_full_PV_constructed=False,
    Nr_Sr_training_launched=False, normal_training_status_reads=0, GPU_calls=0,
    original_source_tracker_exact_snapshot=str(snapshot), full_goal_complete=False)
(root / 'ACTUAL_CPU_TARGET_INTAKE.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k:report[k] for k in ('status','execution_finished_cst','real_dataset_rows',
    'Nr_Sr_training_launched','normal_training_status_reads')}))
