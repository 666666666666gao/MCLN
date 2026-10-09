"""Close only the recorded synthetic CPU check, without modifying sealed inputs."""
import datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
prepared = root / 'referit_native_preparation_20261010'
audit_path = prepared / 'actual_CPU_review/EXPERIMENT_AUDIT.json'
seal_path = prepared / 'actual_CPU_review/OUTPUT_SHA256.json'
audit = json.loads(audit_path.read_bytes())
seal = json.loads(seal_path.read_bytes())
assert audit['verdict'] == 'WARN' and audit['bounded_check_verdict'] == 'PASS'
assert audit['blocking_issue_count'] == 0 and audit['nonblocking_issue_count'] == 1
assert audit['review_independence'] == 'same-family' and audit['acceptance_status'] == 'provisional'
for name, metadata in seal['files'].items():
    raw = Path(name).read_bytes()
    assert len(raw) == metadata['bytes'] and hashlib.sha256(raw).hexdigest() == metadata['sha256']
for inputs in (audit['audited_input_hashes'], audit['primary_file_hashes']):
    for name, digest in inputs.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest.removeprefix('sha256:')
receipt = json.loads((prepared / 'cpu_execution/CPU_EXECUTION.json').read_bytes())
assert receipt['exit_code'] == 0 and receipt['GPU_calls'] == 0
assert receipt['current_training_status_reads'] == 0
assert receipt['full_PV_or_author_state_checked'] is False and receipt['Nr_Sr_training_launched'] is False
value = dict(status='REFERIT_NATIVE_SYNTHETIC_CPU_OUTCOME_REVIEW_CLOSED',
    time_cst=datetime.datetime.now().astimezone().isoformat(),
    audit_verdict='WARN', bounded_check_verdict='PASS', blocking_issue_count=0,
    nonblocking_issue_count=1, audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
    seal_sha256=hashlib.sha256(seal_path.read_bytes()).hexdigest(),
    actual_CPU_outcome_review_pending=False, GPU_or_control_training_admission=False,
    reviewer_identity='UNATTESTED', review_independence='same-family', acceptance_status='provisional',
    compatibility_scope='G old/new ScanRefer loss and gradient equal; C loss equal; C old/new gradient equality not asserted',
    executed_scope='Synthetic targets and local gradients through direct native loss_pos_align and G/C helpers',
    prepared_native_criterion_imported=False, real_dataset_rows=0,
    full_PV_or_author_state_checked=False, normal_training_status_queries=0,
    Nr3D_or_Sr3D_training_launched=False, GPU_calls=0,
    first_normal_observation_cst='2026-10-10T08:19:04.678479+08:00',
    source_tracker_and_intake_preserved_as_pre_audit_history=True, full_goal_complete=False)
target = root / 'ACTUAL_REFERIT_CPU_AUDIT_CLOSURE.json'
assert not target.exists()
target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
note = root / 'HANDOFF_REFERIT_NATIVE_SOURCE_CPU_20261010.md'
with note.open('a', encoding='utf-8') as stream:
    stream.write('\n实际CPU结果复核已封存：总体WARN、限定检查PASS、0阻塞、1项非阻塞措辞限制；原始汇总字段需限定为G的损失与梯度等值、C仅损失等值。逐字节回执和封存输入核验通过。CPU调用的是实际父模型源码中的CE方法与新G/C辅助函数，没有导入本轮准备的完整criterion，也没有执行完整Hungarian或训练流程。当前tracker/入库记录保留其复核前历史状态，最新闭合状态见ACTUAL_REFERIT_CPU_AUDIT_CLOSURE.json；不改写已经封存的输入。\n')
print(json.dumps(value, ensure_ascii=False))
