"""Close the actual CPU load evidence without altering reviewed inputs."""
import datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
prepared = root/'referit_author_core_20261010'
audit_path = prepared/'actual_CPU_review/EXPERIMENT_AUDIT.json'
seal_path = prepared/'actual_CPU_review/OUTPUT_SHA256.json'
audit = json.loads(audit_path.read_bytes())
seal = json.loads(seal_path.read_bytes())
assert audit['verdict'] == 'WARN' and audit['bounded_CPU_evidence_verdict'] == 'PASS'
assert audit['blocking_issue_count'] == 0
assert audit['review_independence'] == 'same-family' and audit['acceptance_status'] == 'provisional'
for name,digest in audit['audited_input_hashes'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
for name,digest in seal['files'].items():
    assert hashlib.sha256((prepared/name).read_bytes()).hexdigest() == digest
execution = json.loads((prepared/'cpu_execution/CPU_EXECUTION.json').read_bytes())
result = json.loads((prepared/'cpu_execution/AUTHOR_CORE_CPU_RESULT.json').read_bytes())
assert execution['exit_code'] == 0 and execution['normal_training_queries'] == execution['GPU_calls'] == 0
assert result['status'] == 'ACTUAL_AUTHOR_CORE_NATIVE_FACTORY_CPU_LOAD_PASS'
assert set(result['cases']) == {'nr3d','sr3d'}
for case in result['cases'].values():
    assert case['author_core_tensors'] == 1235 and case['exact_loaded_core_tensors'] == 1234
    assert case['full_model_state_tensors'] == 1295
    assert case['core_key_shape_dtype_and_values_exact'] is True and case['A_and_B_output_zero_verified'] is True
    assert case['CUDA_initialized'] is False and case['real_loader_rows'] == case['forwards'] == case['optimizer_steps'] == 0
value = dict(status='ACTUAL_AUTHOR_CORE_CPU_LOAD_EVIDENCE_REVIEW_CLOSED',
    time_cst=datetime.datetime.now().astimezone().isoformat(), audit_verdict='WARN',
    bounded_check_verdict='PASS', blocking_issue_count=0,
    audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
    seal_sha256=hashlib.sha256(seal_path.read_bytes()).hexdigest(),
    actual_CPU_outcome_review_pending=False, GPU_or_control_training_admission=False,
    reviewer_identity='UNATTESTED', review_independence='same-family', acceptance_status='provisional',
    supported_scope='Two actual native CPU factories; strict corresponding author core load and exact 1234 retained core shape/dtype/values; deterministic position buffer and 1295 full states; A/B output-layer state zeros only',
    author_protocol_verified=False, real_loader_verified=False, GPU_forward_loss_gradient_verified=False,
    full_optimizer_recovery_verified=False, normal_precision_result=None,
    first_normal_observation_cst='2026-10-10T08:19:04.678479+08:00',
    active_normal_training_source_changed=False, current_normal_training_queries=0,
    Nr_or_Sr_training_launched=False, new_weight_files=0, full_goal_complete=False)
target = root/'ACTUAL_AUTHOR_CORE_CPU_AUDIT_CLOSURE.json'
assert not target.exists()
target.write_text(json.dumps(value,indent=2)+'\n')
with (root/'HANDOFF_REFERIT_AUTHOR_CORE_CPU_20261010.md').open('a',encoding='utf-8') as stream:
    stream.write('\n实际结果复核已封存：限定CPU证据PASS，总体WARN、范围内0阻塞。137项输入、26项输出及原始回执字节核验通过；A/B仅输出层状态为零，不是本轮执行forward的证明。作者对象/混合训练flags与本次构造flags的差异明确保留，真实训练未放行；完整参数加载与真正训练效果继续分开。源码PLAN/SOURCE_PREPARATION保留执行前历史状态，最新闭合状态见ACTUAL_AUTHOR_CORE_CPU_AUDIT_CLOSURE.json。\n')
print(json.dumps(value))
