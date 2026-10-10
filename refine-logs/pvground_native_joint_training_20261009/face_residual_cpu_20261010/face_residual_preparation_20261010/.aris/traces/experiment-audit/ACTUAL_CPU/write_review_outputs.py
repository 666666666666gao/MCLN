"""Write only this outcome review and its designated trace; seal no private stderr."""
import datetime
import hashlib
import json
import os
from pathlib import Path

root = Path(r'C:\Users\gb\.codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_residual_preparation_20261010')
trace = root/'.aris/traces/experiment-audit/ACTUAL_CPU'
out = root/'actual_CPU_review'
evidence = json.loads((trace/'DETERMINISTIC_CHECKS.json').read_bytes())
now = datetime.datetime.now().astimezone().isoformat()
agent = '/root/pvg_face_residual_cpu_actual_20261010'
parent_request = r'''按 experiment-audit 要求做 fresh actual-outcome review。读 C:\Users\gb\.codex\skills\experiment-audit\SKILL.md 与 shared-references\local-codex-policy.md 的需要部分。具体完整任务在 C:\Users\gb\.codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_residual_preparation_20261010\ACTUAL_CPU_AUDIT_REQUEST.md；请直接读实际源码、实际原始传输/child stdout和结果，不依赖执行者总结。请求model=gpt-6-astra,reasoning_effort=max,fork_turns=none；没有独立身份凭据，所以identity UNATTESTED,same-family/provisional。不得 SSH/GPU/查当前训练/改待审输入/重新执行模块，仅写审查报告和trace；对已有证据做必要只读本地核查即可。源复核R1/R2与失败attempt均保留；私有SSH stderr不要公开/整段引用。仅审查合成模块CPU执行是否支撑限定工程结论，不宣称1301fullPV/fulloptimizerrecover/REC/训练准入。保存报告与原始request/response及输出封存，按任务文件给定格式输出。'''


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


supported = [
    '保存的单次 attempt2 传输与 CPU child 均 exit 0；Torch 1.10.2+cu111，seed 2027，合成模块检查完成。',
    '按未变初始化绑定核对 d06/f989 checkpoint SHA 并 strict load A 的10项、B 的14项模块状态的已审 child 路径完成。',
    '两个 source 模式的零输出中心/尺寸均通过与已加载 prior 的 torch.equal 检查；新增 head 隐藏 token 列0–31及fraction列324实际清零。',
    'prior29793、新增23425、合计53218参数；合计20项模块state、新增6项，源断言与独立整数核对一致。',
    '独立0.5内部gate夹具中两次新head反传/AdamW内存更新；首步输出层非零梯度，第一次更新后首层非零梯度。',
    '供应残差经过实际Torch refine_one倒置端点路径，执行raw size=-3及最终size=3的断言。',
    '独立梯度模块的state经BytesIO strict reload后，同一refine_one输入的中心/尺寸通过torch.equal。',
    'R1原件/失败传输保留、R2输入连续性及实际raw/base64/落盘结果一致；没有把当前launcher冒充R1原件。',
]
unsupported = [
    '真实数据GT、真实loader/场景、native criterion或正式REC/精度增益。',
    '1301-state native factory、完整PV恢复、完整optimizer/RNG恢复或f989 wrapper完整forward恢复。',
    '已加载f989 prior实际gate饱和模式下的新head梯度或A/B/PV主干联合梯度。',
    '网络已经预测或学会供应的倒置面残差；来源信息控制的实际效果或消融收益。',
    'GPU执行、GPU/正式训练准入、有效研究贡献或整体研究目标完成。',
    '当前主训练状态或C-off-first执行现状的独立观测；本审查未查询或修改它们。',
    'OpenSSH内部路径解析的精确失败根因、每个传递依赖的独立运行来源证明、独立后端模型身份凭据。',
]
checks = {
    'gt_provenance': dict(status='PASS', details='Random and hand-built synthetic inputs; loaded prior is an engineering equivalence reference, not real GT.', evidence=['check_face_residual_modules_cpu.py:63', 'check_face_residual_modules_cpu.py:95', 'CPU_CHECK_PLAN.md:5', 'cpu_execution_attempt2/CPU_MODULE_RESULT.json:36']),
    'score_normalization': dict(status='PASS', details='Raw gradient norms and artificial MSE; geometry/count input normalization is not reported as a performance score.', evidence=['check_face_residual_modules_cpu.py:109', 'check_face_residual_modules_cpu.py:120', '../source/extremal_span_mixer.py:76']),
    'result_existence': dict(status='PASS', details='All seven attempt2 files exist; raw base64 payloads match saved child output and result bytes; review digest and chronology match.', evidence=['cpu_execution_attempt2/RAW_STDOUT.json:1', 'cpu_execution_attempt2/CPU_EXECUTION.json:2', 'cpu_execution_attempt2/CPU_EXECUTION.json:9', 'run_face_residual_cpu_authorized.py:79', '.aris/traces/experiment-audit/ACTUAL_CPU/DETERMINISTIC_CHECKS.json']),
    'dead_code': dict(status='PASS', details='All claimed tests are called sequentially before the final result write. No claim relies on an unused metric or future native factory.', evidence=['check_face_residual_modules_cpu.py:73', 'check_face_residual_modules_cpu.py:106', 'check_face_residual_modules_cpu.py:138', 'check_face_residual_modules_cpu.py:155', 'check_face_residual_modules_cpu.py:169']),
    'scope': dict(status='PASS', details='One synthetic CPU child, one seed, B=1/Q=256/SP=8/points=50000. Separate gradient and supplied-residual fixtures, module-only refine_one recovery.', evidence=['CPU_CHECK_PLAN.md:7', 'CPU_CHECK_PLAN.md:9', 'check_face_residual_modules_cpu.py:94', 'check_face_residual_modules_cpu.py:126', 'check_face_residual_modules_cpu.py:144', 'cpu_execution_attempt2/CPU_MODULE_RESULT.json:38']),
    'eval_type': 'synthetic_proxy',
    'evaluation_type_classification': dict(status='PASS', classification='synthetic_proxy', subtype='synthetic_module_engineering', claim_ceiling='The listed bounded engineering checks only; no performance or method-efficacy acceptance.'),
}
report = dict(
    audit_skill='experiment-audit', verdict='PASS', bounded_check_verdict='PASS', overall_verdict='PASS', integrity_status='pass',
    reason_code='bounded_actual_synthetic_cpu_engineering_supported', blocking_issue_count=0, blocking_issues=[], nonblocking_issues=[],
    summary='Recorded attempt2 supports the declared synthetic CPU module checks. No broader method efficacy, full-model recovery, GPU or training-admission conclusion is accepted.',
    date='2026-10-10', generated_at=now, agent_id=agent, verdict_id=agent, reviewer_agent_id=agent,
    requested_model='gpt-6-astra', requested_reasoning_effort='max', requested_fork_turns='none',
    actual_model='UNATTESTED', actual_reasoning_effort='UNATTESTED', reviewer_identity='UNATTESTED', independent_identity_evidence=None,
    executor_model='UNATTESTED', executor_family='openai', reviewer_model='UNATTESTED', reviewer_family='openai', reviewer_reasoning='UNATTESTED',
    review_independence='same-family', acceptance_status='provisional', auditor='Fresh Codex reviewer; requested Astra/max; actual identity UNATTESTED',
    supported_scope=supported, unsupported_scope=unsupported, checks=checks,
    eval_type='synthetic_proxy', engineering_subtype='synthetic_module_engineering', formal_method_efficacy_acceptance=False,
    GPU_training_admission=False, current_training_or_C_off_first_changed=False, full_goal_complete=False,
    observed_execution=evidence['execution'], elapsed_child_seconds=evidence['elapsed_child_seconds'], observed_result=evidence['result'],
    checkpoint_bindings=evidence['expected_checkpoint_bindings'],
    input_continuity=dict(R2_input_count=len(evidence['R2_input_hash_matches']), all_R2_input_hashes_match=all(evidence['R2_input_hash_matches'].values()), R1_launcher_verified_at_archived_path=True, current_launcher_distinct_from_R1=True, all_R1_unchanged_inputs_match=True, R1_R2_reports_preserved=True),
    provenance_limits=[evidence['checkpoint_evidence_limit'], evidence['whole_mask_range_evidence_limit'], 'The result field uses bitwise terminology; the actual assertion is torch.equal on centers/sizes, without an independent byte dump.', 'GPU_calls/query-count zero fields are source-scope declarations, not whole-machine instrumented counters.'],
    first_failed_attempt=evidence['first_transport'],
    deterministic_verification=dict(check_count=evidence['check_count'], all_checks_pass=evidence['all_checks_pass'], checks=evidence['checks'], trace_result=str(trace/'DETERMINISTIC_CHECKS.json')),
    audited_input_hashes=evidence['audited_input_hashes'],
    audited_input_sections=dict(NATIVE_SOURCE_PORT='model_source, env_spec_sha256, three A/B/geometry sources, whole_mask_range entry; full file hashed', source_conditioned_init='support/span paths and SHA plus env binding; not full initialization acceptance', private_SSH_stderr='length/hash/error-category check only; no raw text copied', prior_R2_auxiliary_policy_files='digest continuity only; not a new semantic review of unrelated skills', env_spec='canonical digest and PYTHONPATH availability context; no credential output'),
    reviewer_actions=dict(audited_sources_modified=False, audited_program_executions=0, torch_imports=0, SSH_calls=0, GPU_calls=0, current_training_status_queries=0, authentication_files_read=False, raw_private_stderr_published=False, writes_limited_to_report_and_designated_trace=True),
    action_items=[], report_path=str(out/'EXPERIMENT_AUDIT.md'), trace_path=str(trace), output_seal_path=str(out/'OUTPUT_SHA256.json'),
)
write_json(out/'EXPERIMENT_AUDIT.json', report)
request = dict(call_number=1, purpose='actual-bounded-cpu-outcome-review', recorded_at=now, tool='collaboration.spawn_agent', agent_id=agent,
               requested_model='gpt-6-astra', requested_reasoning_effort='max', requested_fork_turns='none', actual_identity='UNATTESTED',
               timestamp_scope='Recorded during this review; no independently attested spawn timestamp.', prompt=parent_request,
               task_request_file=str(root/'ACTUAL_CPU_AUDIT_REQUEST.md'), task_request_sha256=sha(root/'ACTUAL_CPU_AUDIT_REQUEST.md'),
               task_request_raw_copy='001-actual-cpu.request.md')
write_json(trace/'001-actual-cpu.request.json', request)
(trace/'001-actual-cpu.request.md').write_bytes((root/'ACTUAL_CPU_AUDIT_REQUEST.md').read_bytes())
(trace/'001-actual-cpu.response.md').write_bytes((out/'EXPERIMENT_AUDIT.md').read_bytes())
write_json(trace/'001-actual-cpu.meta.json', dict(call_number=1, purpose='actual-bounded-cpu-outcome-review', timestamp=now, agent_id=agent,
    requested_model='gpt-6-astra', requested_reasoning_effort='max', requested_fork_turns='none', actual_model='UNATTESTED', actual_reasoning_effort='UNATTESTED',
    reviewer_identity='UNATTESTED', review_independence='same-family', acceptance_status='provisional', status='ok', verdict='PASS', bounded_check_verdict='PASS', blocking_issue_count=0,
    full_response_is_report=True, raw_request_copied=True, candidate_executed=False))
write_json(trace/'run.meta.json', dict(skill='experiment-audit', run_id='ACTUAL_CPU', recorded_at=now, executor='codex', executor_model='UNATTESTED', executor_family='openai',
    requested_reviewer_model='gpt-6-astra', requested_reasoning_effort='max', actual_reviewer_identity='UNATTESTED', review_independence='same-family', acceptance_status='provisional',
    project_dir=str(root), agent_id=agent, write_scope='actual_CPU_review and .aris/traces/experiment-audit/ACTUAL_CPU only'))
write_json(trace/'review_trace_event.json', dict(event='review_trace', skill='experiment-audit', purpose='actual-bounded-cpu-outcome-review', agent_id=agent, trace_path=str(trace), status='ok',
    event_location_reason='Kept inside the task-authorized trace write scope; shared meta/events was not modified.'))
final_response = '限定实际 CPU 审查 **PASS，0 个阻断项**；36 项确定性核对通过。身份 **UNATTESTED，same-family/provisional**。\n\n仅支持报告列出的合成模块工程结论，不支持完整 PV/optimizer 恢复、REC 或训练准入。未执行 SSH、GPU、模块重跑或当前训练查询。\n\n报告、原始请求/响应、工具 trace 和输出 SHA256 封存已写入 `actual_CPU_review` 与 `.aris/traces/experiment-audit/ACTUAL_CPU`。\n'
(trace/'001-actual-cpu.agent-final.txt').write_text(final_response, encoding='utf-8')

input_matches = {name: sha(Path(name)) == expected for name, expected in evidence['audited_input_hashes'].items()}
assert all(input_matches.values())
assert (trace/'001-actual-cpu.response.md').read_bytes() == (out/'EXPERIMENT_AUDIT.md').read_bytes()
assert (trace/'001-actual-cpu.request.md').read_bytes() == (root/'ACTUAL_CPU_AUDIT_REQUEST.md').read_bytes()
assert report['blocking_issue_count'] == 0 and report['deterministic_verification']['all_checks_pass']
write_json(trace/'FINALIZATION_CHECK.json', dict(recorded_at_cst=now, all_audited_inputs_unchanged=True, audited_input_count=len(input_matches),
    input_hash_matches=input_matches, raw_request_matches_task_file=True, full_response_matches_report=True, verdict='PASS', bounded_check_verdict='PASS', blocking_issue_count=0,
    source_review_reports_preserved=True, private_SSH_stderr_not_copied=True, candidate_executions=0, SSH_calls=0, GPU_calls=0, current_training_queries=0))
files = [p for p in out.iterdir() if p.is_file() and p.name != 'OUTPUT_SHA256.json'] + [p for p in trace.rglob('*') if p.is_file()]
sealed = {os.path.relpath(p, out).replace('\\', '/'): sha(p) for p in sorted(files)}
write_json(out/'OUTPUT_SHA256.json', dict(path_base='directory containing this seal (actual_CPU_review)', files=sealed, excludes=['OUTPUT_SHA256.json', 'all private SSH stderr originals']))
assert all(sha(out/name) == expected for name, expected in sealed.items())
print(json.dumps(dict(verdict='PASS', bounded_check_verdict='PASS', blocking_issue_count=0, input_count=len(input_matches), sealed_output_count=len(sealed),
    report_sha256=sha(out/'EXPERIMENT_AUDIT.md'), report_json_sha256=sha(out/'EXPERIMENT_AUDIT.json'), seal_sha256=sha(out/'OUTPUT_SHA256.json'), all_inputs_unchanged=True), ensure_ascii=False))
