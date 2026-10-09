"""Persist R2 review and trace without modifying any R1 artifact."""
import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOW = datetime.datetime.now().astimezone().isoformat()
AGENT = '/root/pvg_extremal_span_source_20261009'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


baseline = json.loads((HERE / 'R1_ARTIFACT_HASHES.json').read_text(encoding='utf-8-sig'))
for item in baseline:
    assert sha(Path(item['path'])) == item['sha256']
verify = json.loads((HERE / 'R2_STATIC_VERIFICATION.json').read_bytes())
assert verify['status'] == 'PASS_STATIC_ONLY_R1_B1_B2_RESOLVED'
passed = json.loads((HERE / 'verify_r2_002.invocation.json').read_bytes())
failed = json.loads((HERE / 'verify_r2_001.invocation.json').read_bytes())
assert passed['result']['exit_code'] == 0 and failed['result']['exit_code'] == 1
hashes = json.loads((HERE / 'R2_INPUT_HASHES.json').read_bytes())
snapshots = json.loads((HERE / 'R2_SOURCE_SNAPSHOTS.json').read_bytes())
for item in snapshots:
    assert sha(Path(item['snapshot'])) == item['sha256']
raw = (HERE / 'SOURCE_REVIEW_R2.md').read_bytes()
(HERE / 'RAW_RESPONSE_R2.md').write_bytes(raw)
(HERE / '002-source-audit-r2.response.md').write_bytes(raw)
meta = dict(audit_skill='experiment-audit', review_round='R2', date='2026-10-09', generated_at=NOW,
    agent_id=AGENT, verdict_id=AGENT + '@R2', fresh_context=False, original_R1_fresh_context=True,
    review_context='same reviewer minimal follow-up after preserving R1; no new reviewer call',
    requested_backend='codex', requested_model='gpt-6-astra', requested_reasoning_effort='max',
    actual_backend='UNATTESTED', actual_model='UNATTESTED', actual_reasoning_effort='UNATTESTED',
    reviewer_model='UNATTESTED', reviewer_reasoning='UNATTESTED', reviewer_family='openai',
    review_independence='same-family', acceptance_status='provisional')
report = dict(meta, verdict='PASS', overall_verdict='PASS', verdict_scope='STATIC_ONLY', source_verdict='PASS',
    reason_code='r1_native_forward_and_mode_restoration_source_contracts_resolved',
    blocking_findings=[], resolved_findings=[
        dict(id='B1_NATIVE_DICTIONARY_FORWARD_CONTRACT', status='RESOLVED_STATIC_ONLY',
             evidence=['span_refinement_model.py:41-44', 'R2_STATIC_VERIFICATION.json'],
             details='Single positional inputs argument; exactly one parent(inputs) call; native signature binds.'),
        dict(id='B2_SOURCE_MODE_RESTORATION_IDENTITY', status='RESOLVED_STATIC_ONLY',
             evidence=['span_refinement_model.py:47-61', 'R2_STATIC_VERIFICATION.json'],
             details='Payload stores source_mode/head/optimizer/step/parent_identity; both identity checks precede strict head and optimizer load.'),
        dict(id='W1_INVALID_NOT_PROVEN_EMPTY', status='RESOLVED_COMMENTS_ONLY',
             evidence=['extremal_span_mixer.py:75-78', 'R2_MIXER_COMMENTS.diff'],
             details='Distinguishes absent/degenerate references and zero pooled input versus biased encoded token; executable AST unchanged.')],
    diagnostic=dict(verdict='WARN', deterministic_arithmetic='PASS_FROM_R1', changed_inputs=False, replayed_in_R2=False,
        classification='real_gt_conditioned_offline_representational_diagnostic', deployment_valid=False,
        model_metric=False, iou_upper_bound=False, training_label_dataset=False),
    checks=dict(A_gt_provenance=dict(status='PASS', scope='STATIC_ONLY inference code', runtime_limit='Future matching and split wiring unverified; R1 cached-GT warning preserved'),
        B_score_normalization=dict(status='PASS', scope='STATIC_ONLY'),
        C_result_existence=dict(status='PASS', scope='source repair exists; no runner/M0 claimed'),
        D_dead_code=dict(status='WARN', details='Only AST/hash/signature verification executed; neural/checkpoint helpers unexecuted'),
        E_scope=dict(status='WARN', details='Minimal source follow-up; no actual initialization, optimizer restoration, GPU gradients or performance'),
        F_evaluation_type=dict(status='WARN', classification='STATIC_ONLY_SOURCE_PREPARATION', actual_model_evaluation=False)),
    unchanged_architecture=dict(trainable_parameters_per_arm=29793, trainable_tensors_per_arm=14,
        mixer_executable_AST_unchanged=True, objective_bytes_unchanged=True, all256_preserved_static=True),
    deferred_requirements=[
        'Actual runner derives and verifies exact frozen-parent identity and reconstructs the parent.',
        'Both arms share one frozen parent cache and one pre-span actual-GT assignment; initial hidden tensors match with independent storage and optimizers.',
        'Parent remains eval/frozen, including buffers; same-cache or defined RNG replay handles parent stochastic sampling.',
        'Only after active job closure and required handling: real B8 two-step M0, first-output/second-hidden gradients, actual same-mode/parent state and optimizer serialization/restoration.',
        'Record actual runtime version, memory/throughput, all256 masks/scores/candidates, empty/degenerate/tie witnesses and exact native loss reconciliation.'],
    actual_M0='NOT_AVAILABLE', actual_runner_present=False, launch_approved=False,
    actual_model_builds=0, actual_neural_forwards=0, actual_optimizer_updates=0,
    actual_serialization_or_restoration=False, actual_gradient_witness=False, packages_installed=0, ssh_queries=0,
    audited_input_hashes=hashes, source_snapshots=snapshots,
    R1_preserved_artifacts_verified=len(baseline), R1_manifest_sha256=sha(HERE / 'R1_ARTIFACT_HASHES.json'),
    prior_review_json_sha256=sha(HERE / 'SOURCE_REVIEW.json'),
    raw_report_path=str(HERE / 'RAW_RESPONSE_R2.md'), raw_report_sha256=sha(HERE / 'RAW_RESPONSE_R2.md'),
    trace_path=str(HERE), input_hash_manifest='R2_INPUT_HASHES.json',
    verification=dict(path='R2_STATIC_VERIFICATION.json', sha256=sha(HERE / 'R2_STATIC_VERIFICATION.json'),
        script='verify_r2_source.py', script_sha256=sha(HERE / 'verify_r2_source.py'), actual_exit_code=0,
        chunk_id=passed['result']['chunk_id'], invocation='verify_r2_002.invocation.json'),
    actual_failed_invocations=[dict(invocation='verify_r2_001.invocation.json', log='verify_r2_001.log',
        source='verify_r2_source.failed001.py', source_sha256=sha(HERE / 'verify_r2_source.failed001.py'),
        exit_code=1, chunk_id=failed['result']['chunk_id'],
        reason='Expected-unchanged mixer hash detected the newly authorized comment revision; corrected scope explicitly binds new hash and requires unchanged executable AST.')],
    neural_metric_reported=False, checkpoint_exists=False, current_source_edits_by_reviewer=False,
    global_notes_modified=False, weight_promotion_eligible=False, three_effective_contributions=False, full_goal_complete=False)
write('SOURCE_REVIEW_R2.json', report)
parent_message = 'After finishing and preserving the full original R1 report, perform the separate minimal R2 follow-up exactly as C:\\Users\\gb\\.codex\\tmp\\pvground_extremal_span_evidence_20261009\\SOURCE_R2_REQUEST.txt requests. Do not overwrite any R1 file/trace. Native wrapper now uses forward(inputs) → parent(inputs), and small checkpoint payload/restore explicitly binds source_mode and frozen-parent identity. Changed wrapper SHA1139094df8f282715a795469cedc0dd6c4c95fea3f6b68ce13ab56b4a6ffb68b. Neural mixer/objective and diagnostic unchanged. The proposal prose reflects the repaired contract; evaluate only source scope, no actual M0 or runner claim. Save separate RAW_RESPONSE_R2 and SOURCE_REVIEW_R2 JSON/MD, exact follow-up request, input hash manifest and trace.'
comment_followup = 'Also resolved your nonblocking W1 by comments only in extremal_span_mixer.py:39 archived invalid references can be absent OR degenerate, and zero pooled evidence may become a biased face-encoder token. No executable statement changed. New mixerSHA523203b7a5304ba41ca3bfb66c6911777adb0b55287806ddfed57d8ae9566adf. Please bind R2 to this comment-corrected source, verify AST identical to R1, and append this actual follow-up message to your separate R2 trace. Diagnostic/objective unchanged; no repeated numerical replay needed.'
write('002-source-audit-r2.request.json', dict(call_number=2, purpose='minimal-R2-source-follow-up', timestamp=NOW,
    timestamp_meaning='trace materialization time', tool='native collaboration followup_task and send_message',
    requested_model='gpt-6-astra', requested_reasoning_effort='max', parent_message_exact=parent_message,
    file_request_exact=(HERE / 'SOURCE_R2_REQUEST.exact.txt').read_bytes().decode('utf-8'),
    file_request_sha256=sha(HERE / 'SOURCE_R2_REQUEST.exact.txt'),
    subsequent_parent_message_exact=comment_followup, files_referenced=list(hashes)))
write('002-source-audit-r2.meta.json', dict(meta, call_number=2, purpose='minimal-R2-source-follow-up',
    status='completed', source_verdict='PASS', scope='STATIC_ONLY', blockers=0,
    response_sha256=sha(HERE / '002-source-audit-r2.response.md'), actual_M0='NOT_AVAILABLE'))
write('R2_ACTUAL_FAILED_INVOCATIONS.json', report['actual_failed_invocations'])
write('R2_EXECUTION_RECEIPTS.json', dict(failed_verifier=failed, passed_verifier=passed,
    r1_artifacts_unchanged=38, numerical_diagnostic_replays=0, neural_executions=0))
for item in baseline:
    assert sha(Path(item['path'])) == item['sha256']
print(json.dumps(dict(status='R2_REPORT_WRITTEN', source_verdict='PASS', scope='STATIC_ONLY', blockers=0,
    diagnostic_verdict='WARN_UNCHANGED_R1_ARITHMETIC_PASS', r1_preserved_files=len(baseline),
    source_review_r2_sha256=sha(HERE / 'SOURCE_REVIEW_R2.json'), raw_response_r2_sha256=sha(HERE / 'RAW_RESPONSE_R2.md'))))
