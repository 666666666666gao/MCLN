"""Seal the completed read-only review and preserve its response and metadata."""
import datetime
import hashlib
import json
from pathlib import Path

trace = Path(__file__).resolve().parent
root = trace.parents[3]
out = root / 'native_direct_controls_20261010/actual_source_review'
static = json.loads((out / 'DETERMINISTIC_CHECKS.json').read_bytes())
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
task = '/root/pvg_native_direct_control_source_20261010'
scope = 'ISOLATED_NATIVE_CONTROL_SOURCE_NOT_LAUNCHED'
hashes = {path: 'sha256:' + entry['sha256'] for path, entry in static['inputs'].items()}
for path, digest in hashes.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest[7:]

identity = dict(requested_reviewer_model='gpt-6-astra', requested_reviewer_reasoning='max',
                actual_reviewer_model='UNATTESTED', actual_reviewer_reasoning='UNATTESTED',
                reviewer_model='UNATTESTED', reviewer_reasoning='UNATTESTED', reviewer_family='openai',
                review_independence='same-family', acceptance_status='provisional',
                model_attestation_available=False, task_name=task, agent_id='UNATTESTED',
                actual_identity_note='Requested route is not actual-model attestation; no actual identity token is available.')
checks = {
    'gt_provenance': dict(status='WARN', details='Reviewed target assembly uses batch GT; prediction-derived training consistency and reference geometry are not evaluation GT. Actual dataset and evaluator internals are outside this source set.',
        evidence=['R/train_dist_mod.py:55-96', 'R/models/losses.py:856-885', 'R/models/losses.py:592-621', 'R/selected_query_mask_objective.py:25-32', 'R/native_mask_geometry.py:13-23', 'C/check_direct_control_modules_cpu.py:73-88']),
    'score_normalization': dict(status='PASS', details='Reviewed grounding reporting retains integer hits and divides by GT counts. No self-max/mean normalization of accuracy. Formal evaluator implementation was not supplied.',
        evidence=['R/train_dist_mod.py:237-257', 'R/native_root_bbs.py:4-9', 'C/source/extremal_span_mixer.py:99-108']),
    'result_existence': dict(status='WARN', details='Prepared/not-constructed/not-launched labels agree with available evidence. No local CPU execution/witness file exists at the audit snapshot. Historical numbers 5677/4920 have no result artifact in this review set and are not approved.',
        evidence=['C/DIRECT_CONTROL_PREPARATION.json:69-93', 'C/EXPERIMENT_TRACKER.md:7-13', 'C/DIRECT_CONTROLS.md:35', 'C/check_direct_control_modules_cpu.py:140-161', 'DETERMINISTIC_CHECKS.json']),
    'dead_code': dict(status='WARN', details='Factory, forward, criterion and evaluator calls are connected by source. DIoU and native-replacement witness helpers are not called in reviewed artifacts. Source reachability is not execution evidence.',
        evidence=['R/train_dist_mod.py:111-129', 'R/models/pv_ground.py:567-613', 'R/models/losses.py:546-557', 'R/pvground_semantic_assignment.py:53-93', 'R/train_dist_mod.py:208-257']),
    'scope': dict(status='WARN', details='Zero actual control executions or dataset evaluations are established. Four prepared configurations share data/score/loss/optimizer policy, with intended freezing/input ablations; single seed and common trained parent history limit later claims.',
        evidence=['C/common_behavior_check/NORMAL_NATIVE_RUN_PROTOCOL.json:74-88', 'C/DIRECT_CONTROLS.md:10-23', 'R/main_utils.py:292-306', 'C/source/native_model_initialization.py:62-84']),
    'evaluation_type': dict(status='PASS', actual='static_source_audit',
        planned_cpu_checker='synthetic_module_engineering_fixture_not_accuracy_evaluation',
        checklist_taxonomy_for_module_equivalence='synthetic_proxy',
        planned_formal_evaluation='real_gt_intended_not_executed_or_provenance_certified',
        evidence=['C/check_direct_control_modules_cpu.py:73-88', 'C/check_direct_control_modules_cpu.py:140-159'])
}
warnings = [
    dict(id='W1', title='Execution evidence is still absent', blocking_for_cpu_checker=False,
         details='Neither the CPU checker nor full PV/criterion/optimizer/cold recovery was executed in this review. The checker does not establish those full-model properties.'),
    dict(id='W2', title='Formal data/evaluator and complete imported repository remain outside the source boundary', blocking_for_cpu_checker=False,
         details='Before formal result approval, verify the actual dataset/evaluator, complete source environment and corresponding records. No active-run state was read.'),
    dict(id='W3', title='Module coverage must not be promoted to dtype or full-chain gradient guarantees', blocking_for_cpu_checker=False,
         details='Standalone strict loads do not independently attest dtype equality; the fixed-half gradient assertion holds for an already materialized independent reference. Invalid-reference native-prior gradients can differ in full PV.'),
    dict(id='W4', title='Historical accuracy literals are not audited results', blocking_for_cpu_checker=False,
         details='5677/4920 are labeled historical in DIRECT_CONTROLS.md:35; their result artifacts are not inputs.'),
    dict(id='W5', title='Dormant helpers are not completed evidence', blocking_for_cpu_checker=False,
         details='calculate_diou_3d, verify_native_replacement and mask_range_evidence are not invoked in the reviewed control/checker paths. No cleanup is required for this task.')
]
report = dict(
    audit_skill='experiment-audit', verdict='WARN', overall_verdict='warn', integrity_status='warn',
    reason_code='source_consistent_execution_and_full_model_evidence_pending',
    summary='No source blocker found before the bounded CPU module checker. Source/configuration isolation and intended ablations are supported; neural execution and all formal performance claims remain unverified.',
    blocking_issue_count=0, blocking_issues=[],
    blocking_issue_count_scope='Source blockers for the isolated CPU module checker only; not GPU/training admission.',
    execution_scope=scope, audit_result_approval=False,
    generated_at=now, date='2026-10-10', auditor='fresh Codex reviewer; actual model UNATTESTED',
    verdict_id=task, executor_model='UNATTESTED', executor_family='openai', **identity,
    trace_path=str(trace), audited_input_hashes=hashes, audited_input_count=len(hashes),
    audited_inputs_unchanged_at_seal=True,
    citation_roots={'R/': str(root / 'source'), 'C/': str(root / 'native_direct_controls_20261010'),
                    'P/': str(root), 'M': 'C:/Users/gb/.codex/tmp/pvground_extremal_span_evidence_20261009/runner_v1/source_review/snapshots/034_whole_mask_range.py'},
    checks=checks, warnings=warnings,
    deterministic_checks=dict(status=static['status'], python_files_parsed=static['python_files_parsed'],
        changed_sources=static['changed_sources'], byte_identical_sources=static['byte_identical_sources'],
        specification_differences=static['specification_differences'], protocol_differences=static['protocol_differences'],
        controllers_byte_identical=True, manifest_hashes_verified=True, cpu_bundle_hashes_verified=True,
        neural_imports_or_execution=False, checker_file='DETERMINISTIC_CHECKS.json'),
    checker_status='PREPARED_NOT_EXECUTED', checker_execution_observed=False,
    checker_local_witness_exists=static['cpu_module_witness_exists_locally'],
    checker_local_execution_receipt_exists=static['cpu_module_execution_exists_locally'],
    actual_full_PV_constructor_or_1295_state_checked=False, actual_payload_loads_checked=False,
    actual_native_criterion_checked=False, actual_optimizer_scheduler_checked=False,
    actual_full_cold_recovery_checked=False, actual_formal_accuracy=None,
    formal_results_status='NO_NEW_CONTROL_FORMAL_RESULTS_IN_REVIEWED_ARTIFACTS_NOT_LAUNCHED',
    actual_scope=dict(control_runs=0, completed_training_seeds=0, dataset_rows_evaluated=0,
                      planned_configurations=4, planned_seed=2027, planned_epochs=3,
                      planned_batch_size=8, planned_formal_rows=9508),
    gpu_admission=dict(status='NOT_ADMITTED_BY_THIS_SOURCE_AUDIT', ready=False,
        pending_evidence=['Actual full PV factory and parent/state loading',
                          'Controlled same-input common/native output and native-loss comparison',
                          'Actual per-arm gradient and optimizer-group checks',
                          'Native dataset/evaluator/score/criterion and complete environment bindings',
                          'Full checkpoint save and cold recovery',
                          'Previously specified active-normal terminal/result review and necessity decision']),
    claims=[dict(id='source_and_config_isolation', impact='supported_at_sealed_hashes'),
            dict(id='declared_ablation_semantics', impact='supported_by_source_with_history_and_capacity_qualifiers'),
            dict(id='CPU_module_equivalence_or_gradients_passed', impact='unsupported_until_executed'),
            dict(id='full_constructor_criterion_optimizer_recovery_passed', impact='unsupported'),
            dict(id='A_B_C_effectiveness_accuracy_target_or_multiseed_claims', impact='unsupported')],
    reviewer_activities=dict(local_static_checks_only=True, neural_calls=0, gpu_calls=0, ssh_calls=0,
                            current_training_status_reads=0, reviewed_source_mutations=0,
                            credentials_or_private_memory_reads=0),
    tool_issue=dict(first_python_invocation='PATH python failed: No pyvenv.cfg file',
                    completed_static_invocation='E:/python.exe -I -B reviewer_static_checks.py',
                    issue_resolved=True)
)
(out / 'EXPERIMENT_AUDIT.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

supplemental_prompt = '\n\nSupplemental artifact paths supplied during review:\n' + '\n'.join(
    path for path in hashes if path.endswith(('CPU_BUNDLE.json', 'prepare_direct_control_cpu_bundle.py',
                                            'run_direct_control_modules_cpu_authorized.py', '034_whole_mask_range.py')))
request_record = dict(call_number=1, purpose='native-direct-controls-source-review', timestamp=now,
    timestamp_scope='Request trace serialized at review completion; original call time unavailable',
    tool='spawn_agent', model='UNATTESTED', requested_model='gpt-6-astra',
    requested_reasoning_effort='max', actual_reasoning_effort='UNATTESTED',
    files_referenced=list(hashes), prompt=(trace / 'request.txt').read_text(encoding='utf-8') + supplemental_prompt)
(trace / '001-native-source-review.request.json').write_text(json.dumps(request_record, indent=2) + '\n', encoding='utf-8')
response = (out / 'EXPERIMENT_AUDIT.md').read_bytes()
(trace / '001-native-source-review.response.md').write_bytes(response)
meta = dict(call_number=1, purpose='native-direct-controls-source-review', timestamp=now,
    status='ok', verdict='WARN', blocking_issue_count=0, execution_scope=scope,
    duration_ms=None, duration_status='NOT_MEASURED', response_encoding='UTF-8',
    response_content='Full authored substantive review, identical to EXPERIMENT_AUDIT.md', **identity)
(trace / '001-native-source-review.meta.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
run = dict(skill='experiment-audit', run_id=trace.name, started_at=None,
    started_at_status='ORIGINAL_AGENT_START_TIME_UNAVAILABLE',
    first_recorded_static_check_at=static['generated_at'], completed_at=now,
    executor='codex', executor_model='UNATTESTED', executor_family='openai',
    project_dir=str(root), **identity)
(trace / 'run.meta.json').write_text(json.dumps(run, indent=2) + '\n', encoding='utf-8')
event = dict(event='review_trace', skill='experiment-audit', purpose='native-direct-controls-source-review',
    task_name=task, agent_id='UNATTESTED', trace_path=str(trace), status='ok', verdict='WARN',
    event_location_note='Kept in permitted trace directory; no write to out-of-scope .aris/meta/events.jsonl')
(trace / 'review_trace.events.jsonl').write_text(json.dumps(event) + '\n', encoding='utf-8')

instruction_paths = [Path('C:/Users/gb/.codex/skills/experiment-audit/SKILL.md')] + [
    Path('C:/Users/gb/.codex/skills/shared-references') / name for name in
    ['local-codex-policy.md', 'reviewer-independence.md', 'experiment-integrity.md', 'review-tracing.md']]
instruction_hashes = {str(path): 'sha256:' + hashlib.sha256(path.read_bytes()).hexdigest() for path in instruction_paths}
artifacts = [out / 'EXPERIMENT_AUDIT.md', out / 'EXPERIMENT_AUDIT.json', out / 'DETERMINISTIC_CHECKS.json',
             trace / 'request.txt', trace / '001-native-source-review.request.json',
             trace / '001-native-source-review.response.md', trace / '001-native-source-review.meta.json',
             trace / 'run.meta.json', trace / 'review_trace.events.jsonl',
             trace / 'reviewer_static_checks.py', trace / 'write_reviewer_report.py']
seal = dict(seal_type='SHA256_SOURCE_REVIEW_SNAPSHOT_NOT_SIGNED', generated_at=now,
    verdict='WARN', blocking_issue_count=0, execution_scope=scope,
    inputs_unchanged_on_final_read=True, audited_input_count=len(hashes),
    audited_input_hashes=hashes, instruction_input_hashes=instruction_hashes,
    artifact_hashes={str(path): 'sha256:' + hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts},
    result_approval=False, gpu_admission=False, **identity)
(out / 'SEAL.json').write_text(json.dumps(seal, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(verdict=report['verdict'], blocking_issue_count=0,
    audited_input_count=len(hashes), audit_path=str(out / 'EXPERIMENT_AUDIT.json'),
    seal_path=str(out / 'SEAL.json'), seal_sha256=hashlib.sha256((out / 'SEAL.json').read_bytes()).hexdigest())))
