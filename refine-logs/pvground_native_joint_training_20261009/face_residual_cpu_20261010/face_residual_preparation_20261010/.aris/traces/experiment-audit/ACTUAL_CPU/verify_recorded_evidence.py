"""Read saved CPU evidence only; never import or execute an audited module."""
import ast
import base64
import datetime
import hashlib
import json
import math
from pathlib import Path

root = Path(r'C:\Users\gb\.codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\face_residual_preparation_20261010')
trace = root/'.aris/traces/experiment-audit/ACTUAL_CPU'
seen = {}


def raw(path):
    path = Path(path)
    data = path.read_bytes()
    seen[str(path)] = hashlib.sha256(data).hexdigest()
    return data


def obj(path):
    return json.loads(raw(path))


def digest(path):
    raw(path)
    return seen[str(Path(path))]


def remote_code(path):
    tree = ast.parse(raw(path).decode('utf-8-sig'))
    return next(node.value.value for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'code' for t in node.targets))


attempt = root/'cpu_execution_attempt2'
receipt = obj(attempt/'RAW_STDOUT.json')
execution = obj(attempt/'CPU_EXECUTION.json')
result = obj(attempt/'CPU_MODULE_RESULT.json')
transport = obj(attempt/'TRANSPORT_EXIT.json')
stdout = raw(attempt/'CPU_STDOUT.txt')
stderr = raw(attempt/'CPU_STDERR.txt')
transport_stderr = raw(attempt/'RAW_STDERR.txt')
r2_path = root/'source_review/EXPERIMENT_CODE_REVIEW_R2.json'
r2 = obj(r2_path)
canonical = root/'source_review/EXPERIMENT_CODE_REVIEW.json'
r1 = obj(root/'transport_attempt1/EXPERIMENT_CODE_REVIEW_R1.json')
launcher = root/'run_face_residual_cpu_authorized.py'
old_launcher = root/'transport_attempt1/REVIEWED_LAUNCHER_ATTEMPT1.py'
port = obj(root.parent/'NATIVE_SOURCE_PORT.json')
spec = obj(root/'source_conditioned_init.json')
static = obj(root/'STATIC_CPU_ROOT_READ.json')
static_exit = obj(root/'STATIC_CPU_ROOT_TRANSPORT_EXIT.json')
finalization = obj(root/'.aris/traces/experiment-bridge/2026-10-10_run02/REPORT_FINALIZATION_CHECK.json')
r2_input_matches = {name: digest(name) == expected.removeprefix('sha256:')
                    for name, expected in r2['audited_input_hashes'].items()}
r1_launcher_digest = r1['audited_input_hashes'][str(launcher)].removeprefix('sha256:')
unchanged_names = ('CPU_CHECK_PLAN.md', 'face_residual_span_mixer.py',
                   'check_face_residual_modules_cpu.py', 'source_conditioned_init.json')
r1_unchanged = {name: digest(root/name) == r1['audited_input_hashes'][str(root/name)].removeprefix('sha256:')
                for name in unchanged_names}
archived_failure_matches = {name: raw(root/'cpu_execution'/name) == raw(root/'transport_attempt1'/name)
                            for name in ('RAW_STDOUT.json', 'RAW_STDERR.txt', 'TRANSPORT_EXIT.json')}
first_stderr = raw(root/'cpu_execution/RAW_STDERR.txt')
first_stdout = raw(root/'cpu_execution/RAW_STDOUT.json')
first_exit = obj(root/'cpu_execution/TRANSPORT_EXIT.json')
r1_reports_match = {ext: raw(root/f'transport_attempt1/EXPERIMENT_CODE_REVIEW_R1.{ext}') ==
                   raw(root/f'source_review/EXPERIMENT_CODE_REVIEW_20261010_074450.{ext}')
                   for ext in ('md', 'json')}
r2_reports_match = {ext: raw(root/f'source_review/EXPERIMENT_CODE_REVIEW_R2.{ext}') ==
                   raw(root/f'source_review/EXPERIMENT_CODE_REVIEW.{ext}') ==
                   raw(root/f'source_review/EXPERIMENT_CODE_REVIEW_20261010_075336.{ext}')
                   for ext in ('md', 'json')}
source_matches = {name: digest(root.parent/'source'/name) == port['files'][name]['sha256']
                  for name in ('mask_support_corrector.py', 'extremal_span_mixer.py', 'native_mask_geometry.py')}
whole_mask = Path(r'C:\Users\gb\.codex\tmp\pvground_whole_mask_range_20261003\whole_mask_range.py')
whole_match = digest(whole_mask) == port['files']['whole_mask_range.py']['sha256']
env = obj(Path(r'C:\Users\gb\.codex\tmp\pvground_cs_restart_20261002\env_spec.json'))
env_canonical = hashlib.sha256(json.dumps(env, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
trace_r2 = root/'.aris/traces/experiment-bridge/2026-10-10_run02'
for name in ('run.meta.json', '001-transport-rescue.request.json', '001-transport-rescue.meta.json',
             '001-transport-rescue.response.md', 'STATIC_REVIEW_CHECKS_R2.json', 'LAUNCHER_DELTA.diff',
             '010-review-interpreter.tool.json', '011-static-review-checks.tool.json',
             '012-report-finalization.tool.json', 'review_static_inputs.py'):
    raw(trace_r2/name)
raw(root/'ACTUAL_CPU_AUDIT_REQUEST.md')
for relative in ('experiment-audit/SKILL.md', 'shared-references/local-codex-policy.md',
                 'shared-references/reviewer-independence.md', 'shared-references/experiment-integrity.md',
                 'shared-references/review-tracing.md'):
    raw(Path(r'C:\Users\gb\.codex\skills')/relative)

new_parameters = (332*64+64) + (64*32+32) + (32*1+1)
prior_parameters = 2*(288*32+32) + (41*32+32) + (32*32+32) + (105*64+64) + (64*32+32) + (32+1)
steps = result['synthetic_gradient_steps']
start = datetime.datetime.fromisoformat(receipt['started_cst'])
finish = datetime.datetime.fromisoformat(receipt['finished_cst'])
expected_execution = {k: v for k, v in receipt.items() if not k.endswith('_base64')}
expected_execution['source_review_sha256'] = digest(canonical)
checks = {
    'all_R2_input_digests_match_current_files': all(r2_input_matches.values()),
    'R1_relevant_unchanged_inputs_match': all(r1_unchanged.values()),
    'R1_historical_launcher_digest_matches_archived_launcher': digest(old_launcher) == r1_launcher_digest,
    'current_launcher_is_distinct_from_R1_launcher': digest(launcher) != r1_launcher_digest,
    'embedded_remote_program_unchanged_between_R1_and_R2': remote_code(old_launcher) == remote_code(launcher),
    'R1_archived_reports_match_timestamped_originals': all(r1_reports_match.values()),
    'R2_canonical_and_timestamped_reports_match': all(r2_reports_match.values()),
    'R2_full_trace_response_matches_report': raw(trace_r2/'001-transport-rescue.response.md') == raw(root/'source_review/EXPERIMENT_CODE_REVIEW_R2.md'),
    'R2_verdict_PASS_zero_blocking': r2['verdict'] == 'PASS' and r2['blocking_issue_count'] == 0,
    'R2_receipt_digest_matches_canonical_and_R2_finalization': execution['source_review_sha256'] == digest(canonical) == finalization['canonical_json_sha256'],
    'R2_finalization_precedes_CPU_child': datetime.datetime.fromisoformat(finalization['time_cst']) < start,
    'failed_first_transport_archives_are_byte_identical': all(archived_failure_matches.values()),
    'first_transport_exit255_and_empty_stdout': first_exit['exit_code'] == 255 and len(first_stdout) == 0,
    'first_transport_133_byte_private_stderr_has_host_key_error': len(first_stderr) == 133 and b'host key verification failed' in first_stderr.lower(),
    'static_root_observation_exit0_target_and_receipt_absent': static_exit['exit_code'] == 0 and static['root'] == '/root/autodl-tmp/pvground_face_residual_cpu_20261010' and static['exists'] is False and static['CPU_execution_receipt_exists'] is False,
    'static_observation_before_R2_and_child': datetime.datetime.fromisoformat(static['time_cst']) < datetime.datetime.fromisoformat(r2['time_cst']) < start,
    'A_B_geometry_local_digests_match_port': all(source_matches.values()),
    'whole_mask_range_local_digest_matches_port': whole_match,
    'env_canonical_digest_matches_port_spec_and_receipt': env_canonical == port['env_spec_sha256'] == spec['env_spec_sha256'] == receipt['env_spec_sha256'],
    'child_source_matches_port': receipt['source'] == port['model_source'],
    'transport_and_child_exit0': transport['exit_code'] == receipt['exit_code'] == execution['exit_code'] == 0,
    'child_finish_follows_start': finish > start,
    'raw_base64_stdout_matches_saved_CPU_STDOUT': base64.b64decode(receipt['stdout_base64'], validate=True) == stdout,
    'raw_base64_stderr_matches_saved_CPU_STDERR': base64.b64decode(receipt['stderr_base64'], validate=True) == stderr,
    'raw_base64_result_matches_saved_CPU_MODULE_RESULT': base64.b64decode(receipt['result_base64'], validate=True) == raw(attempt/'CPU_MODULE_RESULT.json'),
    'saved_execution_equals_raw_metadata_plus_review_digest': execution == expected_execution,
    'child_stdout_JSON_equals_result_JSON': json.loads(stdout) == result,
    'attempt2_transport_and_child_stderr_are_empty': len(transport_stderr) == len(stderr) == 0,
    'actual_status_and_torch_version': result['status'] == 'ACTUAL_FACE_RESIDUAL_SYNTHETIC_CPU_MODULE_PASS' and result['torch_version'] == '1.10.2+cu111',
    'fixture_metadata_matches_source': result['fixture'] == dict(seed=2027, batch=1, queries=256, superpoints=8, points=50000),
    'parameters_match_independent_integer_calculation': (prior_parameters, new_parameters, prior_parameters+new_parameters) == (result['prior_parameters'], result['new_face_parameters'], result['combined_parameters']) == (29793, 23425, 53218),
    'module_state_counts_match_7_plus_3_linear_layers': (result['combined_module_states'], result['new_face_module_states']) == ((7+3)*2, 3*2) == (20, 6),
    'zero_equivalence_and_hidden_column_assertions_reported_pass': result['zero_output_boxes_bitwise_equal_loaded_prior_both_modes'] is True and result['hidden_source_token_and_fraction_columns_zero'] is True,
    'gradient_steps_finite_and_expected_output_encoder_pattern': len(steps) == 2 and [s['step'] for s in steps] == [0, 1] and all(math.isfinite(s[k]) for s in steps for k in ('output_gradient_norm', 'encoder_gradient_norm', 'loss')) and all(s['output_gradient_norm'] > 0 for s in steps) and steps[0]['encoder_gradient_norm'] == 0 and steps[1]['encoder_gradient_norm'] > 0,
    'supplied_ordering_and_module_roundtrip_reported_pass': result['supplied_inverted_face_decoding_checked'] is True and result['module_state_in_memory_roundtrip_bitwise_equal'] is True,
    'runtime_report_stays_within_CPU_synthetic_scope': result['CUDA_initialized'] is False and all(result[k] == 0 for k in ('GPU_calls', 'real_loader_rows', 'current_training_queries', 'new_weight_files')) and all(result[k] is False for k in ('full_PV_factory_checked', 'native_criterion_checked', 'full_optimizer_recovery_checked', 'learned_inverted_face_prediction_proven', 'GPU_training_admission', 'full_goal_complete')) and result['formal_accuracy'] is None,
}
output = dict(
    recorded_at_cst=datetime.datetime.now().astimezone().isoformat(),
    verifier_scope='Read-only local evidence/hash/base64/AST/integer checks. No candidate imports or execution, Torch imports, SSH, GPU, or training queries.',
    checks=checks, check_count=len(checks), all_checks_pass=all(checks.values()),
    R2_input_hash_matches=r2_input_matches, R1_unchanged_input_hash_matches=r1_unchanged,
    R1_archive_matches=archived_failure_matches, R1_reports_match=r1_reports_match, R2_reports_match=r2_reports_match,
    source_port_hash_matches=source_matches, expected_checkpoint_bindings={k: spec[k] for k in ('support_checkpoint', 'support_checkpoint_sha256', 'span_checkpoint', 'span_checkpoint_sha256')},
    checkpoint_evidence_limit='Bindings above are read from the unchanged spec; completed reviewed child checks their hashes and strict loads. Reviewer did not reopen remote checkpoint bytes.',
    whole_mask_range_evidence_limit='Local helper matches port; the child directly rehashes three A/B/geometry files, not this transitive helper, and does not report its runtime __file__.',
    execution=execution, elapsed_child_seconds=(finish-start).total_seconds(), result=result,
    first_transport=dict(exit_code=first_exit['exit_code'], stdout_bytes=len(first_stdout), stderr_bytes=len(first_stderr), error_kind='HOST_KEY_VERIFICATION_FAILED', raw_stderr_copied_or_published=False),
    attempt2_stderr_bytes=dict(transport=len(transport_stderr), child=len(stderr)),
    static_observation=static, requested_model='gpt-6-astra', requested_reasoning_effort='max',
    actual_model='UNATTESTED', actual_reasoning_effort='UNATTESTED', review_independence='same-family', acceptance_status='provisional',
    audited_input_hashes=dict(sorted(seen.items())),
)
(trace/'DETERMINISTIC_CHECKS.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(check_count=len(checks), all_checks_pass=all(checks.values()), failed_checks=[k for k,v in checks.items() if not v], input_count=len(seen), elapsed_child_seconds=(finish-start).total_seconds(), source_review_sha256=execution['source_review_sha256'], result_sha256=seen[str(attempt/'CPU_MODULE_RESULT.json')], raw_transport_stdout_sha256=seen[str(attempt/'RAW_STDOUT.json')]), ensure_ascii=False))
