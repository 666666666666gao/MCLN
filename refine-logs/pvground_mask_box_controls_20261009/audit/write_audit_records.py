import datetime
import hashlib
import json
from pathlib import Path

A = Path(__file__).absolute().parent
D = A.parent
T = D.parent
O = T / 'pvground_mask_reference_20261006'
C = T / 'pvground_compressed_geometry_support_20261008'
agent = '/root/pvg_fixed_geometry_actual_audit_20261009'
now = datetime.datetime.now().astimezone().isoformat()


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p):
    return json.loads(p.read_bytes())


def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


hashes = load(A / 'REVIEWED_FILE_HASHES.json')
for name, item in hashes.items():
    p = Path(name)
    assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], name
extra_paths = [
    Path(r'C:\Users\gb\.codex\skills\experiment-audit\SKILL.md'),
    *[Path(r'C:\Users\gb\.codex\skills\shared-references') / f for f in (
        'local-codex-policy.md', 'reviewer-independence.md', 'experiment-integrity.md', 'review-tracing.md')],
    T/'pvground_runtime_bundle_20260908_v1/PV-Ground/src/visual_data_handlers.py']
for p in extra_paths:
    hashes[str(p)] = dict(bytes=p.stat().st_size, sha256=sha(p))
write(A / 'REVIEWED_FILE_HASHES.json', hashes)
write(A / 'FINAL_INPUT_RECHECK.json', dict(status='PASS', time_cst=now,
    unchanged_previously_reviewed_inputs=len(hashes)-len(extra_paths),
    original_array_hash_check='DETERMINISTIC_VERIFY.json / VERIFIED_ARRAY_HASHES.json; no unnecessary second array recount'))
det = load(A / 'DETERMINISTIC_VERIFY.json')
sup = load(A / 'SUPPLEMENTARY_VERIFY.json')
errors = load(A / 'INVOCATION_ERRORS.json')
errors['errors'][0]['resolution'] = 'Parent corrected the request to fused_mask_reference_spec.json; successful read chunk e04a27. Request-path correction, not scientific evidence.'
errors['errors'].extend([
    dict(tool='exec_command', chunk_id='379d77', exit_code=1,
         status='AUDITOR_PATH_ALIAS_ASSERTION_FAILURE', raw_log='verify_001.log',
         invocation='verify_001.invocation.json', diagnostic='verify_001.diagnostic.json',
         source='independent_verify_attempt001.py',
         resolution='Preserve specified absolute path instead of resolving the Windows directory junction; source files unchanged.'),
    dict(tool='exec_command/write_stdin', initial_chunk='99c28a', session_id=21138,
         completion_chunk='8279bc', exit_code=1, status='AUDITOR_FLOAT32_RANGE_ASSERTION_FAILURE',
         raw_log='verify_002.log', invocation='verify_002.invocation.json', completion='verify_002.completion.json',
         diagnostic='verify_002.diagnostic.json', source='independent_verify_attempt002.py',
         resolution='Observed float32 excursions above one are reported without clipping; threshold counts verified in completed attempt 003.')])
errors['nonerror_nonzero_discoveries'] = [dict(tool='exec_command', chunk_id='cd9aaf', exit_code=1,
    operation="rg --files in old complete_fit and current experiment directories with -g '*.pth'",
    reason='Normal no-match result; checkpoints are outside those two experiment directories, in the archive.')]
errors['scientific_failure_hidden'] = False
write(A / 'INVOCATION_ERRORS.json', errors)

checks = {
    'gt_provenance': dict(status='WARN', classification='real_gt',
        details='Dataset annotation-derived GT and exact loader/scan-handler source hashes verified; original raw scan/annotation bytes not rederived; stage receipt lacks complete raw-data/GPU-state attestation.',
        evidence=[f'{O}/run_geometry_fit.py:194-213', f'{O}/run_geometry_fit.py:312-338',
            f'{T}/pvground_g_p2_20261002/complete/source/joint_det_dataset.py:1086-1119',
            f'{T}/pv_ground_source_20260905/src/visual_data_handlers.py:129-178',
            f'{T}/pv_ground_source_20260905/src/visual_data_handlers.py:225-259']),
    'score_normalization': dict(status='PASS',
        details='Standard 3D IoU, strict thresholds and fixed N=9508 denominator. Zero requested threshold flips. 94 unmodified float32 Mask IoUs slightly exceed 1; maximum 1.0000050067901611.',
        evidence=[f'{D}/analyze_fixed_query_boxes.py:25-41', f'{D}/analyze_fixed_query_boxes.py:109-119',
            f'{A}/independent_verify.py:150-184', f'{D}/SELECTED_BOX_ROWS.jsonl:982']),
    'result_existence': dict(status='PASS',
        details='9508 rows, 141 scenes, 2068 scene-target pairs, 1189 original NPZ files; all sizes/SHA256s match new manifest and original intake; all tables and transitions recounted.',
        evidence=[f'{D}/SUMMARY.json:6-128', f'{O}/complete_fit/fused_mask_reference/initial_formal/receipt.json:3-19',
            f'{O}/complete_fit/fused_mask_reference/initial_formal.log:25', f'{A}/independent_verify.py:65-184']),
    'dead_code': dict(status='PASS',
        details='Actual independent CPU verification exits 0. Original evaluation call path, fixed score winner, three geometry conditions, no coefficient search or GT-conditioned selection, and source invalid handling checked.',
        evidence=[f'{D}/analyze_fixed_query_boxes.py:61-105', f'{D}/analyze_fixed_query_boxes.py:129-154',
            f'{O}/run_geometry_fit.py:312-356', f'{O}/mask_reference.py:10-28',
            f'{T}/pvground_geometry_readback_20261004/revision2/readback_preflight_checks.py:19-68']),
    'scope': dict(status='WARN',
        details='One historical seed2027 snapshot, selected-query CPU controls, plus separately qualified cross-forward hypothesis; neither blend meets both gates. No current same-forward validation, promotion, full EG reproduction, three-innovation proof or final goal completion.',
        evidence=[f'{D}/SUMMARY.json:135-143', f'{D}/RETAINED_CROSS_FORWARD_HYPOTHESIS.json:12-13',
            f'{D}/RETAINED_CROSS_FORWARD_HYPOTHESIS.json:75-84', f'{D}/FINDINGS.md:7-40',
            f'{T}/pvground_novelty_20261009/EG-3DVG.txt:208-247']),
    'eval_type': dict(status='PASS', primary='real_gt', supplementary='real_gt_cross_forward_geometry_hypothesis',
        details='Predicted Mask boxes are predictions, not proxy ground truth. Cross-forward construction limits model-performance interpretation without changing GT classification.',
        evidence=[f'{O}/run_geometry_fit.py:320-338', f'{D}/compare_retained_with_archived_prior.py:30-42'])
}
claims = [
    dict(id='C1', claim='Closed October 6 fixed-native-winner geometry counts and transitions', impact='supported_with_qualifier',
         qualifier='One archived native PV data contract / one seed2027 snapshot; CPU verification over cached predictions.'),
    dict(id='C2', claim='Fixed half blend satisfies both numerical thresholds', impact='unsupported',
         reason='5671/4839; strict count 11 short of 4850.'),
    dict(id='C3', claim='Cross-forward hybrid arithmetic 5676/4840', impact='supported_as_hypothesis_only',
         qualifier='All9508 input/GT/query/uncorrected-mask alignment exact, but current same-forward native regression boxes absent.'),
    dict(id='C4', claim='Current complete model blend performance or weight promotion', impact='unsupported',
         reason='Cross-forward predictions cannot substitute for the missing current native boxes; strict count also 10 short.'),
    dict(id='C5', claim='Full EG-3DVG reproduction or a novel half-blend contribution', impact='unsupported',
         reason='Only the named prior geometry equation is applied to PV outputs.'),
    dict(id='C6', claim='Three effective innovations and full research goal completion', impact='unsupported',
         reason='No required module/dataset executions established by this CPU diagnostic.'),
    dict(id='C7', claim='3074 zero-root-overlap is a box statistic', impact='supported',
         qualifier='Recount confirms 3074 zero box intersections; source saved scalar Mask IoUs are zero, but raw Mask membership was not freshly recomputed.')]

arrays = load(A / 'VERIFIED_ARRAY_HASHES.json')
audit = dict(audit_skill='experiment-audit', verdict='WARN', overall_verdict='WARN', integrity_status='warn',
    reason_code='cached_arithmetic_pass_with_provenance_and_scope_limits',
    summary='All reported cached box counts and transitions independently verify; accept narrow controls provisionally, retain source-provenance, float32-range and cross-forward claim limits.',
    fresh_context=True, requested_model='gpt-6-astra', requested_reasoning_effort='max', requested_backend='codex',
    actual_backend='UNATTESTED', actual_model='UNATTESTED', actual_reasoning_effort='UNATTESTED',
    reviewer_model='UNATTESTED', reviewer_reasoning='UNATTESTED', reviewer_family='openai',
    executor_model='UNATTESTED', executor_family='openai', review_independence='same-family', acceptance_status='provisional',
    agent_id=agent, verdict_id=agent+'@2026-10-09', generated_at=now, date='2026-10-09',
    checks=checks, claims=claims, blocking_findings=[], execution_blockers=[],
    claim_blockers=[
        dict(id='CURRENT_SAME_FORWARD_OUTPUT_ABSENT', blocks='Current complete-model performance/promotion claim',
             evidence=f'{D}/RETAINED_CROSS_FORWARD_HYPOTHESIS.json:12-13'),
        dict(id='JOINT_GATE_NOT_MET', blocks='Joint threshold completion claim', values=dict(primary_blend=[5671,4839], hybrid=[5676,4840], minimum=[5658,4850])),
        dict(id='MISSING_FULL_GOAL_EXECUTIONS', blocks='Three effective innovations and Nr3D/Sr3D full-goal completion claim',
             evidence=f'{D}/FINDINGS.md:40')],
    deterministic_verification_status='PASS', deterministic_evidence_acceptance='accepted_for_checked_arithmetic_only',
    deterministic_verifier=dict(path=str(A/'independent_verify.py'), sha256=sha(A/'independent_verify.py'),
        result_path=str(A/'DETERMINISTIC_VERIFY.json'), result_sha256=sha(A/'DETERMINISTIC_VERIFY.json'),
        actual_process_exit_code=0, initial_chunk='49c4c7', session_id=53451, completion_chunk='4fa41d'),
    supplementary_verifier=dict(path=str(A/'verify_claims_and_provenance.py'), result_path=str(A/'SUPPLEMENTARY_VERIFY.json'),
        result_sha256=sha(A/'SUPPLEMENTARY_VERIFY.json'), actual_process_exit_code=0, chunk_id='de1cf4'),
    original_controls=det['table'], original_comparisons=det['comparisons'], cross_forward=det['cross_forward'],
    supplementary_provenance=sup,
    warnings=[
        'Original raw scan/annotation box provenance not independently rederived; native sampled-member GT contract retained.',
        'Source load/import files can be overwritten by later stages; source receipt binds rows and intake separately binds arrays, not complete stage-specific GPU state.',
        '94 pure-Mask float32 IoUs exceed one by at most 0.0000050067901611; no accuracy-threshold effect, no clipping applied.',
        'Cross-forward hybrid remains a hypothesis; same-forward current native regression boxes are unsaved.'],
    audited_input_hashes={name: 'sha256:'+item['sha256'] for name,item in hashes.items()},
    audited_array_hashes={item['path']: 'sha256:'+item['sha256'] for item in arrays},
    reviewed_file_hash_manifest=str(A/'REVIEWED_FILE_HASHES.json'),
    raw_report_path=str(A/'RAW_REVIEW_REPORT.md'), raw_report_sha256=sha(A/'RAW_REVIEW_REPORT.md'),
    actual_invocation_errors_path=str(A/'INVOCATION_ERRORS.json'),
    trace_path=str(A/'.aris/traces/experiment-audit/2026-10-09_run01'),
    no_missing_execution_replaced_by_verdict=True, same_forward_current_model_validated=False,
    weight_promotion_eligible=False, three_effective_contributions=False, full_goal_complete=False,
    ssh_queries=0, gpu_or_neural_forwards=0, optimizer_updates=0, packages_installed=0, active_sources_changed=False)
audit['cross_forward'].pop('checkpoint_identity_not_freshly_loaded_or_hashed')
audit['cross_forward'].update(main_numeric_verifier_did_not_inspect_checkpoint=True,
    checkpoint_archive_sha256_verified_by_supplementary_verifier=True, checkpoint_deserialized=False)
write(A/'EXPERIMENT_AUDIT.json', audit)

md = '''# Experiment Audit Report

Date: 2026-10-09. Overall verdict: **WARN**. **Independent deterministic verification: PASS.**

Fresh context; requested gpt-6-astra / max. Actual backend, model and reasoning effort: UNATTESTED. Same-family review; acceptance provisional. No blocking finding prevents accepting the narrowly stated cached CPU geometry comparison.

| Check | Status | Finding |
|---|---|---|
| A. GT provenance | WARN | Dataset-derived GT and exact original loader/scan-handler hashes verified; original raw scan/annotation bytes were not rederived. |
| B. Normalization | PASS | Standard IoU, strict thresholds and fixed denominator 9508; no output-dependent metric normalization. |
| C. Result existence | PASS | 9508 rows, 141 scenes, 2068 targets by scene, 1189 shards; all counts, transitions and hashes match. |
| D. Executed code | PASS | Independent offline CPU checker actually exited 0; fixed native-score Query, coefficient 0.5 and existing invalid handling verified. |
| E. Scope | WARN | One October 6 snapshot plus a separately qualified cross-forward hypothesis; neither blend satisfies both goals. |
| F. Evaluation type | PASS | real_gt under the archived native data contract; cross-forward predictions remain a hypothesis. |

| October 6 same-source condition | Hits @0.25 / @0.5 | Accuracy @0.25 / @0.5 |
|---|---:|---:|
| Native regression | 5615 / 4495 | 59.0555% / 47.2760% |
| Pure Mask reference | 5598 / 4848 | 58.8767% / 50.9886% |
| Fixed half blend | 5671 / 4839 | 59.6445% / 50.8940% |

Blend versus Mask: repairs/damages 121/48 at 0.25, 207/216 at 0.5, net +73/-9. All 28,524 selected IoUs per precision were reconstructed. Float32/float64 and saved GPU/CPU threshold flips are zero. There are 39 selected invalid references and zero native-score winner ties.

The existing float32 formula yields 94 Mask IoUs slightly above 1, maximum 1.0000050067901611. Float64 is at most 1.0; accuracy counts are unaffected. Values were retained without clipping. This is a numerical warning, not score normalization.

The separate old-regression/current-Mask hybrid verifies as **5676/4840**, net +77/-19 versus saved retained 5599/4859. All9508 input-hash/GT/query/uncorrected-Mask identities align. Old/current core dependency records match, and the retained 447109-byte checkpoint archive hashes to 6f9109cb690399880b8dc3bcf6e4b98d1053145732ee2a6b504bfaecbd1eaac2. Current same-forward native boxes are absent, so this is not new current-model performance or promotion evidence.

GT evidence: old run_geometry_fit.py:194-213,312-338; hash-matching joint_det_dataset.py:1086-1119,1387-1392; original visual_data_handlers.py:129-178,225-259. Query and execution evidence: old run_geometry_fit.py:312-356,365-378; native_root_bbs.py:4-9; readback_preflight_checks.py:19-68. Control evidence: analyze_fixed_query_boxes.py:61-119,137-146; independent_verify.py:65-245. Novelty/scope evidence: EG-3DVG.txt:208-247; FINDINGS.md:7-40. Exact absolute path definitions and full per-check references are in RAW_REVIEW_REPORT.md and the JSON report.

The original loader uses annotated sampled object members; full-resolution GT equivalence and complete original GPU state were not newly attested. The later load receipt is not a separate initial-stage state seal. No new GPU replay is needed to validate the complete cached arithmetic.

`blocking_findings=[]` for the closed recount. Claim blockers remain: unsaved current same-forward regression boxes; neither blend meets 5658/4850; no three-effective-module or Nr3D/Sr3D execution evidence supplied by this diagnostic. It is only an EG-3DVG geometry-formula control, not full model reproduction or a novel contribution. No verdict replaces missing experiments.

The missing request path, Windows junction assertion failure, float32-range assertion failure, and normal no-checkpoint search result are preserved in INVOCATION_ERRORS.json with raw logs and invocation receipts. Successful numerical execution: verify_003.log / completion chunk 4fa41d. Supplementary narrative/provenance checks: verify_004.log / chunk de1cf4. The 3074 zero-overlap count is a box statistic; raw Mask membership was not rederived.

Full report: RAW_REVIEW_REPORT.md. Machine-readable judgment: EXPERIMENT_AUDIT.json. Complete SHA256 evidence: REVIEWED_FILE_HASHES.json and VERIFIED_ARRAY_HASHES.json. Deterministic receipts: DETERMINISTIC_VERIFY.json and SUPPLEMENTARY_VERIFY.json. All scientific inputs and active training/source state remain unchanged.
'''
(A/'EXPERIMENT_AUDIT.md').write_text(md, encoding='utf-8')

trace = A/'.aris/traces/experiment-audit/2026-10-09_run01'
trace.mkdir(parents=True, exist_ok=True)
metadata = dict(skill='experiment-audit', run_id='2026-10-09_run01', generated_at=now,
    started_at=None, start_time_tool_attested=False, agent_id=agent,
    requested_model='gpt-6-astra', requested_reasoning_effort='max', actual_backend='UNATTESTED',
    actual_model='UNATTESTED', actual_reasoning_effort='UNATTESTED', fresh_context=True,
    review_independence='same-family', acceptance_status='provisional', project_dir=str(D))
write(trace/'run.meta.json', metadata)
write(trace/'001-actual-audit.request.json', dict(call_number=1, purpose='fresh-actual-experiment-audit',
    timestamp=None, tool='native parent collaboration assignment', model='gpt-6-astra', reasoning_effort='max',
    invocation_parameters_not_independently_attested=True, prompt=(D/'AUDIT_REQUEST.txt').read_text(encoding='utf-8'),
    initial_request_path_correction='Original request named absent old pair_spec.json; parent corrected it to fused_mask_reference_spec.json.',
    additional_parent_requests=[
        'Inspect compare_retained_with_archived_prior.py and RETAINED_CROSS_FORWARD_HYPOTHESIS.json, current complete_fit/formal rows/receipt and current pair_spec.json; verify all9508 alignment and core dependency hashes; do not promote to same-forward current-model validation.',
        'Include FINDINGS.md number/scope claims; 3074 zero_root_overlap denotes BOX overlap, not proof of zero Mask intersection.'],
    files_referenced=list(hashes)))
(trace/'001-actual-audit.response.md').write_bytes((A/'RAW_REVIEW_REPORT.md').read_bytes())
write(trace/'001-actual-audit.meta.json', dict(metadata, call_number=1, purpose='fresh-actual-experiment-audit',
    status='completed', verdict='WARN', deterministic_verification='PASS',
    raw_response_sha256=sha(A/'RAW_REVIEW_REPORT.md'), actual_invocation_errors=str(A/'INVOCATION_ERRORS.json')))
events = A/'.aris/meta/events.jsonl'
events.parent.mkdir(parents=True, exist_ok=True)
with events.open('a', encoding='utf-8') as stream:
    stream.write(json.dumps(dict(event='review_trace', skill='experiment-audit', purpose='fresh-actual-experiment-audit',
        agent_id=agent, trace_path=str(trace), status='completed', generated_at=now))+'\n')
print(json.dumps(dict(status='WRITTEN', verdict='WARN', deterministic='PASS', blocking_findings=[],
    raw_report=str(A/'RAW_REVIEW_REPORT.md'), markdown=str(A/'EXPERIMENT_AUDIT.md'),
    json=str(A/'EXPERIMENT_AUDIT.json'), source_hashes=len(hashes), arrays=len(arrays))))
