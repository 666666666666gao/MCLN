"""Assemble R1 records from completed receipts and immutable R1 snapshots only."""
import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TMP = ROOT.parent
AGENT = '/root/pvg_extremal_span_source_20261009'
NOW = datetime.datetime.now().astimezone().isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, content):
    (HERE / name).write_text(json.dumps(content, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


hashes = json.loads((HERE / 'INPUT_HASHES.json').read_bytes())
official_module = HERE / 'torch_v1_10_2_module.py.txt'
hashes[str(official_module)] = dict(sha256=sha(official_module), bytes=official_module.stat().st_size)
write('INPUT_HASHES.json', hashes)
snapshots = json.loads((HERE / 'R1_SOURCE_SNAPSHOTS.json').read_text(encoding='utf-8-sig'))
for item in snapshots:
    assert sha(Path(item['snapshot'])) == item['sha256'] == hashes[item['source']]['sha256']
assert sha(HERE / 'SOURCE_AUDIT_REQUEST.exact.txt') == hashes[str(ROOT / 'SOURCE_AUDIT_REQUEST.txt')]['sha256']
verification = json.loads((HERE / 'DIAGNOSTIC_VERIFICATION.json').read_bytes())
failed = json.loads((HERE / 'verify_001.invocation.json').read_bytes())
passed = json.loads((HERE / 'verify_002.invocation.json').read_bytes())
assert failed['result']['exit_code'] == 1 and passed['result']['exit_code'] == 0
assert verification['neural_forwards'] == verification['gpu_forwards'] == verification['optimizer_updates'] == 0
raw = (HERE / 'SOURCE_REVIEW.md').read_bytes()
(HERE / 'RAW_RESPONSE.md').write_bytes(raw)
(HERE / '001-source-audit.response.md').write_bytes(raw)
metadata = dict(audit_skill='experiment-audit', review_round='R1', date='2026-10-09',
    generated_at=NOW, fresh_context=True, agent_id=AGENT, verdict_id=AGENT + '@R1',
    requested_backend='codex', requested_model='gpt-6-astra', requested_reasoning_effort='max',
    actual_backend='UNATTESTED', actual_model='UNATTESTED', actual_reasoning_effort='UNATTESTED',
    reviewer_model='UNATTESTED', reviewer_reasoning='UNATTESTED', reviewer_family='openai',
    executor_model='UNATTESTED', executor_family='openai',
    review_independence='same-family', acceptance_status='provisional')

blockers = [
    dict(id='B1_NATIVE_DICTIONARY_FORWARD_CONTRACT', severity='blocking',
         scope='STATIC_ONLY neural source',
         finding='Wrapper forward(self, **inputs) rejects the native model(dict(inputs)) call.',
         evidence=[str(ROOT / 'span_refinement_model.py') + ':41-44',
                   str(TMP / 'pvground_mask_support_correction_20261008_v2/complete_fit/PV-Ground/models/pv_ground.py') + ':355',
                   str(TMP / 'pvground_geometry_readback_20261004/revision2/readback_preflight_checks.py') + ':42'],
         snapshot=str(HERE / 'r1_source_snapshots/span_refinement_model.py'),
         minimal_fix='Use forward(self, inputs) and exactly one self.parent(inputs) call inside the existing no_grad scope.',
         deterministic_witness='Python inspect.Signature.bind rejected the positional dictionary; no model was invoked.'),
    dict(id='B2_SOURCE_MODE_RESTORATION_IDENTITY', severity='blocking',
         scope='STATIC_ONLY neural checkpoint contract',
         finding='source_mode is a plain attribute absent from the identical 14-tensor state and from any implemented checkpoint/restore payload.',
         evidence=[str(ROOT / 'extremal_span_mixer.py') + ':17-27,69-72',
                   str(ROOT / 'span_refinement_model.py') + ':36-38',
                   str(ROOT / 'METHOD_PROPOSAL.md') + ':46',
                   str(official_module) + ':1249-1270'],
         snapshot=str(HERE / 'r1_source_snapshots/extremal_span_mixer.py'),
         minimal_fix='Explicitly save source_mode with the head state_dict; restore using saved mode and validate expected arm, preserving 14 trainable tensors.',
         limitation='Static finding only; no actual Torch load or corrupted historical checkpoint is claimed.')]

diagnostic_checks = {
    'A_gt_provenance': dict(status='WARN', classification='real_gt',
        details='GT-conditioned coefficients use cached root_box. Archived writer sources that field from validation center_label/size_gts. Raw original annotation/point bytes not re-derived.',
        evidence=['analyze_axis_span_diagnostic.py:34-45', 'analyze_fixed_query_boxes.py:69-70,95-102', 'run_geometry_fit.py:194-213,320-325,332-347']),
    'B_score_normalization': dict(status='PASS', details='Standard AABB IoU, strict > thresholds, fixed N=9508. No own-score normalization.'),
    'C_result_existence': dict(status='PASS', details='All result fields and bound input hashes match independent arithmetic and exact-script replay, excluding new timestamp.'),
    'D_dead_code': dict(status='PASS', details='Actual replay executes main and IoU; independent scalar projection, KKT and IoU verifier exits 0.'),
    'E_scope': dict(status='WARN', details='One historical snapshot; 9508 expressions, 141 scenes, 2068 scene-target pairs; no current neural metric or efficacy claim.'),
    'F_evaluation_type': dict(status='PASS', classification='real_gt_conditioned_offline_representational_diagnostic',
        deployment_valid=False, model_metric=False, iou_upper_bound=False, training_label_dataset=False)}
source_checks = {
    'A_gt_provenance': dict(status='PASS', limit='STATIC_ONLY', details='No inference GT/diagnostic input; proposed native training targets only. Future split/matching caller unverified.'),
    'B_score_normalization': dict(status='PASS', limit='STATIC_ONLY', details='No new ranking or score head; pooled evidence and axis gate are internal geometry operations.'),
    'C_result_existence': dict(status='PASS', limit='truthful prepared status only', details='Proposal explicitly says no runner, M0, new weights or neural accuracy; B1/B2 remain source blockers.'),
    'D_dead_code': dict(status='WARN', details='No span runner exists; new neural functions have no actual execution witness.'),
    'E_scope': dict(status='WARN', details='Capacity verified arithmetically; initialization, runtime gradients, frozen matching/state and save/restore remain future witnesses.'),
    'F_evaluation_type': dict(status='WARN', classification='STATIC_ONLY_SOURCE_PREPARATION', intended_future_gt='real_gt', actual_model_evaluation=False)}

report = dict(metadata,
    verdict='FAIL', overall_verdict='FAIL',
    reason_code='neural_source_contract_blockers_with_separately_verified_gt_conditioned_numpy_diagnostic',
    diagnostic_verdict='WARN', source_verdict='FAIL', source_scope='STATIC_ONLY',
    diagnostic=dict(verdict='WARN', integrity_status='warn', deterministic_status='PASS',
        deterministic_evidence_acceptance='accepted_for_checked_cached_arithmetic_only',
        checks=diagnostic_checks, results=verification['diagnostic'],
        rows=9508, scenes=141, scene_target_pairs=2068, independently_checked_axis_solutions=28524,
        iou_decreased_rows=1091, all_result_fields_match_except_timestamp=True,
        neural_metrics=False, no_model_promotion=True),
    source=dict(verdict='FAIL', scope='STATIC_ONLY', checks=source_checks,
        blocking_findings=blockers, static_parameters_per_arm=29793, static_trainable_state_tensors_per_arm=14,
        static_pair_trainable_parameters=59586, candidate_count=256,
        actual_state_dict_instantiated=False, actual_gradients_observed=False,
        clamp_derivative_source_version='v1.10.2', clamp_derivative_at_zero=1,
        real_M0='NOT_AVAILABLE', runner_present=False, launch_approved=False,
        capacity_match='PASS_STATIC', initial_hidden_state_identity='UNVERIFIED',
        checkpoint_mode_identity='MISSING', native_forward_call_contract='FAIL'),
    blocking_findings=blockers,
    nonblocking_findings=[
        dict(id='W1_INVALID_NOT_PROVEN_EMPTY', details='39 invalid extents include the absent/degenerate definition; do not call all39 physically empty. Zero pooled input can become a biased encoded token.', minimal_fix='Clarify wording only.'),
        dict(id='W2_RUNNER_CONTRACTS_PENDING', details='One shared parent cache, same frozen-parent GT assignments, identical copied initial tensors with independent storage, separate optimizers, fixed eval parent and full state invariance need actual implementation.'),
        dict(id='W3_REAL_TWO_STEP_M0_UNAVAILABLE', details='No source/NumPy check proves first-step output gradient or second-step source-encoder gradient. Defer real B8 M0 until active job closure and required handling.'),
        dict(id='W4_ACTIVE_RUNTIME_UNATTESTED', details='Reviewed historical descriptors name Torch1.10.2+cu111 but have hashes different from the supplied active pair env_spec hash; official derivative finding is conditional on v1.10.2.'),
        dict(id='W5_CACHED_GT_PROVENANCE_LIMIT', details='Selected cached GT rows read directly; raw original dataset bytes not freshly re-derived.')],
    audited_input_hashes=hashes, immutable_source_snapshots=snapshots,
    source_snapshot_scope='R1 only; parent permitted to patch live source after immutable snapshot confirmation; R1 does not reread or judge repairs',
    trace_path=str(HERE), raw_report_path=str(HERE / 'RAW_RESPONSE.md'), raw_report_sha256=sha(HERE / 'RAW_RESPONSE.md'),
    exact_request_path=str(HERE / 'SOURCE_AUDIT_REQUEST.exact.txt'),
    deterministic_verifier=dict(path=str(HERE / 'independent_verify.py'), sha256=sha(HERE / 'independent_verify.py'),
        result_path=str(HERE / 'DIAGNOSTIC_VERIFICATION.json'), result_sha256=sha(HERE / 'DIAGNOSTIC_VERIFICATION.json'),
        actual_exit_code=0, chunk_id=passed['result']['chunk_id'], invocation='verify_002.invocation.json'),
    actual_failed_invocations=[dict(invocation='verify_001.invocation.json', log='verify_001.log',
        source='independent_verify.failed001.py', source_sha256=sha(HERE / 'independent_verify.failed001.py'),
        exit_code=1, chunk_id=failed['result']['chunk_id'],
        reason='Reviewer used browser-rendered line indices for raw downloaded Torch source. Corrected to raw line614/618; not a neural failure.')],
    ssh_queries=0, neural_imports=0, gpu_or_neural_forwards=0, optimizer_updates=0, packages_installed=0,
    reviewed_experimental_sources_edited_by_reviewer=False, main_MEMORY_read=False, global_notes_modified=False,
    model_metric_reported=False, weight_promotion_eligible=False, three_effective_contributions=False, full_goal_complete=False)
write('SOURCE_REVIEW.json', report)

parent_request = 'Start a fresh review. Read and follow the exact request at C:\\Users\\gb\\.codex\\tmp\\pvground_extremal_span_evidence_20261009\\SOURCE_AUDIT_REQUEST.txt, including the experiment-audit skill and local execution policy it names. Read the listed files directly and independently verify; do not rely on an executor summary. Output only under its specified source_review folder. No SSH, GPU, model forward, training, credentials, main MEMORY.md, package installations, experimental code edits or global notes. Two separate verdicts: closed real-GT-conditioned NumPy diagnostic versus STATIC_ONLY proposed neural source. Real two-step GPU M0 is not available and must not be fabricated. Save full raw report, exact request trace, input hashes and actual verification receipts; actual reviewer backend/model/effort UNATTESTED unless attested, same-family provisional. Concrete blockers with minimal fixes. You may browse the exact primary Torch v1.10.2 source for static derivative behavior.'
write('001-source-audit.request.json', dict(call_number=1, purpose='fresh-source-and-diagnostic-audit',
    timestamp=NOW, timestamp_meaning='trace materialization time; start time not attested',
    tool='native collaboration delegation; original spawn invocation metadata not independently attested',
    requested_model='gpt-6-astra', requested_reasoning_effort='max',
    parent_task_message_exact=parent_request,
    file_request_exact=(HERE / 'SOURCE_AUDIT_REQUEST.exact.txt').read_bytes().decode('utf-8'),
    file_request_sha256=sha(HERE / 'SOURCE_AUDIT_REQUEST.exact.txt'), files_referenced=list(hashes),
    exact_request_bytes_preserved=True))
write('run.meta.json', dict(metadata, run_id='2026-10-09_R1', started_at='UNATTESTED', trace_written_at=NOW,
    project_dir=str(ROOT), source_snapshot_manifest='R1_SOURCE_SNAPSHOTS.json'))
write('001-source-audit.meta.json', dict(metadata, call_number=1, purpose='fresh-source-and-diagnostic-audit',
    status='completed_with_two_static_source_blockers', response_sha256=sha(HERE / '001-source-audit.response.md'),
    diagnostic_verdict='WARN', deterministic_arithmetic='PASS', source_verdict='FAIL', actual_M0='NOT_AVAILABLE'))
write('ACTUAL_FAILED_INVOCATIONS.json', report['actual_failed_invocations'])
write('EXECUTION_RECEIPTS.json', dict(verification_invocations=[failed, passed],
    offline_python_probe_chunk='baceaa', torch_derivative_download_chunk='16404d',
    torch_module_download_chunk='7ee176', r1_snapshot_copy_chunk='a9e9b2',
    official_sources=[dict(url='https://raw.githubusercontent.com/pytorch/pytorch/v1.10.2/torch/csrc/autograd/FunctionsManual.cpp',
                           path=str(HERE / 'torch_v1_10_2_FunctionsManual.cpp'), sha256=sha(HERE / 'torch_v1_10_2_FunctionsManual.cpp')),
                      dict(url='https://raw.githubusercontent.com/pytorch/pytorch/v1.10.2/torch/nn/modules/module.py',
                           path=str(official_module), sha256=sha(official_module))],
    scope='No Torch import or model execution; downloaded primary source text only.'))
(HERE / 'events.jsonl').write_text(json.dumps(dict(event='review_trace', skill='experiment-audit',
    purpose='fresh-source-and-diagnostic-audit', agent_id=AGENT, trace_path=str(HERE), status='completed',
    written_under_review_folder_instead_of_global_aris=True)) + '\n', encoding='utf-8')
print(json.dumps(dict(status='R1_REPORT_WRITTEN', diagnostic_verdict=report['diagnostic_verdict'],
    source_verdict=report['source_verdict'], blockers=len(blockers), input_files=len(hashes),
    snapshots=len(snapshots), raw_report_sha256=report['raw_report_sha256'],
    review_json_sha256=sha(HERE / 'SOURCE_REVIEW.json'))))
