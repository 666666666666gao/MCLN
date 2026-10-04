"""Assemble reviewer-owned audit JSON and private forensic trace, without altering inputs."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path('C:/Users/gb/.codex/tmp/pvground_geometry_readback_20261004/formal_draft')
OUT = ROOT / 'analysis'
TRACE = ROOT / '.aris/traces/experiment-audit/2026-10-05_readback_terminal_run01'
AGENT = '/root/pvg_readback_terminal_integrity'
NOW = datetime.datetime.now(datetime.timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


cpu = read_json(OUT / 'AUDIT_CPU_CHECK.json')
inputs = read_json(OUT / 'AUDITED_INPUT_SHA256.json')
assert cpu['status'] == 'PASS_RECORDED_DATA_CPU_CHECKS'
assert len(inputs) == 117
unchanged = []
for name, identity in inputs.items():
    data = Path(name).read_bytes()
    assert len(data) == identity['bytes'] and hashlib.sha256(data).hexdigest() == identity['sha256'], name
    unchanged.append(name)
assert cpu['checker_sha256'] == sha(OUT / 'audit_cpu_check.py')
assert all(not any(item['same_query_threshold_disagreements'].values()) for item in cpu['recorded_same_query_cross_forward_drift'].values())

checks = {
    'gt_provenance': {
        'letter': 'A', 'status': 'PASS',
        'details': 'Identity-matched executed dataset source obtains target IDs and sampled-point Box/Mask geometry from ScanRefer/ScanNet annotations, separate from detected input boxes. GT is joined after model forward; no model-derived target substitution was found.',
        'evidence': [
            'C:/Users/gb/.codex/tmp/pvground_g_p2_20261002/complete/source/joint_det_dataset.py:205',
            'C:/Users/gb/.codex/tmp/pvground_g_p2_20261002/complete/source/joint_det_dataset.py:637',
            'C:/Users/gb/.codex/tmp/pvground_g_p2_20261002/complete/source/joint_det_dataset.py:1086',
            'C:/Users/gb/.codex/tmp/pvground_g_p2_20261002/complete/source/joint_det_dataset.py:1189',
            'C:/Users/gb/.codex_mcln_g0_20260905/src/visual_data_handlers.py:129',
            'C:/Users/gb/.codex_mcln_g0_20260905/src/visual_data_handlers.py:225',
            'complete/evidence_hidden/imports.json:8', 'run_readback_fit.py:240', 'run_readback_fit.py:255'],
        'limit': 'Raw original scene caches/annotation files were unavailable locally; no second raw-data GT extraction was performed.'},
    'score_normalization': {
        'letter': 'B', 'status': 'PASS',
        'details': 'Native bbs sums token softmax against binary main positive map plus raw modify/pronoun/relation maps minus raw other map, then ranks all 256 queries. Box/Mask share the selected query. Accuracy divides by example count; IoU divides by geometric union.',
        'evidence': ['runtime_bundle/native_root_bbs.py:4', 'run_readback_fit.py:307', 'run_readback_fit.py:311', 'run_readback_fit.py:334',
            'C:/Users/gb/.codex/tmp/pvground_runtime_bundle_20260908_v1/PV-Ground/src/grounding_evaluator.py:218',
            'C:/Users/gb/.codex/tmp/pvground_runtime_bundle_20260908_v1/PV-Ground/src/grounding_evaluator.py:594'],
        'limit': 'Full logits/maps and raw Mask tensors were not exported, so terminal ranking generation and Mask IoU are source/runtime evidence plus aggregate consistency rather than fresh inference.'},
    'result_existence': {
        'letter': 'C', 'status': 'PASS',
        'details': 'All 40 terminal intake artifacts and 28 corrected-V2 preflight artifacts match SHA256 and byte count. Five exits are zero. Independent stdlib reconstruction checked 56072 recorded Box/GT rows, both 3723-update traces, exact sample membership/order, all report counts, paired changes and volume groups.',
        'evidence': ['complete/INTAKE.json:3', 'complete/status.json:2', 'complete/evidence_hidden/train.jsonl:3723',
            'complete/evidence_visible/train.jsonl:3723', 'analysis/audit_cpu_check.py:1', 'analysis/AUDIT_CPU_CHECK.json:1',
            'analysis/AUDITED_INPUT_SHA256.json:1', 'analysis/AUDIT_PAIRED_CHANGES.csv:1'],
        'limit': 'Recomputation uses recorded dataset GT boxes, not raw scene re-extraction; candidate-oracle flags and Mask IoU are only re-aggregated.'},
    'dead_code': {
        'letter': 'D', 'status': 'WARN',
        'details': 'The actual formal path defers and invokes the final native semantic head once, evaluates native bbs, strictly restores model and AdamW in a new process, and retires only completed inferior owned weights with parent hashes preserved. Several imported preflight helpers and inherited loss functions are not called in formal R fitting; no terminal result is attributed to them.',
        'evidence': ['run_readback_fit.py:216', 'run_readback_fit.py:277', 'run_readback_fit.py:285', 'run_readback_fit.py:288',
            'run_readback_fit.py:389', 'run_readback_fit.py:421', 'runtime_bundle/readback_preflight_checks.py:19',
            'runtime_bundle/readback_preflight_checks.py:71', 'runtime_bundle/whole_model_preflight_checks.py:30',
            'controller.py:72', 'controller.py:114', 'controller.py:120',
            'complete/evidence_hidden/formal_restore.json:2', 'complete/evidence_visible/weight_retention.json:9'],
        'limit': 'No reviewer model/optimizer reload or current remote filesystem inspection occurred; raw tensors and removed R weights are unavailable.'},
    'scope': {
        'letter': 'E', 'status': 'WARN',
        'details': 'Two same-capacity arms, seed 2027, frozen official/G/4506 parents, one R fit pass. Module holdout is pretrained-seen; formal ScanRefer developer validation has 9508 rows and 141 recorded physical scenes. Fixed-frame head replay is a diagnostic, not an independently trained ablation. Recorded complete forwards differ numerically despite matching query IDs and within-frame preservation.',
        'evidence': ['run_readback_fit.py:43', 'run_readback_fit.py:146', 'run_readback_fit.py:166', 'run_readback_fit.py:288',
            'runtime_bundle/readback_model_factory.py:18', 'runtime_bundle/pvground_boundary_evidence_readback.py:63',
            'C:/Users/gb/.codex/tmp/pvground_g_p2_20261002/complete/source/split_protocol.json:1',
            'analysis/AUDIT_CPU_CHECK.json:1', 'analysis/RESULTS.md:18'],
        'limit': 'No stable multi-seed improvement, untouched test-set success, cross-dataset generalization, or novelty priority is demonstrated.'},
    'eval_type': {
        'letter': 'F', 'status': 'PASS', 'classification': 'real_gt',
        'details': 'Initial/terminal module holdout and formal Box/Mask evaluations use real dataset annotations. Full256 and GT-volume diagnostics also use real GT offline. Cached-head bypass is a fixed-frame forward diagnostic; preflight equalities are implementation checks, not accuracy results.',
        'evidence': ['run_readback_fit.py:255', 'run_readback_fit.py:301', 'run_readback_fit.py:315',
            'C:/Users/gb/.codex/tmp/pvground_g_p2_20261002/complete/source/joint_det_dataset.py:1388',
            'C:/Users/gb/.codex/tmp/pvground_geometry_readback_20261004/revision2/complete_preflight/evidence_visible/preflight.json:2']}
}

warnings = [
    {'id': 'W1', 'status': 'WARN', 'blocking': False, 'finding': 'Raw original datasets, full candidate boxes/logits/maps and raw masks are unavailable. Box arithmetic is independently reconstructed from saved GT/selected boxes; terminal ranking, Mask and oracle geometry are not independently rerun.'},
    {'id': 'W2', 'status': 'WARN', 'blocking': False, 'finding': 'Both inferior R weights were already deleted by the executed retention policy. Exact model/AdamW restoration, remote deletion and parent preservation are verified in identity-matched source plus original runtime receipts, not repeated by the reviewer.'},
    {'id': 'W3', 'status': 'WARN', 'blocking': False, 'finding': 'Single seed and dataset; module holdout is pretrained-seen and formal developer validation is reused. The visible arm is +2 strict hits vs hidden but -29 vs the protected parent; this supports a negative descriptive result, not a reliable gain.'},
    {'id': 'W4', 'status': 'WARN', 'blocking': False, 'finding': 'Recorded independent forwards contain coordinate differences on same-query rows. No checked same-query @.25/.50 hit disagreement occurs, but full numerical candidate identity across history/arms cannot be claimed.'},
    {'id': 'W5', 'status': 'WARN', 'blocking': False, 'finding': 'Some imported preflight and inherited loss helpers are unused in formal fit. Earlier preflight/source success cannot stand in for terminal metric proof.'}
]

formal = {arm: {k: cpu['stages'][arm]['formal'][k] for k in ('rows','rec_hits25','rec_hits50','rec_acc25','rec_acc50','mask_hits25','mask_hits50','mask_miou')} for arm in ('evidence_hidden','evidence_visible')}
formal['protected_parent'] = {k: cpu['parent'][k] for k in ('rows','rec_hits25','rec_hits50','rec_acc25','rec_acc50','mask_hits25','mask_hits50','mask_miou')}
route = dict(requested_reviewer_model='gpt-6-astra', requested_reviewer_reasoning_effort='max',
    actual_reviewer_model=None, actual_reviewer_reasoning_effort=None,
    actual_backend_attestation='unavailable', model_and_effort_attested=False,
    reviewer_model=None, reviewer_reasoning=None, executor_model=None,
    reviewer_family='openai', executor_family='openai',
    review_independence='same-family', acceptance_status='provisional')
report = dict(audit_skill='experiment-audit', verdict='WARN', overall_verdict='warn', integrity_status='warn',
    reason_code='qualified_negative_result_with_scope_and_replay_limits',
    summary='Terminal recorded-data metrics and provenance checks support the completed negative comparison. Visible reaches 4477/9508 strict hits versus hidden 4475 and protected parent 4506. No blocker for qualified reporting; independent raw-data/inference/restore replay and broad efficacy claims are unsupported.',
    date='2026-10-05', generated_at=NOW, auditor='Fresh delegated Codex reviewer; requested Astra/max, actual model/effort unattested',
    agent_id=AGENT, agent_uuid=None, verdict_id='2026-10-05_readback_terminal_run01',
    **route, evaluation_type='real_gt', checks=checks,
    required_check_statuses={v['letter']:v['status'] for v in checks.values()},
    blocking_findings=[], nonblocking_findings=warnings,
    actual_limitations=[v['finding'] for v in warnings],
    resolved_findings=[dict(id='R1', status='RESOLVED', finding='The initial local runtime env_spec archive was stale. The exact executed canonical identity was located and checked in witness_repair_v3; failed initial assertion and resolution are preserved.', evidence=cpu['environment_identity'])],
    metrics=formal, formal_visible_vs_hidden=cpu['visible_vs_hidden']['formal'],
    versus_protected_parent=cpu['versus_parent'], formal_gt_volume_groups=cpu['formal_gt_volume_groups'],
    training=cpu['training_logs'], same_training_order=cpu['same_training_order'],
    same_capacity=dict(parameters=96672, state_tensors=23, spec_differences=cpu['spec_differences'], frozen_parent_states_exact='original runtime assertions/receipts, not reviewer tensor reload'),
    target=dict(required_hits25=5615, required_hits50=4754, achieved=False, protected_best_hits50=4506, remaining_strict_hits=248),
    independent_forward_drift=cpu['recorded_same_query_cross_forward_drift'],
    deterministic_check=dict(status=cpu['status'], review_independence='deterministic', acceptance_status='accepted', scope=cpu['scope'],
        rows_recomputed=cpu['rows_recomputed'], terminal_intake_files=cpu['intact_intake_files'], terminal_intake_bytes=cpu['intake_bytes'],
        preflight_intake_files=cpu['preflight_intake_files'], zero_selected_coarse_bypass_threshold_disagreements=True,
        summary_csv_match=True, source_file_syntax_count=cpu['syntax_files'], final_input_hash_recheck_count=len(unchanged)),
    evidence_classes={
        'independently_repeated_cpu': ['all supplied artifact hashes and sizes', 'selected/coarse/bypass IoU vs saved dataset GT boxes', 'threshold hits and paired repairs/damages', 'volume grouping', 'row coverage and physical scene/salt checks', 'sample order and update/batch arithmetic', 'trace/progress/receipt/summary consistency', 'recorded same-query coordinate drift'],
        'source_and_original_runtime_receipts': ['GT extraction route', 'native score formula and all256 ranking path', 'same-query Mask decoding', 'exact frozen model/AdamW tensor restoration', 'gradient/G qualification invariants', 'within-frame preservation', 'parent preservation and terminal weight deletion'],
        'saved_scalar_or_flag_aggregation_only': ['Mask IoU metrics', 'top16/32/64/full256 oracle coverage'],
        'not_available_or_not_performed': ['raw-data GT re-extraction', 'fresh GPU/CPU model inference', 'independent model/optimizer reload', 'current remote filesystem inspection', 'multi-seed evidence', 'Nr3D/Sr3D evaluation', 'external novelty review']},
    claims=[
        dict(id='C1', claim='Completed one-pass same-capacity evidence-visible vs hidden comparison', impact='supported'),
        dict(id='C2', claim='Visible +2 strict formal hits vs hidden; both below 4506 parent', impact='supported'),
        dict(id='C3', claim='Geometry evidence gives a stable accuracy improvement', impact='unsupported'),
        dict(id='C4', claim='All parent and auxiliary outputs are exactly identical across independent full forwards', impact='unsupported'),
        dict(id='C5', claim='Parent state is frozen and same-forward geometry/mask is preserved', impact='needs_qualifier', qualifier='Identity-matched source and original runtime assertions; no reviewer tensor replay.'),
        dict(id='C6', claim='Target, multi-seed, cross-dataset or novelty success', impact='unsupported'),
        dict(id='C7', claim='Fixed-frame bypass is an independently trained ablation', impact='unsupported')],
    action_items=['Report the completed negative comparison with W1-W5 qualifiers.', 'Preserve protected parent result; do not promote either R arm.', 'Reference the exact witness_repair_v3 environment archive for reproduction.', 'Use fresh data/weights and broader runs only if making stronger replay or efficacy claims.'],
    audit_operations=dict(read_only_experiment_inputs=True, training_or_inference_run=False, ssh_or_remote_jobs=False, credentials_or_wrappers_read=False, author_analyzer_executed=False, reviewer_owned_outputs_only=True),
    audited_inputs=inputs, audited_input_hashes={k:'sha256:'+v['sha256'] for k,v in inputs.items()},
    artifacts={name:dict(path=(OUT/name).as_posix(),sha256=sha(OUT/name)) for name in ('EXPERIMENT_AUDIT.md','audit_cpu_check.py','AUDIT_CPU_CHECK.json','AUDIT_PAIRED_CHANGES.csv','AUDITED_INPUT_SHA256.json')},
    trace_path=TRACE.as_posix())
write_json(OUT / 'EXPERIMENT_AUDIT.json', report)

# Trace preserves the actual supplied review request and the full reviewer-written response.
request_text = (TRACE / '001-terminal-integrity.request.txt').read_text(encoding='utf-8')
write_json(TRACE / 'run.meta.json', dict(skill='experiment-audit', run_id='2026-10-05_readback_terminal_run01',
    started_at=None, recorded_at=NOW, executor='codex', parent_agent='/root', reviewer_agent=AGENT,
    project_dir=ROOT.as_posix(), trace_mode='full', **route))
write_json(TRACE / '001-terminal-integrity.request.json', dict(call_number=1, purpose='terminal-integrity', timestamp=None,
    recorded_at=NOW, tool='collaboration.spawn_agent', requested_model='gpt-6-astra', requested_reasoning_effort='max',
    actual_model=None, actual_reasoning_effort=None, files_referenced=list(inputs), prompt=request_text))
(TRACE / '001-terminal-integrity.response.md').write_bytes((OUT / 'EXPERIMENT_AUDIT.md').read_bytes())
(TRACE / '001-terminal-integrity.response.json').write_bytes((OUT / 'EXPERIMENT_AUDIT.json').read_bytes())
(TRACE / '001-cpu-check.final.py').write_bytes((OUT / 'audit_cpu_check.py').read_bytes())
(TRACE / '001-cpu-check.final.json').write_bytes((OUT / 'AUDIT_CPU_CHECK.json').read_bytes())
(TRACE / '001-input-sha256.json').write_bytes((OUT / 'AUDITED_INPUT_SHA256.json').read_bytes())
write_json(TRACE / '001-terminal-integrity.meta.json', dict(call_number=1, purpose='terminal-integrity', timestamp=NOW,
    agent_id=AGENT, agent_uuid=None, duration_ms=None, status='ok', verdict='WARN',
    response_md_sha256=sha(OUT/'EXPERIMENT_AUDIT.md'), response_json_sha256=sha(OUT/'EXPERIMENT_AUDIT.json'), **route))
write_json(TRACE / '001-check-provenance.json', dict(recorded_at=NOW,
    exact_command="& 'C:\\Users\\gb\\AppData\\Roaming\\uv\\python\\cpython-3.11.14-windows-x86_64-none\\python.exe' 'C:\\Users\\gb\\.codex\\tmp\\pvground_geometry_readback_20261004\\formal_draft\\analysis\\audit_cpu_check.py'",
    cpu_check_exit_code=0, python=cpu['python'], executable=cpu['executable'],
    final_checker_sha256=cpu['checker_sha256'], cpu_output_sha256=sha(OUT/'AUDIT_CPU_CHECK.json'),
    report_md_sha256=sha(OUT/'EXPERIMENT_AUDIT.md'), report_json_sha256=sha(OUT/'EXPERIMENT_AUDIT.json'),
    input_hashes_sha256=sha(OUT/'AUDITED_INPUT_SHA256.json'), unchanged_input_count=len(unchanged),
    mutations=['reviewer-owned analysis/check/report files', 'private audit trace', 'private review_trace event'],
    execution_journal=[
        dict(attempt='initial default-PATH invocation', outcome='failed before script: No pyvenv.cfg file', effect_on_experiment='none'),
        dict(attempt='stdlib interpreter checker v1', outcome='assertion on stale initial env_spec archive after numerical checks', preserved='001-cpu-check.v1.py and 001-cpu-check.v1.stderr-and-resolution.txt'),
        dict(attempt='resolved executing env archive', outcome='all then-current CPU assertions passed'),
        dict(attempt='additional dataset-helper/original-G/preflight provenance checks', outcome='passed'),
        dict(attempt='final same-query threshold-stability check', outcome='passed, exit code 0; exact final stdout retained')],
    final_input_recheck='All 117 previously hashed evidence files still match byte-for-byte after audit.'))

events = ROOT / '.aris/meta/events.jsonl'
events.parent.mkdir(parents=True, exist_ok=True)
with events.open('a', encoding='utf-8') as stream:
    stream.write(json.dumps(dict(event='review_trace', skill='experiment-audit', purpose='terminal-integrity',
        agent_id=AGENT, trace_path=TRACE.as_posix(), status='ok', verdict='WARN', timestamp=NOW,
        review_independence='same-family', acceptance_status='provisional'))+'\n')
print(json.dumps(dict(verdict=report['verdict'], checks=report['required_check_statuses'],
    unchanged_inputs=len(unchanged), report_json_sha256=sha(OUT/'EXPERIMENT_AUDIT.json'),
    report_md_sha256=sha(OUT/'EXPERIMENT_AUDIT.md'), trace_path=TRACE.as_posix()), indent=2))
