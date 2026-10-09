"""Write the bounded audit report and forensic trace from verified local evidence."""
import datetime
import hashlib
import json
from pathlib import Path

root = Path(r'C:\Users\gb\.codex\tmp\pvground_native_joint_training_20261009\native_joint_v2\referit_author_core_20261010')
trace = root / '.aris/traces/experiment-audit/author_core_CPU_20261010'
output = root / 'actual_CPU_review'
now = datetime.datetime.now().astimezone().isoformat()
agent = '/root/pvg_referit_author_cpu_actual_20261010'
hashes = json.loads((trace/'audited_input_hashes.json').read_bytes())
checks = json.loads((trace/'deterministic_verification.json').read_bytes())
response = (output/'EXPERIMENT_AUDIT.md').read_text(encoding='utf-8')
request_text = (root/'ACTUAL_CPU_AUDIT_REQUEST.txt').read_text(encoding='utf-8')
assignment = '''请依 experiment-audit 技能做限定、独立于执行者的实际CPU结果复核。只读证据/局部stdlib校验，禁止SSH、模型导入或运行、GPU、训练状态查询、重跑/改输入，不读MEMORY/SOUL/USER/日记/AUTH/askpass。项目 C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/referit_author_core_20261010；任务完整要求在 ACTUAL_CPU_AUDIT_REQUEST.txt。真实CPU已结束，cpu_execution/CPU_EXECUTION.json exit0，2026-10-10 07:13:28–07:13:53 CST；session43694 closed0。两次实际 TrainTester.get_model，严格加载各自作者1235项含position_ids，再nonpersistent保存1234core+新增结构合计1295，逐core shape/dtype/值相等断言、A/B新输出零，CPU/CUDA未初始化，0forward/optimizer/loader/GPU/REC。结果同时记录作者butd_cls=True, butd=False,joint_det/detect_intermediate=True；本次构造butd=True,butd_cls=False,joint_det/detect_intermediate=False，所以不能判定真实训练/对象协议已通过。请核对raw base64回执与stdout/stderr/独立result复制、退出码、真实脚本范围与实际执行断言、R3 114输入哈希及R2 tracker历史例外保护。原native source未改。输出actual_CPU_review/EXPERIMENT_AUDIT.md/json、OUTPUT_SHA256.json，以及 .aris/traces/experiment-audit/author_core_CPU_20261010 完整trace。JSON需verdict、blocking_issue_count、audited_input_hashes和精确支持/不支持scope。请求Astra/max不等于身份已验证；actual identity无法核实时UNATTESTED，same-family/provisional。不要读取无关实验或扩大审计。'''


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


supported = [
    'Exactly two recorded real TrainTester.get_model CPU constructions: nr3d and sr3d, seed 2027 each.',
    'Strict loading of the corresponding author checkpoint containing 1235 entries, including deterministic position_ids, before making that buffer nonpersistent.',
    'Each of 1234 retained core state entries checked for key, shape, dtype, CPU device and exact tensor value against a reread of the same author checkpoint.',
    'In-memory state count 1295 after new module installation; text encoder frozen; G/C supervision flags checked.',
    'A/B output-layer state (weight and bias) is zero; no A/B or model forward was performed.',
    'G/A/B newly installed for each author core, with G partly copied from that author-loaded core according to source; no ScanRefer training checkpoint/module overlay in this initialization path.',
    'Raw base64 stdout/stderr/result, copied files, metadata and terminal exit codes are consistent; all 114 R3 input hashes match.',
    'R2 source/spec and R1/R2 report bytes are preserved; the one historical live tracker exception has its exact archived original and verified path alias.'
]
unsupported = [
    'Scientific ground-truth performance, REC accuracy, performance improvement or independent numerical reference correctness.',
    'Model/G/A/B forward behavior, including a measured claim that A/B forward outputs are zero.',
    'Full criterion, real data-loader/object input correctness, optimizer updates, training or GPU execution.',
    'Model checkpoint serialization/restoration, cold recovery or resume correctness; 1295 is an in-memory state_dict count.',
    'Equivalence between author Nr3D/Sr3D object/detection/training protocols and inherited ScanRefer construction flags.',
    'All G parameters being newly randomized or independently value-verified; G copies parts of the corresponding loaded author core.',
    'Current remote training/source state beyond the preserved execution record; no live query was made.',
    'Cross-family review acceptance or an attested actual reviewer model/reasoning identity.',
    'Multiseed robustness, complete cross-benchmark readiness, or permission to start formal training.'
]
audit = {
    'schema': 'bounded_actual_CPU_author_core_experiment_audit',
    'audit_skill': 'experiment-audit', 'verdict': 'WARN', 'overall_verdict': 'warn',
    'integrity_status': 'pass_with_scope_limits', 'bounded_CPU_evidence_verdict': 'PASS',
    'deterministic_verification_verdict': 'PASS', 'blocking_issue_count': 0,
    'blocking_issue_scope': 'Only the preserved actual CPU constructor/author-core loading evidence, not downstream training readiness.',
    'reason_code': 'engineering_loading_supported_training_protocol_and_reviewer_identity_unverified',
    'summary': 'Actual recorded two-case CPU factory/loading evidence is internally consistent and supports exact author-core loading only. No blocker found within that limited scope.',
    'generated_at': now, 'date': now[:10], 'auditor': agent, 'agent_id': agent,
    'agent_id_type': 'canonical_task_path', 'verdict_id': 'author_core_CPU_20261010',
    'requested_reviewer_route': {'model': 'gpt-6-astra', 'reasoning_effort': 'max'},
    'actual_reviewer_identity': {'agent': agent, 'model': 'UNATTESTED', 'reasoning_effort': 'UNATTESTED'},
    'executor_model': 'UNATTESTED', 'reviewer_model': 'UNATTESTED',
    'reviewer_reasoning': 'UNATTESTED', 'reviewer_family': 'UNATTESTED',
    'review_independence': 'same-family', 'acceptance_status': 'provisional',
    'independence_description': 'Separate reviewer agent directly inspected primary artifacts. Parent assignment included an executor summary; no blind or strict zero-context independence claim. same-family is the policy route label, not a model-identity attestation.',
    'evaluation_type': 'engineering_initialization_check', 'scientific_GT_classification': 'not_applicable',
    'source_review_sha256': '87f68880f93a224aaf89fc4f5bfc67504b135c1d1187d4ae97ed85c3f0e4e287',
    'audited_input_hashes': hashes,
    'audited_input_hash_count': len(hashes),
    'trace_path': '.aris/traces/experiment-audit/author_core_CPU_20261010',
    'checks': {
        'gt_provenance': {'status':'PASS', 'scientific_metric_applicable':False,
            'details':'Reference is the corresponding SHA-bound author state, reread on CPU. Exact loading agreement is not independent scientific ground truth.',
            'evidence':['source/native_model_initialization.py:13-15,30-36','../referit_author_core_20261010_check_cpu.py:43-54','nr3d/init.json:4-5','sr3d/init.json:4-5']},
        'score_normalization': {'status':'PASS', 'scientific_metric_applicable':False,
            'details':'No metric or normalization is computed. Exact equality booleans do not constitute accuracy.',
            'evidence':['../referit_author_core_20261010_check_cpu.py:50-59,77-80']},
        'result_existence': {'status':'PASS', 'details':'All three strict base64 decodings match local bytes, stdout JSON equals saved result JSON, raw/local metadata agree, transport/child/local receipt exit codes are 0.',
            'evidence':['cpu_execution/RAW_STDOUT.json:1','cpu_execution/TRANSPORT_EXIT.json:1','cpu_execution/CPU_EXECUTION.json:2-11','../run_referit_author_core_cpu_authorized.py:54-61,73-86']},
        'dead_code': {'status':'PASS', 'details':'Real get_model is called once per dataset. Assertions precede result writing; successful recorded exit and bound source support execution. No per-operation external instrumentation is claimed.',
            'evidence':['../referit_author_core_20261010_check_cpu.py:29-82','source/train_dist_mod.py:100-129','source/native_model_initialization.py:27-48']},
        'scope': {'status':'WARN', 'details':'Two constructors, one seed, zero dataset rows. Four recorded author/CPU flags differ. The audit does not clear real protocols or training.',
            'evidence':['cpu_execution/AUTHOR_CORE_CPU_RESULT.json:33-59,91-117','source/train_dist_mod.py:61-80,83-96,121']},
        'eval_type': {'status':'PASS','classification':'engineering_initialization_check','details':'No scientific GT/prediction evaluation; real_gt/synthetic_proxy/simulation_only classifications do not apply.'},
        'artifact_continuity': {'status':'PASS_WITH_DOCUMENTED_EXCEPTION',
            'details':'114/114 R3 inputs, 16/16 R2 source/spec files and 8/8 historical reports match. 75/76 original R2 live inputs match; the one tracker exception has verified 03bca archived bytes and C:/D: path aliases.'},
        'reviewer_identity': {'status':'WARN','details':'Requested gpt-6-astra/max is recorded separately; actual model and effort remain UNATTESTED.'}
    },
    'runtime_record': checks['receipt'], 'case_evidence': checks['cases'],
    'artifact_continuity': {key: checks[key] for key in ('R3_inputs','R2_inputs','R2_source_and_spec_count','R2_source_and_spec_matches','protected_R1_R2_report_count','protected_R1_R2_report_matches','changed_since_R2','tracker_alias_checks')},
    'supported_scope': supported, 'unsupported_scope': unsupported,
    'qualifications': [
        {'id':'Q1','details':'A/B zero check applies to output-layer state, not forward output.'},
        {'id':'Q2','details':'G fresh installation includes copies of corresponding author-core projection/attention parameters.'},
        {'id':'Q3','details':'Zero-operation fields are declarative counters corroborated by static call boundaries, not runtime hook measurements.'},
        {'id':'Q4','details':'Independent result file and stdout are two representations from the same process, not independent scientific replications.'},
        {'id':'Q5','details':'Session 43694 closed0 was supplied in delegation metadata; no independent session query or durable session receipt was available in the scoped CPU artifacts.'}
    ],
    'claims': [
        {'id':'C1','claim':'Current isolated factory can construct Nr/Sr CPU models and strictly load their corresponding author core under the recorded flags.','impact':'supported_with_recorded_scope'},
        {'id':'C2','claim':'All 1234 retained core states exactly equal the corresponding author states after new module installation.','impact':'supported'},
        {'id':'C3','claim':'A/B actual forward outputs were evaluated and are zero.','impact':'unsupported'},
        {'id':'C4','claim':'The Nr/Sr author object-input and training protocols have passed.','impact':'unsupported'},
        {'id':'C5','claim':'All live R2 inputs have always stayed unchanged.','impact':'unsupported_documented_tracker_exception'},
        {'id':'C6','claim':'Formal training, recovery or REC accuracy has passed.','impact':'unsupported'}
    ],
    'context_limits': {'executor_summary_received':True,'primary_artifacts_checked_directly':True,
        'strict_blind_review':False,'personal_memory_or_bootstrap_files_read':False,
        'auth_or_askpass_contents_read':False,'historical_bootstrap_disclosure_source':'Existing R3 metadata only'},
    'reviewer_execution': checks['reviewer_execution'],
    'next_steps': ['No CPU source patch required by this bounded audit.','Do not extend this result to training clearance. Final Nr/Sr work must use the settled common method and separately check real data/object/detection protocols.'],
    'training_clearance': False, 'full_goal_complete': False
}
assert len(hashes) == 137
assert checks['status'] == 'PASS'
assert not (output/'OUTPUT_SHA256.json').exists()
write(output/'EXPERIMENT_AUDIT.json', audit)

write(trace/'run.meta.json', {
    'skill':'experiment-audit','run_id':'author_core_CPU_20261010','started_at':None,
    'trace_recorded_at':now,'start_time_limitation':'Actual parent spawn timestamp is not independently exposed in this reviewer context.',
    'executor':'codex_native_agent','executor_model':'UNATTESTED','review_independence':'same-family',
    'acceptance_status':'provisional','project_dir':root.as_posix(),'agent_id':agent,
    'requested_reviewer_route':audit['requested_reviewer_route'],'actual_reviewer_identity':audit['actual_reviewer_identity'],
    'trace_mode':'full','trace_helper_resolution':{'ARIS_REPO_set':False,'project_install_manifest_exists':False,
        'project_tools_save_trace_exists':False,'global_skill_save_trace_exists':False,'method':'Direct files under Policy C'},
    'trace_scope':'Full received reviewer task plus linked request, full review response, source/evidence snapshots and deterministic execution receipt. This is not an export of hidden system prompts or a claim of a platform session transcript.'
})
write(trace/'001-actual-cpu-review.request.json', {
    'call_number':1,'purpose':'bounded-actual-CPU-author-core-review','timestamp':None,
    'recorded_at':now,'tool':'collaboration.spawn_agent (received task; parent call arguments not independently visible)',
    'requested_model':'gpt-6-astra','requested_reasoning_effort':'max','agent_id':agent,
    'files_referenced':list(hashes),'prompt':assignment,'linked_request_file':'ACTUAL_CPU_AUDIT_REQUEST.txt',
    'linked_request_full_text':request_text,'caller_summary_in_prompt':True
})
(trace/'001-actual-cpu-review.response.md').write_bytes((output/'EXPERIMENT_AUDIT.md').read_bytes())
write(trace/'001-actual-cpu-review.meta.json', {
    'call_number':1,'purpose':'bounded-actual-CPU-author-core-review','timestamp':now,
    'agent_id':agent,'model':'UNATTESTED','reasoning_effort':'UNATTESTED',
    'requested_model':'gpt-6-astra','requested_reasoning_effort':'max','reviewer_family':'UNATTESTED',
    'review_independence':'same-family','acceptance_status':'provisional','duration_ms':None,
    'status':'ok','verdict':'WARN','bounded_CPU_evidence_verdict':'PASS','blocking_issue_count':0
})
write(trace/'review_actions.json', {
    'actions':[
        'Read experiment-audit skill and its local routing, independence, integrity and tracing references.',
        'Read the scoped audit request, checker, runner, initializer, native factory and constructor/module initialization call paths without importing them.',
        'Read raw CPU stdout/stderr/metadata/result files; strictly decoded all three base64 payloads and compared bytes.',
        'Recomputed 114 R3 hashes, original R2 76 hashes, 16 source/spec hashes and 8 sealed report hashes.',
        'Verified exact old tracker snapshot and actual same-file C:/D: aliases; did not rewrite old inputs.',
        'Recomputed three archived inventory comparisons and matched historical strict-load script/checkpoint attribution.',
        'Parsed 17 current Python files and one embedded runner block with stdlib AST; did not execute audited source.',
        'Saved raw CPU evidence copies, source snapshots, local verifier source and actual successful local verifier tool receipt.',
        'Wrote qualified audit and full received request/response trace; no remote or model execution.'
    ],
    'local_interpreter_probe':{'default_python':'E:/Scripts/python.exe','result':'No pyvenv.cfg file',
        'py_launcher':'not installed','resolution':'uv python find --offline returned existing CPython 3.13',
        'executed_interpreter':'C:/Users/gb/AppData/Roaming/uv/python/cpython-3.13-windows-x86_64-none/python.exe',
        'install_performed':False,'audited_CPU_rerun':False},
    'output_writer_correction':{'first_writer_exit_code':1,'cause':'Windows text writing changed response newline bytes; report/trace byte identity assertion caught it.',
        'fix':'Copy Markdown report bytes directly to the raw response; retry only output serialization before output manifest is sealed.',
        'CPU_or_input_changed':False,'raw_failure_receipt':'output_writer_initial_failure.json'},
    'execution':checks['reviewer_execution'], 'input_edits':False,
    'identity':audit['actual_reviewer_identity'],'context_limits':audit['context_limits']
})

old_validation = json.loads((root/'.aris/traces/experiment-bridge/R3/report_validation.json').read_bytes())
assert all(hashes[name] == expected for name,expected in old_validation['output_sha256'].items())
required = {'verdict','blocking_issue_count','audited_input_hashes','supported_scope','unsupported_scope','actual_reviewer_identity','review_independence','acceptance_status'}
assert required <= set(audit)
assert (trace/'001-actual-cpu-review.response.md').read_bytes() == (output/'EXPERIMENT_AUDIT.md').read_bytes()
write(trace/'report_validation.json', {
    'status':'PASS','required_fields':sorted(required),'audited_input_hash_count':len(hashes),
    'R3_sealed_reports_match_recorded_hashes':True,'response_byte_identical_to_report':True,
    'verdict':'WARN','bounded_CPU_evidence_verdict':'PASS','blocking_issue_count':0,
    'no_training_clearance':True,'actual_identity_UNATTESTED':True,
    'validation_type':'Artifact-schema/content-consistency only; no repeat CPU run.'
})
meta = root/'.aris/meta'
meta.mkdir(exist_ok=True)
event = {'event':'review_trace','skill':'experiment-audit','purpose':'bounded-actual-CPU-author-core-review',
    'agent_id':agent,'trace_path':audit['trace_path'],'status':'ok','verdict':'WARN','time':now}
with (meta/'events.jsonl').open('a',encoding='utf-8') as stream:
    stream.write(json.dumps(event,ensure_ascii=False,separators=(',',':'))+'\n')
manifest = {}
for directory in (output,trace):
    for path in sorted(directory.rglob('*')):
        if path.is_file() and path != output/'OUTPUT_SHA256.json':
            manifest[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
write(output/'OUTPUT_SHA256.json', {'algorithm':'sha256','generated_at':now,
    'self_excluded':True,'shared_append_only_event_log_excluded':True,'files':manifest})
print(json.dumps({'status':'WRITTEN','verdict':'WARN','bounded_CPU_evidence_verdict':'PASS',
    'blocking_issue_count':0,'audited_input_hash_count':len(hashes),'output_file_count':len(manifest),
    'report_sha256':manifest['actual_CPU_review/EXPERIMENT_AUDIT.json'],
    'markdown_sha256':manifest['actual_CPU_review/EXPERIMENT_AUDIT.md']},ensure_ascii=False))
