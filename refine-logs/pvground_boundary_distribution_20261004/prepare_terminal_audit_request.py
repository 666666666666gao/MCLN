"""Prepare a fresh audit request only after the full native pair is collected."""
import json
from pathlib import Path

local = Path(__file__).resolve().parent
assert (local / 'complete/INTAKE.json').is_file()
assert (local / 'analysis/SUMMARY.json').is_file()
assert not (local / 'TERMINAL_AUDIT_REQUEST.txt').exists()
paths = [local / name for name in (
    'collect_terminal.py', 'analyze_terminal.py', 'controller.py',
    'launch.json', 'residual_spec.json', 'distribution_spec.json',
    'EXPERIMENT_PLAN.md', 'run_boundary_fit.py', 'pvground_boundary_box_refiner.py',
    'PREFLIGHT_ANALYSIS.json', 'launch_publication.json',
    'terminal_tool_preparation.json', 'terminal_tool_text_correction.json',
    'terminal_tool_volume_group_update.json')]
paths += sorted(path for path in (local / 'complete').rglob('*') if path.is_file())
paths += [local / 'analysis/SUMMARY.json', local / 'analysis/REPORT.md',
    local.parent / 'pvground_candidate_consistency_20261003/analyze_complete_initial_qualified.py']
native = local.parent / 'pvground_fused_support_20261002/complete_tail_fused_retry/source/imported'
paths += [native / name for name in (
    'models.pv_ground.py', 'models.losses.py', 'main_utils.py',
    'evaluator.py', 'src.joint_det_dataset.py', 'prepare_data.py')]
assert all(path.is_file() for path in paths)
request = '''You are a fresh experiment integrity auditor. Read primary files directly; do not trust the executor's interpretation. Use the installed experiment-audit same-family/provisional route. Do not claim independent backend SKU or cross-family acceptance.

Read every listed configuration, source, receipt and actual JSON/NDJSON evaluation record. You may run deterministic local CPU recounts. Do not use SSH/CUDA, import/run models, read weights, raw goals or credentials/auth wrappers, query or restart jobs, modify primary artifacts or implementation, or spawn agents. Write only analysis/EXPERIMENT_AUDIT.md, analysis/EXPERIMENT_AUDIT.json and identified audit-owned CPU reports.

Audit A-F: A GT provenance versus model-derived proxies; B denominators and prediction-derived normalization; C actual artifact/metric/claim existence and terminal status; D actual called metric/scorer paths; E seeds, configurations, examples, development versus pretrained-seen scope and claim ceiling; F evaluation-type classification. Give exact file:line/record references, PASS/WARN/FAIL per item and overall, execution_scope, blocking_issues[], nonblocking_issues[], actual read identities and limitations. Record review_independence=same-family and acceptance_status=provisional.

Verify frozen original-G parameters and buffers/eval state, ten trainable head tensors, matched final native Hungarian responsibilities, target direction and nonuniform interpolation/finite-before-clamp/outside counts, standalone DFL/7 normalization, deployed final-box losses, common1e-6 size floor and actual floor statistics. Residual400614 versus distribution456102 changes representation, supervision and output parameter count; do not claim pure DFL causality or a completed direction-conditioned token decoder. Boundary entropy is uncalibrated.

Check identical whole/global/local support input and all256 candidates, actual fit rows once/B8-tail2/3723 updates/order, finite losses, actual cross-process initial comparison, source SHA declarations and native imports copies tied to this run. Verify native unique last/bbs selection, selected Box/Mask Query identity, no GT inference filters or dual rank mixing. Check selected-box CPU reconstruction versus saved scalar Mask/oracle recount; do not invent raw Mask or full candidate box replay. Same-Query internal coarse/final repair is distinct from between-arm effects and historical-G improvement.

Check actual restore/formal9508 closure, controller retention bounds and deletion receipts; protected original G/official PV/V99 chain, no failed local weight archive. Report actual issues and observed numerical/scope limits, not hypothetical fallbacks or unrelated refactors. New Nr3D/Sr3D/teacher/quality callback results must not be inferred from this experiment.

Actual existing files:
'''
request += '\n'.join(str(path) for path in paths) + '\n'
output = local / 'TERMINAL_AUDIT_REQUEST.txt'
with output.open('x', encoding='utf-8') as stream:
    stream.write(request)
print(json.dumps(dict(request_path=str(output), actual_files_listed=len(paths),
    auditor_launched=False, executor_audit_interpretation=False)))
