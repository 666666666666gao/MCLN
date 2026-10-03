"""List completed primary paths for the installed fresh-context audit route."""
import json
from pathlib import Path

local = Path(__file__).parent
assert (local/'complete/INTAKE.json').is_file()
assert (local/'analysis/SUMMARY.json').is_file()
output = local/'TERMINAL_AUDIT_REQUEST.txt'
assert not output.exists()
paths = [local/name for name in ('collect_terminal.py', 'analyze_terminal.py',
    'controller.py', 'launch.json', 'local_range_spec.json', 'whole_range_spec.json',
    'FORMAL_RANGE_CONTROL_PLAN.md', 'FUTURE_METHOD_PLAN.md')]
paths += sorted(path for path in (local/'complete').rglob('*') if path.is_file())
paths += [local/'analysis/SUMMARY.json', local/'analysis/REPORT.md',
    local.parent/'pvground_candidate_consistency_20261003/analyze_complete_initial_qualified.py']
native = local.parent/'pvground_fused_support_20261002/complete_tail_fused_retry/source/imported'
paths += [native/name for name in ('models.pv_ground.py', 'models.losses.py',
    'main_utils.py', 'evaluator.py', 'src.joint_det_dataset.py', 'prepare_data.py')]
assert all(path.is_file() for path in paths)
request = '''You are a fresh experiment integrity auditor. Start adversarially; read primary files directly, trust no executor interpretation. Use the installed experiment-audit same-family/provisional route; do not claim independent backend SKU or cross-family acceptance.

Read every listed evaluation/source/config/result file and its actual saved JSON/NDJSON records. You may run deterministic local CPU recounts on saved evidence. Do not use SSH/CUDA, import/run models, read weights or credentials/auth wrappers, query/restart jobs, modify primary artifacts or implementation, or spawn agents. Write only analysis/EXPERIMENT_AUDIT.md, analysis/EXPERIMENT_AUDIT.json, and identified audit-owned CPU reports.

Audit A-F: A dataset GT provenance (model-derived target must be declared proxy); B metric denominators and prediction-derived normalization; C actual file/metric/claim existence and completed status; D called metric/scorer paths; E actual configurations, seeds, examples, development versus pretrained-seen scope and honest claim ceiling; F evaluation-type classification. Report exact file:line/record references, PASS/WARN/FAIL for each and overall, execution_scope, blocking_issues[], nonblocking_issues[], actual input identities/read scope and limitations. Record review_independence=same-family and acceptance_status=provisional. Do not invent missing executions or optimizer/raw-Mask/full-candidate reconstruction.

Additionally check native bbs/bbf scoring and selected-frame identity, common source-control inputs and fit/update budgets, cross-process starting differences, finite losses and actual fit order, final versus coarse evidence, matched-G training use of GT versus inference inputs, saved rows/summary/receipts/source hashes, and controller retention boundaries/actual deletion receipts. Imported native-source copies must be tied to this run's imports identities before using them as current-source evidence. Review the preservation of protected parents/V99/source/rows and lack of extra failed archives. Specify which CPU checks reconstruct boxes versus simply recount saved native IoUs or Mask values. Report real defects or limitations rather than speculative fallbacks, hashing frameworks, compatibility layers or unrelated refactors.

Files (actual existing paths):
'''
request += '\n'.join(str(path) for path in paths) + '\n'
with output.open('x', encoding='utf-8') as handle:
    handle.write(request)
print(json.dumps(dict(request_path=str(output), actual_files_listed=len(paths),
    executor_audit_interpretation=False)))
