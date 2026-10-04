"""List the actual closed face experiment's primary artifacts for a fresh auditor."""
import json
from pathlib import Path

local = Path(__file__).resolve().parent
assert (local / 'complete/INTAKE.json').is_file() and (local / 'analysis/SUMMARY.json').is_file()
assert not (local / 'TERMINAL_AUDIT_REQUEST.txt').exists()
paths = [local / name for name in ('collect_terminal.py', 'analyze_terminal.py', 'controller.py',
    'launch.json', 'face_fit_spec.json', 'EXPERIMENT_PLAN.md', 'run_face_fit.py',
    'pvground_face_conditioned_box_refiner.py', 'pvground_boundary_box_refiner.py',
    'pvground_whole_mask_box_refiner.py', 'whole_mask_range.py', 'pvground_tail_support_box_refiner.py',
    'pvground_semantic_assignment.py', 'PREFLIGHT_ANALYSIS.json', 'launch_publication.json')]
paths += sorted(path for path in (local / 'complete').rglob('*') if path.is_file())
paths += [local / 'analysis/SUMMARY.json', local / 'analysis/REPORT.md']
control = local.parent / 'pvground_boundary_distribution_20261004'
paths += [control / 'run_boundary_fit.py', control / 'distribution_spec.json']
for stage in ('initial', 'terminal', 'formal'):
    paths += [control / 'complete/distribution' / stage / name for name in ('receipt.json', 'rows.jsonl')]
native = local.parent / 'pvground_fused_support_20261002/complete_tail_fused_retry/source/imported'
paths += [native / name for name in ('models.pv_ground.py', 'models.losses.py', 'main_utils.py',
    'evaluator.py', 'src.joint_det_dataset.py', 'prepare_data.py')]
assert all(path.is_file() for path in paths), [str(path) for path in paths if not path.is_file()]
request = '''You are a fresh experiment integrity auditor. Read primary files directly, not executor interpretations. Use the installed experiment-audit same-family/provisional route; requested Astra/max backend and effort are not independently attested.

Read all listed sources/configurations/receipts and parse actual evaluation JSON/NDJSON records. Deterministic local CPU recounts are allowed. No SSH/CUDA/model imports or executions, weight reads, private memory, goals/credentials/auth wrappers, job queries/restarts, primary artifact changes, or agents. Write only analysis/EXPERIMENT_AUDIT.md, analysis/EXPERIMENT_AUDIT.json and identified audit-owned CPU reports.

Audit A-F: A dataset GT versus model-derived proxies; B metric denominators/prediction normalization; C actual files, numbers, runtime closure and claim existence; D called metric/scoring paths; E seed/configuration/scope/claim ceiling; F evaluation-type classification. Give precise file:line/record references, per-item and overall PASS/WARN/FAIL, execution_scope, blocking_issues[], nonblocking_issues[], actual read identities and limitations. Record review_independence=same-family, acceptance_status=provisional; no cross-family or backend-SKU claim.

Verify original G fixed/eval parameters and buffers; only the64737-parameter/25-tensor face head learned. Compare the456102-parameter flat control under the actual common start/budget: architecture and capacity change together, not a pure direction-conditioning effect. Check actual producer109-D schema, axis/face order, face local-member mapping, nonuniform33-bin distribution and final-box decode/floor, native final Hungarian responsibilities, DFL/7 and finite/outside target handling, final native box/GIoU losses. Entropy is an uncalibrated model statistic.

Check29778 fit rows exactly once/B8-tail2/3723 updates, row order, finite losses, source SHA declarations/import witnesses and cross-process initial differences. Audit actual unique native last/bbs scoring and same-Query Box/Mask under the documented two-stage/object-input protocol; distinguish retained native assistance conventions from added GT inference gates. All256 candidates remain; GT oracle/rank diagnostics must not enter deployed selection. Distinguish selected-box CPU reconstruction from saved scalar Mask/full-candidate-oracle recount, and internal same-Query effects from independently trained control differences.

Verify complete9508 formal restore/closure, no metric stitching, actual controller retention/delete receipts and protected current-best/required parent/V99 chains. No weight was downloaded and no failed weight archive was created. This audit cannot imply raw Mask/full candidate-box or model/optimizer replay if not performed. No new readback, Nr3D/Sr3D, teacher or quality-loss result is supported. Report actual defects and observed scope/numerical limits, not hypothetical defensive changes.

Actual existing primary files:
'''
request += '\n'.join(str(path) for path in paths) + '\n'
(local / 'TERMINAL_AUDIT_REQUEST.txt').write_text(request, encoding='utf-8')
print(json.dumps(dict(request_path=str(local / 'TERMINAL_AUDIT_REQUEST.txt'),
    actual_files_listed=len(paths), auditor_launched=False)))
