"""List actual completed artifacts for a fresh terminal audit; no interpretation."""
import json
from pathlib import Path


local = Path(__file__).parent
assert (local / 'complete/INTAKE.json').is_file()
assert (local / 'analysis/SUMMARY.json').is_file()
assert not (local / 'TERMINAL_AUDIT_REQUEST.txt').exists()
comparators = local.parent / 'pvground_candidate_consistency_20261003'
paths = [local / name for name in ('collect_complete.py', 'analyze_complete.py',
    'normalized_spec.json', 'launch.json', 'EXPERIMENT_PLAN.md', 'EXPERIMENT_TRACKER.md',
    'WEIGHT_RETENTION.md', 'retire_completed_nonbest.py')]
paths += sorted(path for path in (local / 'complete').rglob('*') if path.is_file())
paths += [local / 'analysis/SUMMARY.json', local / 'analysis/REPORT.md',
    comparators / 'analyze_complete_initial_qualified.py', comparators / 'analysis/SUMMARY.json',
    comparators / 'TERMINAL_EXPERIMENT_AUDIT.json']
for arm in ('g_control', 'g_consistent'):
    directory = comparators / 'complete' / arm
    paths += [directory / name for name in ('receipt.json', 'train.jsonl', 'spec.json')]
    for phase in ('initial', 'terminal', 'formal'):
        paths += [directory / phase / name for name in ('rows.jsonl', 'receipt.json')]
assert all(path.is_file() for path in paths)
request = '''You are a fresh experiment-integrity auditor. Read actual listed sources and terminal artifacts directly; trust no executor interpretation. This is the installed experiment-audit same-family/provisional route, not cross-family acceptance.

Read source and saved JSON/NDJSON. You may run deterministic local CPU recounts on saved records, but do not import/run models, use CUDA/SSH, read binary weights or credentials/auth wrappers, query jobs, alter implementation/primary results, or spawn agents. Write only analysis/EXPERIMENT_AUDIT.md, analysis/EXPERIMENT_AUDIT.json, and clearly identified audit-owned CPU reports. No new training.

Check real dataset ground-truth provenance, native metric denominator and scorer calls, saved-record/summary/receipt agreement, actual train/evaluation coverage and updates, starting-input/supervision/control comparability, denominator N vs N+A and effective scale, real complete/exit/source identity, retained evidence and weight recommendation, and any narrow evidence used for broad claims. Review the prepared retire_completed_nonbest.py: it must target only this one completed normalized endpoint if original G is metric leader, preserve original G/parents/V99/logs, avoid new archives, and keep a remote receipt recoverable after observation loss. Verify actual reporting scope and disclose missing independent all-Mask/full-256 reconstruction or optimizer replay.

Report A-F PASS/WARN/FAIL with precise source/record references: GT provenance, prediction-derived score normalization, result existence, dead-code use, scope, and evaluation type. Overall JSON must include verdict (PASS/WARN/FAIL), execution_scope, blocking_issues (list), nonblocking_issues (list), actual file identities/read scope, checks, claims and review_independence=same-family / acceptance_status=provisional. Do not invent backend-SKU independence. Any executed CPU recount must name its actual inputs/results and limits; saved IoU recount is not independent raw-mask reconstruction. Report concrete scientific or deletion-boundary defects; no speculative fallback, compatibility, retry, hashing framework or unrelated refactor. State plainly when correct.

Files (actual existing paths):
'''
request += '\n'.join(str(path) for path in paths) + '\n'
with (local / 'TERMINAL_AUDIT_REQUEST.txt').open('x', encoding='utf-8') as handle:
    handle.write(request)
print(json.dumps(dict(actual_files_listed=len(paths), request_path=str(local / 'TERMINAL_AUDIT_REQUEST.txt'),
    result_interpretation_performed=False)))
