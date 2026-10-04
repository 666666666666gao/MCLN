"""Create a concrete fresh-source review request, before any deployment."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import shutil

local = Path(__file__).resolve().parent
assert not (local / 'CODE_REVIEW_REQUEST.txt').exists()
reference = local.parent / 'pvground_six_face_design_20261004'
(local / 'references').mkdir(exist_ok=True)
for name in ('SOURCE_IDENTITIES.json','INTERFACE_NOTES.md','CPU_PARAMETERIZATION_CHECK.json'):
    shutil.copyfile(reference/name, local/'references'/name)
shutil.copyfile(reference/'reference/LICENSE', local/'references/DFINE_LICENSE')
contract = local/'idea-stage/docs/research_contract.md'
contract.parent.mkdir(parents=True)
contract.write_text('''# Research contract: next boundary control\n
Question: with preserved original G and identical whole/local predicted support,
does an explicit six-face offset distribution plus independently normalized
boundary supervision improve deployed native bbs ScanRefer Acc@.50 over direct
center/size residuals and original G4495 strict hits?\n
This is a controlled package, not a completed direction-conditioned face decoder
or a novel three-module system. Representation, extra supervision and output
parameter count change together; claims must say so. All GT belongs to native
training targets/evaluation only. Keep256 candidates and one native final score.
No teacher, quality readback, dataset-ID gate, dual ranking or validation fitting.
6887 modules have pretrained-seen scenes;9508 is development validation, not
untouched final test. One seed2027; same data/order/effectivebatch8/3723 updates.
Engineering pass never proves accuracy. Native same-query repairs/damages,
size-floor/target-saturation and actual formal metrics decide the next step.
Only source review plus actual two-step probes authorizes bounded fit; original
G remains protected and metric best until demonstrated otherwise. Final goal
requires same-checkpoint5615/4754 and independent Nr/Sr protocol-correct training.
''', encoding='utf-8')
(local/'EXPERIMENT_TRACKER.md').write_text('''# Boundary comparison tracker\n
| Stage | Status | Actual result |\n|---|---|---|\n| Fresh source review | not requested yet | none |\n| Residual real two-update probe | not launched | none |\n| Distribution real two-update probe | not launched | none |\n| Residual3723 updates/formal9508 | not launched | none |\n| Distribution3723 updates/formal9508 | not launched | none |\n
This tracker is a preparation snapshot; actual receipts supersede it.\n''',encoding='utf-8')
paths = sorted(p for p in local.rglob('*') if p.is_file() and '.aris' not in p.parts)
historical = local.parent/'pvground_range_head_only_20261004'
paths += [historical/'analysis/REPORT.md',historical/'NEXT_EXPERIMENT_DECISION.md']
paths += list(sorted((local.parent/'pvground_fused_support_20261002/complete_tail_fused_retry/source/imported').glob('*.py')))
for path in local.glob('*.py'):
    ast.parse(path.read_text(encoding='utf-8'), feature_version=(3,7) if path.name not in ('prepare_source.py','prepare_control_tools.py','prepare_code_review.py') else None)
request = '''Perform the installed experiment-bridge fresh code review, source-only.
Read every exact existing primary path below; no SSH, CUDA, model imports,
weight reads, auth wrapper, private goal reads, job changes, or sub-agents.
Check the implementation against EXPERIMENT_PLAN and native GT/matching/loss/
evaluation calls. In particular: real final Hungarian index order; GT filtering;
nonuniform interpolation and endpoint saturation; zero-head common-floor reference;
positive final size semantics; DFL-only task gradient; unchanged1302 support;
frozen original state; delta/AdamW restoration; actual sample/update budget;
controlled extra loss/parameter count claims; no GT leakage or candidate pruning;
GPU lock/probe prerequisites, owned-only retention and authorized transfers.
This is a representation+supervision package control, not a fully conditioned
six-face decoder or established novel method. Do not invent findings. Report
existing defects including rare cases that the current sources actually produce.
Keep fixes minimal; no speculative fallback/compat/extra exception logic. Do not
propose additional digest schemes, environment upgrades, unrelated refactors or
hyperparameter sweeps. Existing identity checks are inherited invariants.
Write only your review-owned files EXPERIMENT_CODE_REVIEW.md and JSON in this
new directory. JSON must include verdict PASS/WARN/BLOCKED; blocking_findings,
nonblocking_findings arrays; execution_scope SOURCE_ONLY; reviewed_files list
of actual path/bytes/sha256; review_independence same-family, acceptance_status
provisional, backend_sku_independently_verified false. Review CPU/source scope
does not satisfy the actual real-model probe. Report the actual result when done.
Primary paths:\n''' + '\n'.join(str(p) for p in paths) + '\n'
(local/'CODE_REVIEW_REQUEST.txt').write_text(request,encoding='utf-8')
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
    status='ACTUAL_REQUEST_PREPARED_REVIEWER_NOT_SPAWNED', primary_paths=len(paths),
    request_sha256=hashlib.sha256((local/'CODE_REVIEW_REQUEST.txt').read_bytes()).hexdigest(),
    local_py37_AST_checks=True, model_forwards=0, optimizer_steps=0, remote_jobs=0)
(local/'code_review_preparation.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record))
