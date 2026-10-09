import json
from pathlib import Path

root = Path(__file__).resolve().parent
controls = root / 'native_direct_controls_20261010'
trace = root / '.aris/traces/experiment-audit/2026-10-10_native_direct_controls_source'
assert not trace.exists()
trace.mkdir(parents=True)
paths = [str(path) for path in sorted((root / 'source').rglob('*.py'))]
paths += [str(path) for path in sorted((controls / 'source').rglob('*.py'))]
paths += [str(path) for path in sorted(controls.rglob('init.json'))]
paths += [str(path) for path in sorted(controls.rglob('NORMAL_NATIVE_RUN_PROTOCOL.json'))]
paths += [str(path) for path in sorted(controls.rglob('normal_joint_controller.py'))]
paths += [str(controls / name) for name in (
    'DIRECT_CONTROL_PREPARATION.json', 'DIRECT_CONTROLS.md',
    'EXPERIMENT_TRACKER.md', 'check_direct_control_modules_cpu.py')]
paths += [str(root / 'prepare_normal_direct_controls.py')]
assert all(Path(path).is_file() for path in paths)
message = '''You are a fresh experiment integrity/source auditor under experiment-audit.
Use gpt-6-astra / max as requested by that skill, but distinguish requested route from
actual attested model identity; when unavailable record actual UNATTESTED, same-family,
acceptance provisional. Independently read every listed artifact. Do not trust author
claims. Audit source correctness, strict payload/state loading, shared native data/score/
criterion/optimization scope, ablation semantics, recorded versus unexecuted evidence,
and the CPU checker coverage/type. Apply the skill's A-F integrity checklist with exact
file:line evidence. Do not treat a synthetic module fixture as dataset accuracy or a full
PV constructor/optimizer/recovery test. Assess whether any unresolved issue must be
fixed before running that CPU checker or later admitting GPU work.

Do not access AUTH, credentials, SSH helpers, MEMORY.md, private profiles, or remote
servers. Do not launch any CPU neural or GPU workload yourself, do not change listed
inputs, and do not poll the current training. You may perform local deterministic source
or hash checks. Output ONLY into the designated review/trace directories, not the model
source. This is a read-only fresh source review, not result approval or permission flow.

Write EXPERIMENT_AUDIT.md, EXPERIMENT_AUDIT.json, and a seal under:
''' + str(controls / 'actual_source_review') + '''
Preserve raw response and metadata in the trace directory:
''' + str(trace) + '''
Machine report: verdict PASS/WARN/FAIL, blocking_issue_count, per-input SHA256,
execution_scope=ISOLATED_NATIVE_CONTROL_SOURCE_NOT_LAUNCHED, requested/actual reviewer
identity, same-family/provisional attribution, and actual checker/formal-results status.

Files to read:
''' + '\n'.join('- ' + path for path in paths) + '''
Skill instructions:
- C:/Users/gb/.codex/skills/experiment-audit/SKILL.md
- C:/Users/gb/.codex/skills/shared-references/local-codex-policy.md
- C:/Users/gb/.codex/skills/shared-references/reviewer-independence.md
- C:/Users/gb/.codex/skills/shared-references/experiment-integrity.md
- C:/Users/gb/.codex/skills/shared-references/review-tracing.md
'''
(trace / 'request.txt').write_text(message, encoding='utf-8')
(trace / 'executor_request_metadata.json').write_text(json.dumps(dict(
    skill='experiment-audit', requested_reviewer='gpt-6-astra', requested_reasoning='max',
    files=paths, purpose='Fresh isolated normal-control source audit',
    current_native_training_queried=False, current_native_training_source_changed=False,
    source_only=True, actual_model_identity='UNATTESTED',
    review_independence='same-family', acceptance_status='provisional'), indent=2) + '\n', encoding='utf-8')
print(str(trace / 'request.txt'))
