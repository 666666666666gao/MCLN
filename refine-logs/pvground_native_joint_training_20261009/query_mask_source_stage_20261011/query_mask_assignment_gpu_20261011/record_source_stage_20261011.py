import hashlib
import json
from pathlib import Path

root = Path(__file__).parent
receipt_path = root / 'SOURCE_STAGE_RECEIPT.json'
receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
review_path = root / 'source_stage_review' / 'SOURCE_REVIEW.json'
review = json.loads(review_path.read_text(encoding='utf-8'))
assert receipt['source_stage_review_sha256'] == hashlib.sha256(review_path.read_bytes()).hexdigest()
assert review['blocking_findings'] == []
assert receipt['source_file_count'] == 116
assert receipt['changed_parent_source_files'] == ['main_utils.py', 'models/losses.py']
for key in ('serial_gpu_preflight_admitted', 'full_training_admitted', 'GPU_preflight_executed', 'normal_training_started'):
    assert receipt[key] is False
for key in ('neural_calls', 'optimizer_updates', 'training_status_reads', 'active_training_source_mutations'):
    assert receipt[key] == 0

summary = dict(receipt)
summary['source_stage_receipt_sha256'] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
summary['next_main_observation_cst'] = '2026-10-11T09:07:13.156040+08:00'
summary['current_best_scope'] = 'retained E0 5677/4920, not a new normal-training gain'
summary['source_review_acceptance'] = 'same-family/provisional; actual identity UNATTESTED'
summary['three_effective_contributions_proven'] = False
(root / 'SOURCE_STAGING_SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

state_path = root.parent / 'NORMAL_CONTINUATION_STATE.json'
state = json.loads(state_path.read_text(encoding='utf-8'))
controls = root.parent / 'normal_controls_20261010'
plan = json.loads((controls / 'NORMAL_EPOCH3_OBSERVATION_PLAN.json').read_text(encoding='utf-8'))
owner = json.loads((controls / 'NORMAL_EPOCH3_OBSERVER_OWNER.json').read_text(encoding='utf-8'))
assert plan['due_cst'] == owner['first_due_cst'] == summary['next_main_observation_cst']
assert owner['local_pid'] == 28804
state['current_normal_next_observation_plan'] = plan
state['current_normal_observer'] = owner
state['current_normal_observer_native_session'] = 35461
state['current_normal_observer_pid'] = owner['local_pid']
state['next_observation_cst'] = plan['due_cst']
state['query_mask_assignment_source_staging'] = summary
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

note = '\nPV-Ground ' + receipt['staged_cst'] + ': isolated QueryMask assignment native source staging completed, 116 files/24660643 bytes, only main_utils.py and models/losses.py differ from the bound parent. Receipt SHA ' + summary['source_stage_receipt_sha256'] + '; fresh review WARN/source correctness PASS, 0 blockers, same-family/provisional/UNATTESTED. Both GPU and training admissions remain false: no model/data imports, neural calls, optimizer updates, training-status reads or active-source changes. Shared runtime still required; actual import origins not yet verified. Current C-off E3 continues based on last scheduled observation; sole observer35461/PID28804 next main query09:07:13.156040 CST unchanged. Latest complete E2 5615/4598; retained E0 best5677/4920 is not new normal-training gain. Three effective mechanisms and complete Nr/Sr results unproved; goalACTIVE_UNMET. Public handoff remains section20.376.151 until the next evidence sync.\n'
note += 'Local stage recorder first exited1 on an obsolete top-level continuation-state deadline; no remote action occurred and no observer was started. Corrected the local current observer fields using the existing E3 owner and plan files, retaining the original process and deadline.\n'
memory_path = Path('C:/Users/gb/memory/2026-10-11.md')
assert summary['source_stage_receipt_sha256'] not in memory_path.read_text(encoding='utf-8')
with memory_path.open('a', encoding='utf-8') as handle:
    handle.write(note)
with (root / 'MANIFEST.md').open('a', encoding='utf-8') as handle:
    handle.write('\n| ' + receipt['staged_cst'] + ' | /experiment-bridge | SOURCE_STAGE_RECEIPT.json | source staging | Actual 116-file remote copy/hash/AST checks; GPU and training not admitted |\n')
print(json.dumps({'recorded': True, 'source_staging_complete': True, 'new_training_started': False, 'next_main_observation_cst': summary['next_main_observation_cst']}))
