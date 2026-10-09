"""Bind an explicit retained parent only after the prior actual pair is closed."""
import hashlib
import json
from pathlib import Path


runner = Path(__file__).resolve().parent
prior = runner.parents[1] / 'pvground_selected_mask_training_20261009'
assert not (runner / 'pair_spec.json').exists()
wait = json.loads((prior / 'fit_wait.json').read_bytes())
assert wait['observer_closed'] and not wait['terminal']['controller_alive']
assert wait['terminal']['exitcode'] == 0 and wait['terminal']['status']['status'] == 'complete'
recount = json.loads((prior / 'postrun_results/CPU_RECOUNT.json').read_bytes())
assert recount['status'] == 'CLOSED_SELECTED_MASK_CPU_RECOUNT_COMPLETE'
assert recount['rows'] == 9508 and recount['cumulative_support_updates'] == 11169
audit = json.loads((prior / 'postrun_results/EXPERIMENT_AUDIT.json').read_bytes())
assert audit['execution_scope'] == 'ACTUAL_CLOSED_TRAINED_PAIR' and audit['fresh_context'] is True
assert audit['verdict'] in ('PASS', 'WARN') and not audit['blocking_findings']
required = {(prior / name).resolve() for name in
            ('postrun_results/CPU_RECOUNT.json', 'complete_fit/INTAKE.json', 'pair_spec.json')}
assert required.issubset({Path(row['path']).resolve() for row in audit['reviewed_files']})
for row in audit['reviewed_files']:
    assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() == row['sha256']
inspection = json.loads((prior / 'postrun_results/checkpoint_inspection.json').read_bytes())
assert inspection['status'] == 'PASS_CLOSED_CPU_BOTH_TERMINALS_STRICT_RECONSTRUCTION'
assert all(row['full_state_exact'] for row in inspection['weights'])
decision_path = prior / 'postrun_results/RETAINED_PARENT_DECISION.json'
decision = json.loads(decision_path.read_bytes())
assert decision['status'] == 'EXPLICIT_RETAINED_PARENT_DECISION_AFTER_ACTUAL_CLOSURE'
assert decision['current_pair_closed'] and not decision['automatic_promotion']
assert decision['cpu_recount_sha256'] == hashlib.sha256((prior / 'postrun_results/CPU_RECOUNT.json').read_bytes()).hexdigest()
assert decision['checkpoint_inspection_sha256'] == hashlib.sha256((prior / 'postrun_results/checkpoint_inspection.json').read_bytes()).hexdigest()
assert decision['actual_audit_sha256'] == hashlib.sha256((prior / 'postrun_results/EXPERIMENT_AUDIT.json').read_bytes()).hexdigest()
assert Path(decision['local_parent_path']).is_file()
assert hashlib.sha256(Path(decision['local_parent_path']).read_bytes()).hexdigest() == decision['parent_sha256']
template_path = runner / 'pair_spec_template.json'
spec = json.loads(template_path.read_bytes())
assert spec['parent_selection_status'].startswith('PENDING_')
approved = {spec['parent_support_terminal_sha256']: spec['starting_hits']}
approved.update({row['sha256']: recount['table'][row['arm']]['hits'] for row in inspection['weights']})
assert decision['parent_sha256'] in approved and decision['parent_hits'] == approved[decision['parent_sha256']]
rank = lambda hits: (hits[1] >= 4850, hits[0], hits[1])
assert rank(decision['parent_hits']) == max(rank(hits) for hits in approved.values())
spec.update(parent_selection_status='FINALIZED_AFTER_PRIOR_PAIR_CLOSED',
    parent_support_terminal=decision['remote_parent_path'],
    parent_support_terminal_sha256=decision['parent_sha256'], starting_hits=decision['parent_hits'],
    parent_decision_sha256=hashlib.sha256(decision_path.read_bytes()).hexdigest(),
    previous_completed_experiment='/root/autodl-tmp/pvground_selected_mask_training_20261009')
path = runner / 'pair_spec.json'
path.write_text(json.dumps(spec, indent=2, sort_keys=True) + '\n')
receipt = dict(status='FINAL_SPEC_BOUND_TO_EXPLICIT_CLOSED_PARENT_DECISION',
    parent_path=spec['parent_support_terminal'], parent_sha256=spec['parent_support_terminal_sha256'],
    parent_hits=spec['starting_hits'], pair_spec_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    template_sha256=hashlib.sha256(template_path.read_bytes()).hexdigest(),
    parent_decision_sha256=spec['parent_decision_sha256'],
    prior_evidence={name: hashlib.sha256((prior / name).read_bytes()).hexdigest() for name in
        ('fit_wait.json', 'postrun_results/CPU_RECOUNT.json', 'postrun_results/EXPERIMENT_AUDIT.json',
         'postrun_results/checkpoint_inspection.json', 'postrun_results/RETAINED_PARENT_DECISION.json')},
    new_model_builds=0, new_ssh_queries=0, new_weights=0)
(runner / 'FINAL_SPEC_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
