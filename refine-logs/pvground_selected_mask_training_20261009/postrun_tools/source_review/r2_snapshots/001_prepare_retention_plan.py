"""Archive inspected terminals and propose a rank; never promote or delete."""
import datetime
import hashlib
import json
from pathlib import Path


local = Path(__file__).resolve().parents[1]
results = local / 'postrun_results'
assert not (results / 'RETENTION_PLAN.json').exists()
wait = json.loads((local / 'fit_wait.json').read_bytes())
assert wait['observer_closed'] and not wait['terminal']['controller_alive']
assert wait['terminal']['exitcode'] == 0 and wait['terminal']['status']['status'] == 'complete'
recount_path = results / 'CPU_RECOUNT.json'
recount = json.loads(recount_path.read_bytes())
assert recount['status'] == 'CLOSED_SELECTED_MASK_CPU_RECOUNT_COMPLETE'
assert recount['rows'] == 9508 and recount['cumulative_support_updates'] == 11169
audit_path = results / 'EXPERIMENT_AUDIT.json'
audit = json.loads(audit_path.read_bytes())
assert audit['execution_scope'] == 'ACTUAL_CLOSED_TRAINED_PAIR' and audit['fresh_context'] is True
assert audit['verdict'] in ('PASS', 'WARN') and not audit['blocking_findings']
required = {(local / name).resolve() for name in
            ('postrun_results/CPU_RECOUNT.json', 'complete_fit/INTAKE.json', 'pair_spec.json')}
assert required.issubset({Path(row['path']).resolve() for row in audit['reviewed_files']})
for row in audit['reviewed_files']:
    assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() == row['sha256']
inspection_path = results / 'checkpoint_inspection.json'
inspection = json.loads(inspection_path.read_bytes())
assert inspection['status'] == 'PASS_CLOSED_CPU_BOTH_TERMINALS_STRICT_RECONSTRUCTION'
assert len(inspection['weights']) == 2 and all(row['full_state_exact'] for row in inspection['weights'])
spec = json.loads((local / 'pair_spec.json').read_bytes())
archive = Path('C:/Users/gb/.codex/archives/pvg_selected_mask_pair_20261009')
assert not archive.exists()
old_archive = Path('C:/Users/gb/.codex/archives/pvg_compressed_support_best_20261008/terminal.pth')
assert hashlib.sha256(old_archive.read_bytes()).hexdigest() == spec['warm_support_terminal_sha256']
choices = [dict(arm='protected_prior', hits=spec['starting_hits'],
    remote_parent_path=spec['warm_support_terminal'], parent_sha256=spec['warm_support_terminal_sha256'],
    local_parent_path=str(old_archive))]
archive.mkdir()
for row in inspection['weights']:
    arm = row['arm']
    assert arm in ('content', 'selected_query')
    source = local / 'complete_fit' / arm / 'terminal.pth'
    data = source.read_bytes()
    assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
    destination = archive / arm / 'terminal.pth'
    destination.parent.mkdir()
    destination.write_bytes(data)
    assert destination.read_bytes() == data
    choices.append(dict(arm=arm, hits=recount['table'][arm]['hits'],
        remote_parent_path=row['path'], parent_sha256=row['sha256'], local_parent_path=str(destination),
        optimizer_preserved=True, full_cpu_state_tensors=1314, total_support_updates=11169))
rank = lambda row: (row['hits'][1] >= 4850, row['hits'][0], row['hits'][1])
chosen = max(choices, key=rank)
plan = dict(status='PROPOSED_RANKED_RETENTION_AFTER_ACTUAL_CLOSURE',
    time_cst=datetime.datetime.now().astimezone().isoformat(), choices=choices, proposed_parent=chosen,
    criterion='strict gate4850, then wide hits, then strict hits; exact ties keep earlier protected parent',
    cpu_recount_sha256=hashlib.sha256(recount_path.read_bytes()).hexdigest(),
    checkpoint_inspection_sha256=hashlib.sha256(inspection_path.read_bytes()).hexdigest(),
    actual_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
    archived_terminals=2, current_best_marker_changed=False, automatic_promotion=False,
    new_trained_weights_created=False, archived_terminal_copies=2, ssh_queries=0, deletions=0)
(archive / 'ARCHIVE_MANIFEST.json').write_text(json.dumps(plan, indent=2) + '\n')
(results / 'RETENTION_PLAN.json').write_text(json.dumps(plan, indent=2) + '\n')
print(json.dumps(plan), flush=True)
