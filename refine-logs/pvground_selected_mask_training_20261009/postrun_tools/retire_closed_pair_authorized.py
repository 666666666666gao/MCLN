"""Delete only this closed pair's unselected terminals and verified local arrays."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex

import paramiko


local = Path(__file__).resolve().parents[1]
results = local / 'postrun_results'
assert not (results / 'RETIREMENT_RECEIPT.json').exists()
review = json.loads((local / 'postrun_tools/source_review/RETENTION_SOURCE_REVIEW.json').read_bytes())
assert review['execution_scope'] == 'SOURCE_ONLY' and review['verdict'] in ('PASS', 'WARN')
assert not review['blocking_findings']
required = {(local / 'postrun_tools' / name).resolve() for name in
            ('prepare_retention_plan.py', 'retire_closed_pair_authorized.py')}
assert required.issubset({Path(row['path']).resolve() for row in review['reviewed_files']})
for row in review['reviewed_files']:
    assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() == row['sha256']
decision = json.loads((results / 'RETAINED_PARENT_DECISION.json').read_bytes())
assert decision['status'] == 'EXPLICIT_RETAINED_PARENT_DECISION_AFTER_ACTUAL_CLOSURE'
assert decision['current_pair_closed'] and not decision['automatic_promotion']
plan = json.loads((results / 'RETENTION_PLAN.json').read_bytes())
assert plan['status'] == 'PROPOSED_RANKED_RETENTION_AFTER_ACTUAL_CLOSURE'
assert all(decision[key] == plan['proposed_parent'][key] for key in
           ('parent_sha256', 'remote_parent_path', 'local_parent_path'))
assert decision['parent_hits'] == plan['proposed_parent']['hits']
for key, name in (('cpu_recount_sha256', 'CPU_RECOUNT.json'),
                  ('checkpoint_inspection_sha256', 'checkpoint_inspection.json'),
                  ('actual_audit_sha256', 'EXPERIMENT_AUDIT.json')):
    assert decision[key] == plan[key] == hashlib.sha256((results / name).read_bytes()).hexdigest()
inspection = json.loads((results / 'checkpoint_inspection.json').read_bytes())
assert inspection['status'] == 'PASS_CLOSED_CPU_BOTH_TERMINALS_STRICT_RECONSTRUCTION'
audit = json.loads((results / 'EXPERIMENT_AUDIT.json').read_bytes())
assert audit['execution_scope'] == 'ACTUAL_CLOSED_TRAINED_PAIR' and audit['fresh_context'] is True
assert audit['verdict'] in ('PASS', 'WARN') and not audit['blocking_findings']
for row in audit['reviewed_files']:
    assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() == row['sha256']
wait = json.loads((local / 'fit_wait.json').read_bytes())
assert wait['observer_closed'] and not wait['terminal']['controller_alive']
assert wait['terminal']['exitcode'] == 0 and wait['terminal']['status']['status'] == 'complete'
assert hashlib.sha256(Path(decision['local_parent_path']).read_bytes()).hexdigest() == decision['parent_sha256']
spec = json.loads((local / 'pair_spec.json').read_bytes())
intake = json.loads((local / 'complete_fit/INTAKE.json').read_bytes())
assert intake['status'] == 'CLOSED_FIT_ARTIFACTS_COLLECTED' and intake['weights_copied'] == 2
by_name = {row['name']: row for row in intake['files']}
chosen = [row for row in intake['files'] if re.fullmatch(r'formal/batch_[0-9]{5}\.npz', row['name'])]
assert len(chosen) == 1189
assert {row['name'] for row in chosen} == {'formal/batch_%05d.npz' % offset for offset in range(0, 9508, 8)}
for choice in plan['choices'][1:]:
    if choice['parent_sha256'] != decision['parent_sha256']:
        name = choice['arm'] + '/terminal.pth'
        row = by_name[name]
        assert row['sha256'] == choice['parent_sha256']
        assert hashlib.sha256(Path(choice['local_parent_path']).read_bytes()).hexdigest() == row['sha256']
        chosen.append(row)
for row in chosen:
    path = local / 'complete_fit' / row['name']
    assert path.stat().st_size == row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
assert all(spec['root'] + '/' + row['name'] != decision['remote_parent_path'] for row in chosen)
archive_manifest = Path('C:/Users/gb/.codex/archives/pvg_selected_mask_pair_20261009/ARCHIVE_MANIFEST.json')
assert json.loads(archive_manifest.read_bytes()) == plan
manifest = dict(remote_root=spec['root'], files=chosen,
    retained_parent_path=decision['remote_parent_path'], retained_parent_sha256=decision['parent_sha256'],
    original_protected_prior_path=spec['warm_support_terminal'],
    original_protected_prior_sha256=spec['warm_support_terminal_sha256'],
    archives_verified=True, closed_consumers=True, deletion_scope='only1189 formalNPZ and this pair unselected terminals')
(results / 'RETIREMENT_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
code = r'''import hashlib,json,shutil,sys
from pathlib import Path
b=json.load(sys.stdin);root=Path(b['remote_root'])
assert root==Path('/root/autodl-tmp/pvground_selected_mask_training_20261009') and root.resolve()==root
assert (root/'fit_controller.exit').read_text().strip()=='0'
assert json.loads((root/'fit_status.json').read_bytes())['status']=='complete'
protected={Path(b['retained_parent_path']).resolve():b['retained_parent_sha256'],
           Path(b['original_protected_prior_path']).resolve():b['original_protected_prior_sha256']}
assert all(hashlib.sha256(path.read_bytes()).hexdigest()==digest for path,digest in protected.items())
paths=[]
for row in b['files']:
 path=(root/row['name']).resolve();assert root in path.parents and path not in protected
 assert len(path.read_bytes())==row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'];paths.append(path)
before=shutil.disk_usage(root).free
for path in paths:path.unlink()
assert all(hashlib.sha256(path.read_bytes()).hexdigest()==digest for path,digest in protected.items())
print(json.dumps(dict(status='CLOSED_PAIR_REDUNDANT_FILES_RETIRED',files_deleted=len(paths),
 bytes_removed=sum(row['bytes'] for row in b['files']),data_free_before=before,data_free_after=shutil.disk_usage(root).free,
 original_protected_prior_unchanged=True,retained_parent_unchanged=True,
 logs_rows_receipts_preserved=True,new_neural_forwards=0,optimizer_updates=0)))
'''
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
stdin, stdout, stderr = client.exec_command(shlex.join([spec['runtime'] + '/venv/bin/python', '-B', '-c', code]), timeout=120)
stdin.write(json.dumps(manifest))
stdin.channel.shutdown_write()
raw, error = stdout.read(), stderr.read()
exit_code = stdout.channel.recv_exit_status()
(results / 'RETIREMENT_STDOUT.json').write_bytes(raw)
(results / 'RETIREMENT_STDERR.txt').write_bytes(error)
(results / 'RETIREMENT_EXIT.json').write_text(json.dumps(dict(exit_code=exit_code)) + '\n')
client.close()
assert exit_code == 0, error.decode()
record = json.loads(raw)
record['time_cst'] = datetime.datetime.now().astimezone().isoformat()
(results / 'RETIREMENT_RECEIPT.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
