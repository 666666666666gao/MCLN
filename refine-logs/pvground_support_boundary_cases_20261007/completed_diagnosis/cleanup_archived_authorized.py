"""Clear only the191 closed, locally verified diagnostic arrays."""
import hashlib
import json
import os
from pathlib import Path
import shlex

import paramiko

root = Path(__file__).resolve().parent
assert not (root / 'array_cleanup_receipt.json').exists()
review = json.loads((root / 'CLEANUP_SOURCE_REVIEW.json').read_bytes())
assert review['execution_scope'] == 'SOURCE_ONLY' and review['verdict'] in ('PASS', 'WARN') and not review['blocking_findings']
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
wait = json.loads((root / 'wait.json').read_bytes())
assert wait['observer_closed'] and not wait['terminal']['controller_alive'] and wait['terminal']['exitcode'] == 0
summary = json.loads((root / 'analysis/SUMMARY.json').read_bytes())
assert summary['cases'] == 191 and summary['optimizer_updates'] == summary['weights_created'] == 0
# Remote-copy retirement requires closure and the complete verified local archive.
# Scientific publication retains its separate actual-evidence review gate.
spec = json.loads((root / 'diagnostic_spec.json').read_bytes())
policy_path = root.parent / 'pvground_face_support_20261007/CLEANUP_POLICY.json'
policy = json.loads(policy_path.read_bytes())
assert policy['future_repeated_approval_required'] is False
intake = json.loads((root / 'complete/INTAKE.json').read_bytes())
items = [item for item in intake['files'] if item['name'].endswith('.npz')]
cases = json.loads((root / 'case_manifest.json').read_bytes())['diagnostic_rows']
expected = {'arrays/row_%05d.npz' % item['cached']['row_id'] for item in cases}
assert len(items) == 191 and {item['name'] for item in items} == expected
for item in items:
    path = root / 'complete' / item['name']
    assert path.stat().st_size == item['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
code = '''import datetime,hashlib,json,shutil,sys
from pathlib import Path
b=json.load(sys.stdin);root=Path(b['root'])
assert root==Path('/root/autodl-tmp/pvground_support_boundary_cases_20261007') and root.resolve()==root
assert (root/'controller.exit').read_text().strip()=='0'
assert json.loads((root/'status.json').read_bytes())['status']=='complete'
assert json.loads((root/'receipt.json').read_bytes())['protected_weight_chain_exact']
spec=json.loads((root/'diagnostic_spec.json').read_bytes());parent=Path(spec['selected_terminal'])
assert hashlib.sha256(parent.read_bytes()).hexdigest()==spec['selected_terminal_sha256']
items=b['files'];assert len(items)==191 and len({i['name'] for i in items})==191
paths=[]
for item in items:
    rel=Path(item['name']);assert len(rel.parts)==2 and rel.parts[0]=='arrays' and rel.suffix=='.npz'
    path=root/rel;assert path.resolve()==path and root in path.parents
    raw=path.read_bytes();assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
    paths.append(path)
before=shutil.disk_usage(root).free
for path in paths:path.unlink()
assert all(not path.exists() for path in paths)
assert hashlib.sha256(parent.read_bytes()).hexdigest()==spec['selected_terminal_sha256']
r=dict(status='STANDING_AUTHORIZED_CLOSED_DIAGNOSTIC_ARRAYS_REMOVED',time_cst=datetime.datetime.now().astimezone().isoformat(),
    deleted_count=191,released_file_bytes=sum(i['bytes'] for i in items),local_archive_preserved=True,
    local_archive=b['archive'],policy_sha256=b['policy_sha256'],best_weights_touched=0,datasets_touched=0,text_logs_touched=0,
    free_bytes_before=before,free_bytes_after=shutil.disk_usage(root).free,deleted_files=items)
(root/'array_cleanup_receipt.json').write_text(json.dumps(r,indent=2)+'\\n')
print(json.dumps(r))
'''
client = paramiko.SSHClient(); client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
stdin, stdout, stderr = client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python', '-B', '-c', code]), timeout=180)
payload = dict(root=spec['root'], files=items, archive=str(root / 'complete'), policy_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest())
stdin.write(json.dumps(payload).encode()); stdin.flush(); stdin.channel.shutdown_write()
raw = stdout.read(); assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
receipt = json.loads(raw)
assert receipt['deleted_count'] == 191 and receipt['best_weights_touched'] == 0
(root / 'array_cleanup_receipt.json').write_bytes(raw)
client.close()
print(json.dumps({key:value for key,value in receipt.items() if key != 'deleted_files'}), flush=True)
