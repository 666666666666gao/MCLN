"""Publish actual engineering closure and original fit launch, never fit progress."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

import paramiko


root = Path(__file__).resolve().parent
results = root / 'preflight_results'
workspace = Path('C:/Users/gb')
repos = [workspace / '.codex_mcln_g0_20260905', workspace / '.codex_pvground_cs_20261002',
         workspace / '.codex_mcln_v99_internal_20260928']
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop = workspace / 'Desktop/document' / Path(doc).name
prior = json.loads((root.parents[1] / 'pvground_selected_mask_training_20261009/postrun_results/publication.json').read_bytes())
assert not (results / 'publication.json').exists()
old = (repos[0] / doc).read_bytes()
assert hashlib.sha256(old).hexdigest() == prior['doc_sha256'] and b'20.376.123' not in old
assert all((repo / doc).read_bytes() == old for repo in repos) and desktop.read_bytes() == old
for index, repo in enumerate(repos):
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip() == prior['heads'][index]
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain']).strip()
audit = json.loads((results / 'EXPERIMENT_AUDIT.json').read_bytes())
assert audit['execution_scope'] == 'ACTUAL_PREFLIGHT' and audit['fresh_context']
assert audit['verdict'] in ('PASS', 'WARN') and not audit['blocking_findings']
for row in audit['reviewed_files']:
    assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() == row['sha256']
intake = json.loads((root / 'preflight_complete/INTAKE.json').read_bytes())
assert intake['status'] == 'CLOSED_SPAN_PHASE_ARTIFACTS_COLLECTED'
fit = json.loads((root / 'fit_launch.json').read_bytes())
observer = json.loads((root / 'fit_observer_started.json').read_bytes())
assert fit['status'] == 'FIT_STARTED_NOT_COMPLETED'
assert observer['controller_pid'] == fit['controller_pid'] and observer['remote_queries_performed'] == 0
notes = (results / 'HANDOFF_NOTES.md').read_text(encoding='utf-8')
document = old + ('\n\n' + notes).replace('\n', '\r\n').encode('utf-8')
digest = hashlib.sha256(document).hexdigest()
prefix = 'refine-logs/pvground_extremal_span_evidence_20261009/runner_v1/actual_preflight/'
files = {}
for name in ('EXPERIMENT_AUDIT.json', 'EXPERIMENT_AUDIT.md', 'HANDOFF_NOTES.md'):
    files[prefix + name] = (results / name).read_bytes()
for name in ('pair_spec.json', 'FINAL_SPEC_RECEIPT.json', 'preflight_launch.json', 'preflight_wait.json',
             'fit_launch.json', 'fit_admission.stdout.json', 'fit_admission.exit',
             'fit_observer_started.json', 'fit_observer_spawn.json', 'PREFLIGHT_ACTUAL_REVIEW_REQUEST.txt',
             'publish_actual_preflight_authorized.py', 'update_continuation.py'):
    files[prefix + name] = (root / name).read_bytes()
for name in ('INTAKE.json', 'preflight.json', 'preflight_status.json', 'preflight_controller.exit',
             'preflight.exit', 'imports.json', 'load.json'):
    files[prefix + 'collected/' + name] = (root / 'preflight_complete' / name).read_bytes()
files[prefix + '.gitattributes'] = b'** -text whitespace=blank-at-eol,space-before-tab,cr-at-eol,-blank-at-eof\n'
assert all(not (repos[0] / name).exists() for name in files)
code = r'''import base64,hashlib,json,sys
from pathlib import Path
b=json.load(sys.stdin);project=Path('/home/gb/new butd/butd_detr-main/MCLN-main');doc=project/b['doc']
assert hashlib.sha256(doc.read_bytes()).hexdigest()==b['old_sha']
leaf=(project/b['prefix']).resolve()
assert leaf==Path('/root/autodl-tmp/mcln_published_evidence_20261005/pvground_extremal_span_evidence_20261009/runner_v1/actual_preflight') and not leaf.exists()
for name,data in b['files'].items():
 path=project/name;assert leaf in path.resolve().parents
 path.parent.mkdir(parents=True,exist_ok=True);raw=base64.b64decode(data);path.write_bytes(raw);assert path.read_bytes()==raw
raw=base64.b64decode(b['document']);temp=doc.with_name(doc.name+'.tmp_span_actual_preflight_20261009')
assert not temp.exists();temp.write_bytes(raw);assert hashlib.sha256(temp.read_bytes()).hexdigest()==b['sha'];temp.replace(doc)
print(json.dumps(dict(doc_sha256=b['sha'],files=len(b['files']),neural_forwards=0,training_status_reads=0,deletions=0)))
'''
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
stdin, stdout, stderr = client.exec_command(shlex.join([
    '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python', '-B', '-c', code]), timeout=120)
stdin.write(json.dumps(dict(doc=doc, old_sha=prior['doc_sha256'], prefix=prefix, sha=digest,
    document=base64.b64encode(document).decode(),
    files={name: base64.b64encode(data).decode() for name, data in files.items()})))
stdin.channel.shutdown_write()
raw, error = stdout.read(), stderr.read()
status = stdout.channel.recv_exit_status()
(results / 'PUBLICATION_REMOTE_STDOUT.json').write_bytes(raw)
(results / 'PUBLICATION_REMOTE_STDERR.txt').write_bytes(error)
(results / 'PUBLICATION_REMOTE_EXIT.json').write_text(json.dumps(dict(exit_code=status)) + '\n')
client.close()
assert status == 0, error.decode()
heads = []
for index, repo in enumerate(repos):
    selected = [doc]
    if index < 2:
        for name, data in files.items():
            path = repo / name
            assert not path.exists()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        selected += sorted(files)
    (repo / doc).write_bytes(document)
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--'] + selected)
    subprocess.check_call(['git', '-C', str(repo), '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--cached', '--check'])
    assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + doc]) == document
    subprocess.check_call(['git', '-C', str(repo), 'commit', '--quiet', '-m', 'Verify actual span preflight and launch controlled geometry training'])
    heads.append(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip())
desktop.write_bytes(document)
subprocess.check_call(['git', '-C', str(repos[0]), '-c', 'http.version=HTTP/1.1', 'push', 'origin', 'HEAD:main'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == heads[0]
guard = root.parents[1] / 'sync_cs_handoff_remote_20260923.py'
data = guard.read_bytes()
assert prior['doc_sha256'].encode() in data
guard.write_bytes(data.replace(prior['doc_sha256'].encode(), digest.encode()))
record = dict(status='ACTUAL_SPAN_PREFLIGHT_AUDITED_FORMAL_FIT_STARTED_PUBLISHED',
    section='20.376.123', time_cst=datetime.datetime.now().astimezone().isoformat(), heads=heads,
    doc_sha256=digest, new_files=len(files), actual_m0_review_verdict=audit['verdict'],
    formal_training_started=True, formal_training_completed=False,
    first_observation_cst=fit['first_observation_cst'], parent_hits=[5606,4881],
    new_accuracy_result=False, three_effective_contributions=False, full_goal_complete=False)
(results / 'publication.json').write_text(json.dumps(record, indent=2) + '\n')
state_path = root.parents[1] / 'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state = json.loads(state_path.read_bytes())
state.update(published_heads=heads, handoff_sha256=digest, latest_handoff_section=record['section'],
    latest_publication=str(results / 'publication.json'), full_goal_status='ACTIVE_UNMET')
state['extremal_span_preparation']['actual_preflight_publication'] = record
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record), flush=True)
