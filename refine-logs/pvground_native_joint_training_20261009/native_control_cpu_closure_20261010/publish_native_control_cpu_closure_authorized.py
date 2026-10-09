"""Publish closed CPU engineering evidence; never inspect the active GPU job."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
prior = json.loads((root / 'native_direct_controls_publication.json').read_bytes())
assert prior['section'] == '20.376.130' and prior['remote_handoff_sync_complete'] is True
assert prior['remote_current_sha256'] == prior['doc_sha256']
assert not (root / 'native_control_cpu_closure_publication.json').exists()
assert not (root / 'NATIVE_CONTROL_CPU_CLOSURE_REMOTE_RECEIPT.json').exists()
closure = json.loads((root / 'ACTUAL_CONTROL_CPU_AUDIT_CLOSURE.json').read_bytes())
assert closure['audit_verdict'] == 'WARN' and closure['blocking_issue_count'] == 0
assert closure['actual_CPU_outcome_review_pending'] is False
assert closure['GPU_or_control_training_admission'] is False
controls = root / 'native_direct_controls_20261010'
audit = json.loads((controls / 'actual_CPU_review/EXPERIMENT_AUDIT.json').read_bytes())
seal = json.loads((controls / 'actual_CPU_review/SEAL.json').read_bytes())
for inputs in (audit['audited_input_hashes'], audit['hash_only_prior_source_inputs'], seal['artifact_hashes']):
    for name, digest in inputs.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest.removeprefix('sha256:')
repos = [Path('C:/Users/gb') / name for name in (
    '.codex_mcln_g0_20260905', '.codex_pvground_cs_20261002', '.codex_mcln_v99_internal_20260928')]
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop = Path('C:/Users/gb/Desktop/document') / Path(doc).name
old = (repos[0] / doc).read_bytes()
assert hashlib.sha256(old).hexdigest() == prior['doc_sha256'] and b'20.376.131' not in old
assert all((repo / doc).read_bytes() == old for repo in repos) and desktop.read_bytes() == old
for index, repo in enumerate(repos):
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip() == prior['heads'][index]
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain']).strip()
notes = (root / 'HANDOFF_NATIVE_CONTROL_CPU_CLOSURE_20261010.md').read_text(encoding='utf-8')
document = old + ('\n\n' + notes.replace('\r\n', '\n')).replace('\n', '\r\n').encode('utf-8')
digest = hashlib.sha256(document).hexdigest()
prefix = 'refine-logs/pvground_native_joint_training_20261009/native_control_cpu_closure_20261010/'
names = [
    'HANDOFF_NATIVE_CONTROL_CPU_CLOSURE_20261010.md', 'ACTUAL_CONTROL_CPU_AUDIT_CLOSURE.json',
    'record_actual_control_cpu_audit_closure.py', 'publish_native_control_cpu_closure_authorized.py',
    'STATIC_COPY_AND_CPU_RECOVERY.json', 'STATIC_CPU_ATTEMPT2_PREPARATION.json',
    'prepare_static_cpu_recovery_attempts.py', 'record_static_copy_and_cpu_recovery.py',
    'read_only_remote_static_diagnostic_authorized.py',
    'run_direct_control_modules_cpu_attempt2_authorized.py',
    'sync_native_direct_control_publication_attempt2_authorized.py',
    'NATIVE_DIRECT_CONTROL_REMOTE_SYNC_ATTEMPT2_RECEIPT.json',
    'NATIVE_DIRECT_CONTROL_REMOTE_SYNC_ATTEMPT2_STDOUT.json',
    'NATIVE_DIRECT_CONTROL_REMOTE_SYNC_ATTEMPT2_STDERR.txt',
    'NATIVE_DIRECT_CONTROL_REMOTE_SYNC_ATTEMPT2_EXIT.json',
    'read_only_static_diagnostic_20261010_attempt1/REMOTE_STATIC_WITNESS.json',
    'read_only_static_diagnostic_20261010_attempt1/TRANSPORT.json',
    'native_direct_controls_20261010/CPU_BUNDLE.json',
    'native_direct_controls_20261010/check_direct_control_modules_cpu.py']
for leaf in (controls / 'actual_CPU_review', controls / 'cpu_transport_attempt2',
             root / '.aris/traces/experiment-audit/2026-10-10_native_controls_actual_CPU'):
    names.extend(path.relative_to(root).as_posix() for path in leaf.rglob('*') if path.is_file())
assert not any(Path(name).name in ('RAW_SSH_STDERR_PRIVATE.txt', 'MEMORY.md', 'SOUL.md', 'USER.md',
    'run_mcln_authorized_20260908.py', 'NativeSshAskPass.exe', 'native_scp_known_hosts') for name in names)
files = {prefix + name: (root / name).read_bytes() for name in sorted(set(names))}
files[prefix + '.gitattributes'] = b'** -text whitespace=-blank-at-eol,-space-before-tab,cr-at-eol,-blank-at-eof\n'
for repo in repos[:2]:
    assert not (repo / prefix).exists()
file_hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}
code = r'''import base64,hashlib,json,sys
from pathlib import Path
b=json.load(sys.stdin);project=Path('/home/gb/new butd/butd_detr-main/MCLN-main');doc=project/b['doc']
assert hashlib.sha256(doc.read_bytes()).hexdigest()==b['old_sha']
leaf=(project/b['prefix']).resolve()
assert leaf==Path('/root/autodl-tmp/mcln_published_evidence_20261005/pvground_native_joint_training_20261009/native_control_cpu_closure_20261010') and not leaf.exists()
for name,data in b['files'].items():
 path=project/name;assert leaf in path.resolve().parents
 path.parent.mkdir(parents=True,exist_ok=True);raw=base64.b64decode(data);path.write_bytes(raw);assert path.read_bytes()==raw
raw=base64.b64decode(b['document']);temporary=doc.with_name(doc.name+'.tmp_native_cpu_closure')
assert not temporary.exists();temporary.write_bytes(raw);assert hashlib.sha256(temporary.read_bytes()).hexdigest()==b['sha'];temporary.replace(doc)
print(json.dumps(dict(doc_sha256=b['sha'],files=len(b['files']),neural_calls=0,training_status_reads=0)))
'''
witness = json.loads((root.parent / 'SCP_TRANSPORT_WITNESS.json').read_bytes())
environment = dict(os.environ, SSH_ASKPASS='C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/NativeSshAskPass.exe',
    SSH_ASKPASS_REQUIRE='force', DISPLAY='codex-byte-transfer')
runtime = '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
argv = ['C:/Windows/System32/OpenSSH/ssh.exe', '-T', '-p', '33476', '-o', 'ProxyCommand=none',
    '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_scp_known_hosts',
    '-o', 'HostKeyAlgorithms=' + witness['negotiated_host_key_algorithm'], '-o', 'NumberOfPasswordPrompts=1',
    'root@region-9.autodl.pro', shlex.join([runtime, '-B', '-u', '-c', code])]
payload = dict(doc=doc, old_sha=prior['doc_sha256'], prefix=prefix, sha=digest,
    document=base64.b64encode(document).decode(),
    files={name: base64.b64encode(raw).decode() for name, raw in files.items()})
response = subprocess.run(argv, env=environment, input=json.dumps(payload).encode(),
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
(root / 'NATIVE_CONTROL_CPU_CLOSURE_REMOTE_STDOUT.json').write_bytes(response.stdout)
(root / 'NATIVE_CONTROL_CPU_CLOSURE_REMOTE_STDERR.txt').write_bytes(response.stderr)
(root / 'NATIVE_CONTROL_CPU_CLOSURE_REMOTE_EXIT.json').write_text(json.dumps(dict(exit_code=response.returncode)) + '\n', encoding='utf-8')
assert response.returncode == 0, response.stderr.decode()
remote = json.loads(response.stdout)
assert remote['doc_sha256'] == digest and remote['files'] == len(files)
remote.update(time_cst=datetime.datetime.now().astimezone().isoformat(), remote_sync_complete=True)
(root / 'NATIVE_CONTROL_CPU_CLOSURE_REMOTE_RECEIPT.json').write_text(json.dumps(remote, indent=2) + '\n', encoding='utf-8')
heads = []
for index, repo in enumerate(repos):
    selected = [doc]
    if index < 2:
        for name, raw in files.items():
            target = repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        selected += sorted(files)
    (repo / doc).write_bytes(document)
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--'] + selected)
    subprocess.check_call(['git', '-C', str(repo), '-c',
        'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--cached', '--check'])
    assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + doc]) == document
    subprocess.check_call(['git', '-C', str(repo), 'commit', '--quiet', '-m',
        'Close bounded native-control CPU engineering review and restore static evidence sync'])
    heads.append(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip())
desktop.write_bytes(document)
local = dict(status='NATIVE_CPU_CLOSURE_ALL_LOCAL_AND_REMOTE_COPIES_COMMITTED_GITHUB_PENDING',
    section='20.376.131', time_cst=datetime.datetime.now().astimezone().isoformat(),
    heads=heads, doc_sha256=digest, prefix=prefix, new_files=len(files),
    file_sha256=file_hashes, remote_handoff_sync_complete=True, remote_evidence_sync_complete=True,
    remote_sync_receipt=str(root / 'NATIVE_CONTROL_CPU_CLOSURE_REMOTE_RECEIPT.json'),
    CPU_outcome_review_completed=True, CPU_audit_verdict='WARN', CPU_bounded_blocking_issue_count=0,
    reviewer_identity='UNATTESTED', review_independence='same-family', acceptance_status='provisional',
    control_training_launched=False, active_training_source_changed=False,
    normal_training_status_queries=0, normal_precision_result=None,
    first_observation_cst='2026-10-10T08:19:04.678479+08:00', full_goal_complete=False)
(root / 'native_control_cpu_closure_local_commit.json').write_text(json.dumps(local, indent=2) + '\n', encoding='utf-8')
subprocess.check_call(['git', '-C', str(repos[0]), '-c', 'http.version=HTTP/1.1', 'push', 'origin', 'HEAD:main'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == heads[0]
assert all((repo / doc).read_bytes() == document for repo in repos) and desktop.read_bytes() == document
for repo in repos[:2]:
    for name, raw in files.items():
        assert (repo / name).read_bytes() == raw
assert all(not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain']).strip() for repo in repos)
guard = root.parent.parent / 'sync_cs_handoff_remote_20260923.py'
raw = guard.read_bytes()
assert prior['doc_sha256'].encode() in raw
guard.write_bytes(raw.replace(prior['doc_sha256'].encode(), digest.encode()))
local.update(status='NATIVE_CPU_CLOSURE_ALL_COPIES_AND_GITHUB_SYNCHRONIZED',
    github_main_verified=True, time_cst=datetime.datetime.now().astimezone().isoformat())
(root / 'native_control_cpu_closure_publication.json').write_text(json.dumps(local, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: local[k] for k in ('status', 'section', 'heads', 'doc_sha256', 'new_files',
    'CPU_outcome_review_completed', 'normal_precision_result', 'full_goal_complete')}), flush=True)
