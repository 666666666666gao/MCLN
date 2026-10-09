"""Finish the observed whitespace-only publication interruption, once."""
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
home = Path('C:/Users/gb')
repos = [home / '.codex_mcln_g0_20260905', home / '.codex_pvground_cs_20261002',
         home / '.codex_mcln_v99_internal_20260928']
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop = home / 'Desktop/document' / Path(doc).name
prior = json.loads((root.parent / 'pvground_extremal_span_evidence_20261009/runner_v1/preflight_results/publication.json').read_bytes())
prefix = 'refine-logs/pvground_native_joint_training_20261009/source_integration_r2/'
assert not (root / 'publication.json').exists()
document = (repos[0] / doc).read_bytes()
digest = hashlib.sha256(document).hexdigest()
old = (repos[1] / doc).read_bytes()
assert hashlib.sha256(old).hexdigest() == prior['doc_sha256']
assert (repos[2] / doc).read_bytes() == old and desktop.read_bytes() == old
assert document == old + ('\n\n' + (root / 'HANDOFF_NOTES.md').read_text(encoding='utf-8')).replace('\n', '\r\n').encode('utf-8')
for index, repo in enumerate(repos):
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip() == prior['heads'][index]
    if index:
        assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain']).strip()
names = subprocess.check_output(['git', '-C', str(repos[0]), 'diff', '--cached', '--name-only']).decode().splitlines()
assert doc in names and all(name == doc or name.startswith(prefix) for name in names)
assert subprocess.check_output(['git', '-C', str(repos[0]), 'show', ':' + doc]) == document
files = {name: (repos[0] / name).read_bytes() for name in names if name != doc}
for name, raw in files.items():
    relative = name[len(prefix):]
    if relative != '.gitattributes':
        assert (root / relative).read_bytes() == raw
receipt = json.loads((root / 'RESUME_CORRECTION_R2.json').read_bytes())
for name, expected in receipt['source_files'].items():
    assert hashlib.sha256((root / 'source' / name).read_bytes()).hexdigest() == expected
review = json.loads((root / 'source_review/SOURCE_REVIEW_R2.json').read_bytes())
assert review['execution_scope'] == 'SOURCE_ONLY' and review['verdict'] in ('PASS', 'WARN')
assert not review['blocking_findings']
assert json.loads((root / 'PUBLICATION_REMOTE_EXIT.json').read_bytes())['exit_code'] == 0
failed = subprocess.run(['git', '-C', str(repos[0]), '-c',
    'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol',
    'diff', '--cached', '--check'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
assert failed.returncode == 2
(root / 'PUBLICATION_FAILED_DIFF_CHECK.txt').write_bytes(failed.stdout + failed.stderr)
failure = dict(status='REMOTE_DOCUMENT_AND_EVIDENCE_SYNCED_MAIN_STAGED_NO_COMMITS_WHITESPACE_CHECK_FAILED',
    native_publisher_exit_code=1, git_diff_check_exit_code=2,
    time_cst=datetime.datetime.now().astimezone().isoformat(),
    original_publisher_sha256=hashlib.sha256((root / 'publish_native_sources_authorized.py').read_bytes()).hexdigest(),
    reason='Verbatim original reviewed source contains existing trailing spaces; preserve computational bytes and correct the evidence-leaf whitespace attribute',
    computational_source_bytes_changed=False, neural_calls=0, training_status_reads=0)
(root / 'PUBLICATION_FAILURE_R1.json').write_text(json.dumps(failure, indent=2) + '\n')
original_hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}
attributes = b'** -text whitespace=-blank-at-eol,-space-before-tab,cr-at-eol,-blank-at-eof\n'
files[prefix + '.gitattributes'] = attributes
for name in ('finish_native_publication_authorized.py', 'PUBLICATION_FAILURE_R1.json', 'PUBLICATION_FAILED_DIFF_CHECK.txt'):
    files[prefix + name] = (root / name).read_bytes()
new = {name: raw for name, raw in files.items() if name not in original_hashes or name.endswith('/.gitattributes')}
code = r'''import base64,hashlib,json,sys
from pathlib import Path
b=json.load(sys.stdin);project=Path('/home/gb/new butd/butd_detr-main/MCLN-main');doc=project/b['doc']
assert hashlib.sha256(doc.read_bytes()).hexdigest()==b['doc_sha']
leaf=(project/b['prefix']).resolve();assert leaf==Path('/root/autodl-tmp/mcln_published_evidence_20261005/pvground_native_joint_training_20261009/source_integration_r2') and leaf.is_dir()
for name,digest in b['original_hashes'].items():
 path=project/name;assert leaf in path.resolve().parents;assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
for name,data in b['new'].items():
 path=project/name;assert leaf in path.resolve().parents
 if not name.endswith('/.gitattributes'):assert not path.exists()
 raw=base64.b64decode(data);path.write_bytes(raw);assert path.read_bytes()==raw
print(json.dumps(dict(doc_sha256=b['doc_sha'],verified_original_files=len(b['original_hashes']),new_or_attribute_files=len(b['new']),neural_calls=0,training_status_reads=0)))
'''
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
stdin, stdout, stderr = client.exec_command(shlex.join([
    '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python', '-B', '-c', code]), timeout=120)
stdin.write(json.dumps(dict(doc=doc, prefix=prefix, doc_sha=digest,
    original_hashes=original_hashes,
    new={name: base64.b64encode(raw).decode() for name, raw in new.items()})))
stdin.channel.shutdown_write()
raw, error = stdout.read(), stderr.read()
status = stdout.channel.recv_exit_status()
(root / 'PUBLICATION_FINISH_REMOTE_STDOUT.json').write_bytes(raw)
(root / 'PUBLICATION_FINISH_REMOTE_STDERR.txt').write_bytes(error)
(root / 'PUBLICATION_FINISH_REMOTE_EXIT.json').write_text(json.dumps(dict(exit_code=status)) + '\n')
client.close()
assert status == 0, error.decode()
heads = []
for index, repo in enumerate(repos):
    selected = [doc]
    if index < 2:
        for name, data in files.items():
            path = repo / name
            if index == 0 and name in original_hashes:
                if not name.endswith('/.gitattributes'):
                    assert path.read_bytes() == data
            else:
                assert not path.exists()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        selected += sorted(files)
    (repo / doc).write_bytes(document)
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--'] + selected)
    subprocess.check_call(['git', '-C', str(repo), '-c',
        'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--cached', '--check'])
    assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + doc]) == document
    subprocess.check_call(['git', '-C', str(repo), 'commit', '--quiet', '-m',
        'Integrate PV modules into native joint training source'])
    heads.append(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip())
desktop.write_bytes(document)
subprocess.check_call(['git', '-C', str(repos[0]), '-c', 'http.version=HTTP/1.1', 'push', 'origin', 'HEAD:main'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == heads[0]
guard = root.parent / 'sync_cs_handoff_remote_20260923.py'
data = guard.read_bytes()
assert prior['doc_sha256'].encode() in data
guard.write_bytes(data.replace(prior['doc_sha256'].encode(), digest.encode()))
record = dict(status='NATIVE_NORMAL_TRAINING_SOURCE_REVIEWED_PUBLISHED_GPU_M0_PENDING',
    section='20.376.124', time_cst=datetime.datetime.now().astimezone().isoformat(),
    heads=heads, doc_sha256=digest, prefix=prefix, new_files=len(files),
    source_review_verdict=review['verdict'], native_joint_training_started=False,
    actual_gpu_preflight=False, new_accuracy_result=False, current_best_hits=[5606,4881],
    three_effective_contributions=False, full_goal_complete=False,
    publication_whitespace_interruption_preserved=True,
    exact_computational_source_bytes_preserved=True,
    training_status_reads=0, neural_calls=0, deletions=0)
(root / 'publication.json').write_text(json.dumps(record, indent=2) + '\n')
state_path = root.parent / 'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state = json.loads(state_path.read_bytes())
state.update(published_heads=heads, handoff_sha256=digest, latest_handoff_section=record['section'],
    latest_publication=str(root / 'publication.json'), full_goal_status='ACTIVE_UNMET')
state['normal_training_requirement'].update(source_review_status='R2_SOURCE_REVIEW_COMPLETE_GPU_M0_PENDING',
    source_review=str(root / 'source_review/SOURCE_REVIEW_R2.json'), publication=record)
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
goal_path = root.parent / 'pvground_referit_mask_reference_20261006/current_research_goals.json'
goal = json.loads(goal_path.read_bytes())
goal['normal_training_requirement'] = state['normal_training_requirement']
goal_path.write_text(json.dumps(goal, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record), flush=True)
