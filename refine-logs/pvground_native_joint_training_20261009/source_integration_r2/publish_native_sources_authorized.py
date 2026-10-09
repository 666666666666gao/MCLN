"""Publish reviewed native-training source only; no job admission or polling."""
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
assert not (root / 'publication.json').exists()
old = (repos[0] / doc).read_bytes()
assert hashlib.sha256(old).hexdigest() == prior['doc_sha256']
assert b'20.376.124' not in old
assert all((repo / doc).read_bytes() == old for repo in repos) and desktop.read_bytes() == old
for index, repo in enumerate(repos):
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip() == prior['heads'][index]
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain']).strip()
review = json.loads((root / 'source_review/SOURCE_REVIEW_R2.json').read_bytes())
assert review['execution_scope'] == 'SOURCE_ONLY'
assert review['verdict'] in ('PASS', 'WARN') and not review['blocking_findings']
receipt = json.loads((root / 'RESUME_CORRECTION_R2.json').read_bytes())
for relative, digest in receipt['source_files'].items():
    assert hashlib.sha256((root / 'source' / relative).read_bytes()).hexdigest() == digest
notes = (root / 'HANDOFF_NOTES.md').read_text(encoding='utf-8')
document = old + ('\n\n' + notes).replace('\n', '\r\n').encode('utf-8')
digest = hashlib.sha256(document).hexdigest()
prefix = 'refine-logs/pvground_native_joint_training_20261009/source_integration_r2/'
paths = sorted((root / 'source').rglob('*.py')) + sorted((root / 'init_manifests').glob('*.json'))
paths += [root / name for name in (
    'HANDOFF_NOTES.md', 'NORMAL_TRAINING_PLAN.md', 'NORMAL_TRAINING_DIRECTION.json',
    'SOURCE_PREPARATION.json', 'ENTRY_PREPARATION.json', 'METRIC_PREPARATION.json',
    'RESUME_CORRECTION_R2.json', 'RESUME_CORRECTION_R2.diff', 'INIT_MANIFEST_PREPARATION.json',
    'prepare_native_sources.py', 'integrate_native_entry.py', 'finish_native_metrics.py',
    'seal_resume_correction_r2.py', 'prepare_native_init_manifests.py',
    'record_native_training_direction.py', 'publish_native_sources_authorized.py',
    'verify_native_publication_authorized.py', 'SOURCE_REVIEW_REQUEST.txt')]
paths += [root / 'source_review' / name for name in (
    'SOURCE_REVIEW_R1.md', 'SOURCE_REVIEW_R1.json', 'SOURCE_REVIEW_R1.seal.json',
    'RAW_RESPONSE_R1.md', 'INPUT_MANIFEST_R1.json', 'R1_STATIC_VERIFICATION.json',
    'SOURCE_REVIEW_R2.md', 'SOURCE_REVIEW_R2.json', 'SOURCE_REVIEW_R2.seal.json',
    'RAW_RESPONSE_R2.md')]
files = {prefix + path.relative_to(root).as_posix(): path.read_bytes() for path in paths}
files[prefix + '.gitattributes'] = b'** -text whitespace=blank-at-eol,space-before-tab,cr-at-eol,-blank-at-eof\n'
assert len(files) == len(paths) + 1
assert all(not (repos[0] / name).exists() for name in files)
remote_code = r'''import base64,hashlib,json,sys
from pathlib import Path
b=json.load(sys.stdin);project=Path('/home/gb/new butd/butd_detr-main/MCLN-main');doc=project/b['doc']
assert hashlib.sha256(doc.read_bytes()).hexdigest()==b['old_sha']
leaf=(project/b['prefix']).resolve()
assert leaf==Path('/root/autodl-tmp/mcln_published_evidence_20261005/pvground_native_joint_training_20261009/source_integration_r2') and not leaf.exists()
for name,data in b['files'].items():
 path=project/name;assert leaf in path.resolve().parents
 path.parent.mkdir(parents=True,exist_ok=True);raw=base64.b64decode(data);path.write_bytes(raw);assert path.read_bytes()==raw
raw=base64.b64decode(b['document']);temp=doc.with_name(doc.name+'.tmp_native_source_20261009')
assert not temp.exists();temp.write_bytes(raw);assert hashlib.sha256(temp.read_bytes()).hexdigest()==b['sha'];temp.replace(doc)
print(json.dumps(dict(doc_sha256=b['sha'],files=len(b['files']),neural_calls=0,training_status_reads=0,deletions=0)))
'''
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
stdin, stdout, stderr = client.exec_command(shlex.join([
    '/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python', '-B', '-c', remote_code]), timeout=120)
stdin.write(json.dumps(dict(doc=doc, old_sha=prior['doc_sha256'], prefix=prefix,
    sha=digest, document=base64.b64encode(document).decode(),
    files={name: base64.b64encode(data).decode() for name, data in files.items()})))
stdin.channel.shutdown_write()
raw, error = stdout.read(), stderr.read()
status = stdout.channel.recv_exit_status()
(root / 'PUBLICATION_REMOTE_STDOUT.json').write_bytes(raw)
(root / 'PUBLICATION_REMOTE_STDERR.txt').write_bytes(error)
(root / 'PUBLICATION_REMOTE_EXIT.json').write_text(json.dumps(dict(exit_code=status)) + '\n')
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
