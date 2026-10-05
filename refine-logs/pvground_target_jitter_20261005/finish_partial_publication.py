"""Finish the observed EOF-format publication failure without altering evidence."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

import paramiko

root = Path(__file__).resolve().parent
study = root.parent / 'pvground_query_supported_geometry_20261005'
state_path = study / 'active_continuation_state.json'
state = json.loads(state_path.read_bytes())
previous = json.loads(Path(state['latest_publication']).read_bytes())
assert previous['section'] == '20.376.67' and not (root / 'terminal_publication.json').exists()
repos = [Path(r'C:\Users\gb\.codex_mcln_g0_20260905'), Path(r'C:\Users\gb\.codex_pvground_cs_20261002'),
         Path(r'C:\Users\gb\.codex_mcln_v99_internal_20260928')]
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies = [repo / doc for repo in repos] + [Path(r'C:\Users\gb\Desktop\document') / Path(doc).name]
new = copies[0].read_bytes()
assert all(path.read_bytes() == new for path in copies) and new.count(b'## 20.376.68 ') == 1
audit = json.loads((root / 'analysis/EXPERIMENT_AUDIT.json').read_bytes())
assert audit['verdict'] == 'PASS' and not audit['blocking_findings']
assert all(hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256'] for item in audit['reviewed_files'])


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args]).decode().strip()


for repo, head in zip(repos, previous['heads']):
    assert git(repo, 'rev-parse', 'HEAD') == head
    old = subprocess.check_output(['git', '-C', str(repo), 'show', head + ':' + doc])
    raw_previous = new[:previous['handoff_bytes']]
    assert hashlib.sha256(raw_previous).hexdigest() == previous['handoff_sha256']
    assert raw_previous.replace(b'\r\n', b'\n') == old.replace(b'\r\n', b'\n')
prefix = 'refine-logs/pvground_target_jitter_20261005/'
raw_reports = [prefix + name for name in ('SOURCE_REVIEW.json', 'SOURCE_REVIEW.md',
    'analysis/EXPERIMENT_AUDIT.json', 'analysis/EXPERIMENT_AUDIT.md',
    'complete/SOURCE_REVIEW.json', 'complete/SOURCE_REVIEW.md')]
stamp = datetime.datetime.now().astimezone().isoformat()
recovery = dict(time_cst=stamp, failure='git diff --check: new blank line at EOF in six raw reviewer artifacts',
    raw_reports=raw_reports, raw_report_bytes_changed=False, new_experiment_run=False,
    source_and_doc_whitespace_check='strict excluding six raw reports; existing CRLF accepted',
    all_staged_check='default checks except blank-at-eof; existing CRLF accepted',
    observed_index_normalization='SOURCE_REVIEW_CALL, TERMINAL_REVIEW_CALL and SUMMARY differed from raw bytes',
    namespace_staging='namespace-only .gitattributes -text plus per-command core.autocrlf=false; no global configuration change',
    partial_doc_sha256=hashlib.sha256(new).hexdigest())
(root / 'publication_recovery.json').write_text(json.dumps(recovery, indent=2) + '\n', encoding='utf-8')
(root / '.gitattributes').write_bytes(b'** -text\n')
payloads = {prefix + file.relative_to(root).as_posix(): file.read_bytes() for file in root.rglob('*')
            if file.is_file() and file.suffix in ('.py', '.json', '.md', '.jsonl', '.log', '.exit')}
payloads[prefix + '.gitattributes'] = (root / '.gitattributes').read_bytes()
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
    password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
project = '/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(project + '/' + doc, 'rb') as stream:
    assert stream.read() == new
for name, raw in payloads.items():
    for repo in repos[:2]:
        destination = repo / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    with sftp.open(project + '/' + name, 'wb') as stream:
        stream.write(raw)
    with sftp.open(project + '/' + name, 'rb') as stream:
        assert stream.read() == raw
sftp.close()
client.close()
heads = []
for index, repo in enumerate(repos):
    stage = [doc] + (list(payloads) if index < 2 else [])
    if index < 2:
        if index == 1:
            with (repo / 'MANIFEST.md').open('a', encoding='utf-8') as stream:
                stream.write('\n- ' + stamp + ' CPU64 native target-jitter check closed and reviewed; exact native inputs, no model/checkpoint/optimizer. Best5614/4509 unchanged.\n')
        stage.append('MANIFEST.md')
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', doc] + (['MANIFEST.md'] if index < 2 else []), stderr=subprocess.DEVNULL)
    if index < 2:
        subprocess.check_call(['git', '-C', str(repo), '-c', 'core.autocrlf=false', 'add', '-f', '--', *payloads], stderr=subprocess.DEVNULL)
        subprocess.check_call(['git', '-C', str(repo), '-c', 'core.autocrlf=false', 'add', '--renormalize', '--', *payloads], stderr=subprocess.DEVNULL)
    excludes = [':(exclude)' + name for name in raw_reports] if index < 2 else []
    subprocess.check_call(['git', '-C', str(repo), '-c', 'core.whitespace=cr-at-eol', 'diff', '--cached', '--check', '--', *stage, *excludes])
    subprocess.check_call(['git', '-C', str(repo), '-c', 'core.whitespace=cr-at-eol,-blank-at-eof',
                           'diff', '--cached', '--check', '--', *stage])
    for name in payloads if index < 2 else []:
        assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + name]) == payloads[name]
    assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + doc]) == new.replace(b'\r\n', b'\n')
    subprocess.check_call(['git', '-C', str(repo), 'commit', '--quiet', '-m', 'Record bounded native target-jitter diagnostic'])
    heads.append(git(repo, 'rev-parse', 'HEAD'))
    assert not git(repo, 'status', '--porcelain')
subprocess.check_call(['git', '-C', str(repos[0]), '-c', 'http.version=HTTP/1.1', 'push', 'origin', 'HEAD:main'])
assert git(repos[0], 'ls-remote', 'origin', 'refs/heads/main').split()[0] == heads[0]
digest = hashlib.sha256(new).hexdigest()
guard = Path(r'C:\Users\gb\.codex\tmp\sync_cs_handoff_remote_20260923.py')
raw = guard.read_bytes()
assert raw.count(previous['handoff_sha256'].encode()) == 1
guard.write_bytes(raw.replace(previous['handoff_sha256'].encode(), digest.encode()))
record = dict(previous)
record.update(time_cst=stamp, section='20.376.68', heads=heads, github_main=heads[0], handoff_bytes=len(new),
    handoff_sha256=digest, four_local_and_remote_equal=True, payload_count=len(payloads),
    status='ACTUAL_CLOSED_CPU_LABEL_CHECK_PUBLISHED', new_accuracy_result=False,
    diagnostic_CPU_executed=True, diagnostic_GPU_executed=False,
    diagnostic_finished_cst=json.loads((root / 'complete/receipt.json').read_bytes())['finished_cst'],
    diagnostic_integrity_verdict=audit['verdict'], raw_reviewer_report_bytes_preserved=True,
    publication_recovery=str(root / 'publication_recovery.json'))
(root / 'terminal_publication.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
state.update(latest_publication=str(root / 'terminal_publication.json'), published_heads=heads,
    handoff_section='20.376.68', handoff_sha256=digest, status='CLOSED_CPU_LABEL_CHECK_PUBLISHED',
    label_diagnostic_closed=True, next_action='Deploy reviewed auxiliary-target sanity from protected4509',
    current_goal_turn_classification='PROGRESS_CPU_LABEL_CHECK_AND_AUXILIARY_SOURCE')
state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
with Path(r'C:\Users\gb\memory\2026-10-05.md').open('a', encoding='utf-8') as stream:
    stream.write('\nPVGround ' + stamp + ': Doc68 CPU64 exact data check published after observed raw-report EOF formatting failure; raw audit bytes preserved. Qualification1090→861, removed395/new166; fixed outside201→87. Median max face jitter0.053923m. No formal accuracy change; best5614/4509. Main ' + heads[0] + '. Next reviewed auxiliary-target two-step sanity; ACTIVE_UNMET.\n')
print(json.dumps({key: record[key] for key in ('time_cst', 'section', 'heads', 'handoff_sha256',
    'four_local_and_remote_equal', 'diagnostic_integrity_verdict', 'new_accuracy_result')}), flush=True)
