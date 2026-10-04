"""Finish the observed staged packet after explicit Git index re-normalization."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import paramiko

local = Path(__file__).resolve().parent
workspace = Path('C:/Users/gb')
assert not (local / 'launch_publication.json').exists()
previous = json.loads((workspace / '.codex/tmp/pvground_range_head_only_20261004/terminal_publication.json').read_bytes())
repos = [workspace / name for name in ('.codex_mcln_g0_20260905',
    '.codex_pvground_cs_20261002', '.codex_mcln_v99_internal_20260928')]
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies = [repo / doc for repo in repos] + [workspace / 'Desktop/document' / Path(doc).name]
new = copies[0].read_bytes()
assert all(path.read_bytes() == new for path in copies) and new.count(b'## 20.376.43 ') == 1
assert hashlib.sha256(new[:previous['handoff_bytes']]).hexdigest() == previous['handoff_sha256']
for repo, head in zip(repos, previous['heads']):
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip() == head
prefix = 'refine-logs/pvground_boundary_distribution_20261004/'
trace = local / '.aris/traces/experiment-bridge/2026-10-04_boundary_source_run01'
staged = subprocess.check_output(['git', '-C', str(repos[0]), 'diff', '--cached', '--name-only']).decode().splitlines()
assert {doc, 'MANIFEST.md', '.gitattributes'}.issubset(staged)
payloads = {}
for relative in staged:
    if relative in (doc, 'MANIFEST.md', '.gitattributes'):
        continue
    assert relative.startswith(prefix)
    owned = Path(relative[len(prefix):])
    source = trace / owned.name if owned.parts[0] == 'code_review_trace' else local / owned
    raw = source.read_bytes()
    assert all((repo / relative).read_bytes() == raw for repo in repos[:2])
    payloads[relative] = raw
raw = Path(__file__).read_bytes()
relative = prefix + Path(__file__).name
assert all(not (repo / relative).exists() for repo in repos[:2])
for repo in repos[:2]:
    (repo / relative).write_bytes(raw)
payloads[relative] = raw
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
    password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
remote = '/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(remote + '/' + doc, 'rb') as stream:
    assert stream.read() == new
with sftp.open(remote + '/' + relative, 'wx') as stream:
    stream.write(raw)
with sftp.open(remote + '/' + relative, 'rb') as stream:
    assert stream.read() == raw
sftp.close()
client.close()
heads = []
for index, repo in enumerate(repos):
    stage = [doc] + (['MANIFEST.md', '.gitattributes', *payloads] if index < 2 else [])
    changed = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage])
    if index < 2:
        subprocess.check_call(['git', '-C', str(repo), 'add', '--renormalize', '--', prefix])
    check = [doc] + (['MANIFEST.md', '.gitattributes'] + [path for path in payloads
        if '/' not in path[len(prefix):]] if index < 2 else [])
    subprocess.check_call(['git', '-C', str(repo), '-c',
        'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol',
        'diff', '--cached', '--check', '--', *check])
    for relative, raw in (payloads.items() if index < 2 else []):
        assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + relative]) == raw
    subprocess.check_call(['git', '-C', str(repo), 'commit', '-m',
        'Record actual boundary checks and bounded original G comparison launch'])
    heads.append(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip())
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
subprocess.check_call(['git', '-C', str(repos[0]), '-c', 'http.version=HTTP/1.1',
    'push', 'origin', 'HEAD:main'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == heads[0]
assert all(path.read_bytes() == new for path in copies)
for relative in (doc, *payloads):
    assert subprocess.check_output(['git', '-C', str(repos[0]), 'rev-parse', 'HEAD:' + relative]) == subprocess.check_output(['git', '-C', str(repos[1]), 'rev-parse', 'HEAD:' + relative])
digest = hashlib.sha256(new).hexdigest()
guard = workspace / '.codex/tmp/sync_cs_handoff_remote_20260923.py'
guard_raw = guard.read_bytes()
assert guard_raw.count(previous['handoff_sha256'].encode()) == 1
guard.write_bytes(guard_raw.replace(previous['handoff_sha256'].encode(), digest.encode()))
publication = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), section='20.376.43',
    heads=heads, github_main=heads[0], handoff_sha256=digest, handoff_bytes=len(new),
    four_local_and_remote_equal=True, exact_committed_payloads=True, payload_count=len(payloads),
    source_review='PASS/0blocking/same-family/provisional', actual_engineering_pass=True,
    native_model_updates=4, fit_started=True, formal_metrics_available=False, retained_best='original_g',
    publication_correction='observed29 CRLF normalizations; added owned -text rule and explicitly renormalized staged index',
    goal_achieved=False)
(local / 'launch_publication.json').write_text(json.dumps(publication, indent=2) + '\n', encoding='utf-8')
state_path = local / 'active_continuation_state.json'
state = json.loads(state_path.read_bytes())
state.update(latest_actual_publication=publication, sole_local_observer_session=46724,
    formal_launcher_execution=dict(session_id=51531, status='CLOSED_EXIT0'),
    launch_publisher_execution=dict(session_id=78487, status='CLOSED_EXIT1_RAW_CRLF_INDEX_IDENTITY'),
    newline_resume_execution=dict(session_id=9493, status='CLOSED_EXIT1_INDEX_NEEDED_EXPLICIT_RENORMALIZE'),
    goal_status='ACTIVE_UNMET', formal_metrics_available=False)
state_path.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
with (workspace / 'memory/2026-10-04.md').open('a', encoding='utf-8') as stream:
    stream.write('\nPV-Ground ' + publication['time_cst'] + ' actual §20.376.43 pushed ' + heads[0] +
        '; raw evidence preserved with owned -text and explicit renormalize after29 observed Git CRLF conversions; boundary pair running; goalACTIVE_UNMET.\n')
print(json.dumps(publication), flush=True)
