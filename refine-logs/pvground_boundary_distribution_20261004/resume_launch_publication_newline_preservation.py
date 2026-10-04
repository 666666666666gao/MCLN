"""Resume the observed staged publication after29 raw CRLF identities changed."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import paramiko

local = Path(__file__).resolve().parent
assert not (local / 'launch_publication.json').exists()
workspace = Path('C:/Users/gb')
previous = json.loads((workspace / '.codex/tmp/pvground_range_head_only_20261004/terminal_publication.json').read_bytes())
repos = [workspace / name for name in ('.codex_mcln_g0_20260905',
    '.codex_pvground_cs_20261002', '.codex_mcln_v99_internal_20260928')]
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies = [repo / doc for repo in repos] + [workspace / 'Desktop/document' / Path(doc).name]
new = copies[0].read_bytes()
assert all(path.read_bytes() == new for path in copies)
assert new.count(b'## 20.376.43 ') == 1
assert hashlib.sha256(new[:previous['handoff_bytes']]).hexdigest() == previous['handoff_sha256']
for repo, head in zip(repos, previous['heads']):
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip() == head
prefix = 'refine-logs/pvground_boundary_distribution_20261004/'
trace = local / '.aris/traces/experiment-bridge/2026-10-04_boundary_source_run01'
staged = subprocess.check_output(['git', '-C', str(repos[0]), 'diff', '--cached', '--name-only']).decode().splitlines()
assert doc in staged and 'MANIFEST.md' in staged
payloads = {}
mismatches = []
for relative in staged:
    if relative in (doc, 'MANIFEST.md'):
        continue
    assert relative.startswith(prefix)
    owned = Path(relative[len(prefix):])
    source = trace / owned.name if owned.parts[0] == 'code_review_trace' else local / owned
    raw = source.read_bytes()
    assert all((repo / relative).read_bytes() == raw for repo in repos[:2])
    indexed = subprocess.check_output(['git', '-C', str(repos[0]), 'show', ':' + relative])
    if indexed != raw:
        assert indexed == raw.replace(b'\r\n', b'\n')
        mismatches.append(dict(path=relative, raw_sha256=hashlib.sha256(raw).hexdigest(),
            indexed_sha256=hashlib.sha256(indexed).hexdigest(), raw_bytes=len(raw), indexed_bytes=len(indexed)))
    payloads[relative] = raw
assert len(mismatches) == 29
manifest = (repos[0] / 'MANIFEST.md').read_bytes()
old_manifest = subprocess.check_output(['git', '-C', str(repos[0]), 'show', 'HEAD:MANIFEST.md'])
assert manifest.replace(b'\r\n', b'\n').startswith(old_manifest)
addition = manifest.replace(b'\r\n', b'\n')[len(old_manifest):]
assert addition.count(b'frozen original-G boundary residual/distribution:') == 1
pv_manifest = (repos[1] / 'MANIFEST.md').read_bytes()
assert pv_manifest.replace(b'\r\n', b'\n') == subprocess.check_output(['git', '-C', str(repos[1]), 'show', 'HEAD:MANIFEST.md'])
(repos[1] / 'MANIFEST.md').write_bytes(pv_manifest + addition)
rule = (prefix + '** -text whitespace=cr-at-eol\n').encode()
attributes = {}
for repo in repos[:2]:
    path = repo / '.gitattributes'
    raw = path.read_bytes()
    assert rule.strip() not in raw
    assert raw.replace(b'\r\n', b'\n') == subprocess.check_output(['git', '-C', str(repo), 'show', 'HEAD:.gitattributes'])
    raw += b'\n# Preserve actual boundary experiment evidence bytes.\n' + rule
    path.write_bytes(raw)
    attributes[str(repo)] = hashlib.sha256(raw).hexdigest()
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),
    original_publisher_session=78487, original_publisher_exit=1,
    failure='indexed raw-artifact SHA assertion; observed29 CRLF-only Git normalizations; no commit or push had occurred',
    mismatches=mismatches, correction='one -text rule for this owned evidence subtree in each participating Git worktree',
    original_raw_evidence_changed=False, training_source_changed=False, model_or_job_restarted=False,
    attributes_sha256=attributes)
record_path = local / 'publication_newline_correction.json'
assert not record_path.exists()
record_path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
for path in (Path(__file__), record_path):
    payloads[prefix + path.name] = path.read_bytes()
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
    password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
remote = '/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(remote + '/' + doc, 'rb') as stream:
    assert stream.read() == new
for relative in (prefix + Path(__file__).name, prefix + record_path.name):
    raw = payloads[relative]
    for repo in repos[:2]:
        (repo / relative).write_bytes(raw)
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
    publication_correction=record_path.name, goal_achieved=False)
(local / 'launch_publication.json').write_text(json.dumps(publication, indent=2) + '\n', encoding='utf-8')
state_path = local / 'active_continuation_state.json'
state = json.loads(state_path.read_bytes())
state.update(latest_actual_publication=publication, sole_local_observer_session=46724,
    formal_launcher_execution=dict(session_id=51531, status='CLOSED_EXIT0'),
    launch_publisher_execution=dict(session_id=78487, status='CLOSED_EXIT1_RAW_CRLF_INDEX_IDENTITY'),
    goal_status='ACTIVE_UNMET', formal_metrics_available=False)
state_path.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
with (workspace / 'memory/2026-10-04.md').open('a', encoding='utf-8') as stream:
    stream.write('\nPV-Ground ' + publication['time_cst'] + ' actual §20.376.43 pushed ' + heads[0] +
        '; raw evidence preserved after29 observed Git CRLF conversions; bounded frozen-G boundary pair running, observer46724; goalACTIVE_UNMET.\n')
print(json.dumps(publication), flush=True)
