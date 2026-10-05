"""Publish reviewed closed-weight retention sources; no cleanup or doc rewrite."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import paramiko

local = Path(__file__).resolve().parent
workspace = Path('C:/Users/gb')
previous = json.loads((local / 'code_publication.json').read_bytes())
review = json.loads((local / 'RETENTION_REVIEW.json').read_bytes())
call = json.loads((local / 'RETENTION_REVIEW_CALL.json').read_bytes())
assert review['verdict'] == 'PASS' and not review['blocking_findings'] and call['result_received']
for entry in review['reviewed_files']:
    assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() == entry['sha256']
assert not (local / 'retention_code_publication.json').exists()
repos = [workspace / '.codex_mcln_g0_20260905', workspace / '.codex_pvground_cs_20261002']
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
for index, repo in enumerate(repos):
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip() == previous['heads'][index]
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
    assert hashlib.sha256((repo / doc).read_bytes()).hexdigest() == previous['handoff_sha256']
names = ['retain_metric_best.py', 'retain_metric_best_authorized.py',
         'RETENTION_REVIEW.md', 'RETENTION_REVIEW.json', 'RETENTION_REVIEW_CALL.json',
         'RETENTION_REVIEW_20261005_100750.md', 'RETENTION_REVIEW_20261005_100750.json',
         'record_retention_review_result.py', Path(__file__).name]
prefix = 'refine-logs/pvground_query_supported_geometry_20261005/'
payloads = {prefix + name: (local / name).read_bytes() for name in names}
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
project = '/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(project + '/' + doc, 'rb') as stream:
    assert hashlib.sha256(stream.read()).hexdigest() == previous['handoff_sha256']
for name, raw in payloads.items():
    with sftp.open(project + '/' + name, 'wx') as stream:
        stream.write(raw)
    with sftp.open(project + '/' + name, 'rb') as stream:
        assert stream.read() == raw
sftp.close()
client.close()
stamp = datetime.datetime.now().astimezone().isoformat()
heads = []
for repo in repos:
    for name, raw in payloads.items():
        (repo / name).write_bytes(raw)
    with (repo / 'MANIFEST.md').open('a', encoding='utf-8') as stream:
        stream.write('\n- ' + stamp + ' Query-supported geometry closed-weight retention SOURCE_ONLY review PASS, same-family/provisional. Actual closure and fresh terminal audit remain required; no cleanup executed or new fit accuracy.\n')
    stage = ['MANIFEST.md', *payloads]
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage], stderr=subprocess.DEVNULL)
    subprocess.check_call(['git', '-C', str(repo), 'diff', '--cached', '--check', '--', *stage])
    for name, raw in payloads.items():
        assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + name]) == raw.replace(b'\r\n', b'\n')
    subprocess.check_call(['git', '-C', str(repo), 'commit', '--quiet', '-m', 'Prepare reviewed metric-best checkpoint retention after terminal audit'])
    heads.append(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip())
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
subprocess.check_call(['git', '-C', str(repos[0]), '-c', 'http.version=HTTP/1.1', 'push', 'origin', 'HEAD:main'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == heads[0]
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), heads=heads + [previous['heads'][2]],
              github_main=heads[0], handoff_bytes=previous['handoff_bytes'], handoff_sha256=previous['handoff_sha256'],
              previous_publication=str(local / 'code_publication.json'), section='20.376.64',
              handoff_rewritten=False, new_fit_result=False, cleanup_executed=False,
              source_review_scope='SOURCE_ONLY', source_review_acceptance='same-family/provisional',
              payload_count=len(payloads), current_best_hits=[5616, 4506], full_goal_status='ACTIVE_UNMET')
(local / 'retention_code_publication.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
state = json.loads((local / 'active_continuation_state.json').read_bytes())
state.update(latest_publication=str(local / 'retention_code_publication.json'), time_cst=record['time_cst'],
             retention_tools_published=True, retention_executed=False)
(local / 'active_continuation_state.json').write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
with (workspace / 'memory/2026-10-05.md').open('a', encoding='utf-8') as stream:
    stream.write('\nPVGround ' + stamp + ': reviewed closed-weight retention sources published main ' + heads[0] + '. SOURCE_ONLY PASS, same-family/provisional; no deletions/new fit metrics. Keep actual best plus reconstruction parents and active recovery. Controller 584730/observer 11178; first scheduled remote observation 11:05:34 CST.\n')
print(json.dumps(record), flush=True)
