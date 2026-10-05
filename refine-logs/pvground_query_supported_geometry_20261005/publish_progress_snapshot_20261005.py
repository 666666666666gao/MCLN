"""Publish actual scheduled progress only; no formal accuracy or document rewrite."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import paramiko

local = Path(__file__).resolve().parent
workspace = Path('C:/Users/gb')
state = json.loads((local / 'active_continuation_state.json').read_bytes())
previous = json.loads(Path(state['latest_publication']).read_bytes())
check = json.loads((local / 'FIRST_PROGRESS_RESOURCE_CHECK.json').read_bytes())
estimate = json.loads((local / 'PROGRESS_ESTIMATE.json').read_bytes())
assert check['training']['completed_logged_updates'] == 3723
assert check['training']['logged_fit_rows'] == 29778
assert not estimate['new_formal_accuracy_result']
assert not (local / 'progress_publication.json').exists()
repos = [workspace / '.codex_mcln_g0_20260905', workspace / '.codex_pvground_cs_20261002']
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
for index, repo in enumerate(repos):
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip() == previous['heads'][index]
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
    assert hashlib.sha256((repo / doc).read_bytes()).hexdigest() == previous['handoff_sha256']
names = ['inspect_scheduled_progress.py', 'FIRST_PROGRESS_RESOURCE_CHECK.json',
         'record_progress_estimate.py', 'PROGRESS_ESTIMATE.json', Path(__file__).name]
prefix = 'refine-logs/pvground_query_supported_geometry_20261005/'
payloads = {prefix + name: (local / name).read_bytes() for name in names}
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
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
        stream.write('\n- ' + stamp + ' Actual scheduled control progress3723/3723 updates,29778 fit rows,terminal holdout5120/6887 at11:08 snapshot. Same-throughput pair estimate14:00; no formal accuracy, document rewrite or cleanup.\n')
    stage = ['MANIFEST.md', *payloads]
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage], stderr=subprocess.DEVNULL)
    subprocess.check_call(['git', '-C', str(repo), 'diff', '--cached', '--check', '--', *stage])
    for name, raw in payloads.items():
        assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + name]) == raw.replace(b'\r\n', b'\n')
    subprocess.check_call(['git', '-C', str(repo), 'commit', '--quiet', '-m', 'Record actual scheduled geometry fit progress and calibrated timing'])
    heads.append(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip())
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
subprocess.check_call(['git', '-C', str(repos[0]), '-c', 'http.version=HTTP/1.1', 'push', 'origin', 'HEAD:main'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == heads[0]
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), heads=heads + [previous['heads'][2]],
              github_main=heads[0], handoff_bytes=previous['handoff_bytes'], handoff_sha256=previous['handoff_sha256'],
              previous_publication=state['latest_publication'], section='20.376.64', handoff_rewritten=False,
              new_fit_result=False, new_fit_progress=True, cleanup_executed=False, source_ast_check_only=True,
              payload_count=len(payloads), current_best_hits=[5616, 4506], full_goal_status='ACTIVE_UNMET')
(local / 'progress_publication.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
state.update(latest_publication=str(local / 'progress_publication.json'), time_cst=record['time_cst'],
             scheduled_progress_snapshot_published=True)
(local / 'active_continuation_state.json').write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
with (workspace / 'memory/2026-10-05.md').open('a', encoding='utf-8') as stream:
    stream.write('\nPVGround ' + stamp + ': actual scheduled progress/resources and calibrated estimate published main ' + heads[0] + '. Control3723 updates/29778 rows complete; terminal holdout5120/6887 observed11:08:50. No formal accuracy/docrewrite/cleanup. Pair estimate14:00:41 based same-throughput assumption; strategy unmeasured. Controller584730/observer11178. Current turn actual progress; GoalACTIVE_UNMET.\n')
print(json.dumps(record), flush=True)
