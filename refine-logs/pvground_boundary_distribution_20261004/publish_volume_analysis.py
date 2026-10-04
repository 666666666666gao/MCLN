"""Publish only the prepared GT-volume result analysis update."""
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import paramiko

local = Path(__file__).resolve().parent
assert not (local / 'volume_analysis_publication.json').exists()
workspace = Path('C:/Users/gb')
previous = json.loads((local / 'terminal_tools_publication.json').read_bytes())
update = json.loads((local / 'terminal_tool_volume_group_update.json').read_bytes())
repos = [workspace / name for name in ('.codex_mcln_g0_20260905',
    '.codex_pvground_cs_20261002', '.codex_mcln_v99_internal_20260928')]
for repo, head in zip(repos, previous['heads']):
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip() == head
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
prefix = 'refine-logs/pvground_boundary_distribution_20261004/'
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
assert all(hashlib.sha256((repo / doc).read_bytes()).hexdigest() == previous['handoff_sha256'] for repo in repos)
assert hashlib.sha256((local / 'analyze_terminal.py').read_bytes()).hexdigest() == update['current_analyzer_sha256']
changed = ('analyze_terminal.py', 'prepare_terminal_audit_request.py')
for repo in repos[:2]:
    assert hashlib.sha256((repo / prefix / 'analyze_terminal.py').read_bytes()).hexdigest() == update['previous_analyzer_sha256']
names = [*changed, 'terminal_tool_volume_group_update.json', Path(__file__).name]
payloads = {prefix + name: (local / name).read_bytes() for name in names}
for name in names:
    if name.endswith('.py'):
        ast.parse((local / name).read_text(encoding='utf-8'))
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
    password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
remote = '/home/gb/new butd/butd_detr-main/MCLN-main'
for name in changed:
    with sftp.open(remote + '/' + prefix + name, 'rb') as stream:
        assert stream.read() == (repos[0] / prefix / name).read_bytes()
for relative, raw in payloads.items():
    for repo in repos[:2]:
        (repo / relative).write_bytes(raw)
    mode = 'wb' if relative[len(prefix):] in changed else 'wx'
    with sftp.open(remote + '/' + relative, mode) as stream:
        stream.write(raw)
    with sftp.open(remote + '/' + relative, 'rb') as stream:
        assert stream.read() == raw
sftp.close()
client.close()
stamp = datetime.datetime.now().astimezone().isoformat()
heads = []
for repo in repos[:2]:
    with (repo / 'MANIFEST.md').open('a', encoding='utf-8') as stream:
        stream.write('\n- ' + stamp + ' boundary result tools: offline GT-volume quartiles prepared,2377 rows/group, same-query repairs/damages; no result execution or inference/training change.\n')
    stage = ['MANIFEST.md', *payloads]
    status = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all']).decode()
    assert {line[3:] for line in status.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage], stdout=subprocess.DEVNULL)
    subprocess.check_call(['git', '-C', str(repo), '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol',
        'diff', '--cached', '--check', '--', *stage])
    for relative, raw in payloads.items():
        assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + relative]) == raw
    subprocess.check_call(['git', '-C', str(repo), 'commit', '-q', '-m',
        'Add offline target-volume groups to boundary result analysis'])
    heads.append(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip())
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
heads.append(previous['heads'][2])
subprocess.check_call(['git', '-C', str(repos[0]), '-c', 'http.version=HTTP/1.1', 'push', 'origin', 'HEAD:main'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == heads[0]
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), heads=heads,
    github_main=heads[0], handoff_sha256=previous['handoff_sha256'], handoff_bytes=previous['handoff_bytes'],
    handoff_unchanged=True, payload_count=len(payloads), raw_payloads_exact=True, remote_payloads_exact=True,
    analysis_executed=False, active_training_changed=False, inference_uses_gt_volume=False,
    current_analyzer_sha256=update['current_analyzer_sha256'],
    current_audit_request_preparer_sha256=hashlib.sha256((local/'prepare_terminal_audit_request.py').read_bytes()).hexdigest(),
    goal_achieved=False)
(local / 'volume_analysis_publication.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
p = local / 'active_continuation_state.json'
state = json.loads(p.read_bytes())
state.update(latest_actual_volume_analysis_publication=record, latest_publication_heads=heads,
    goal_status='ACTIVE_UNMET', formal_metrics_available=False)
p.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record), flush=True)
