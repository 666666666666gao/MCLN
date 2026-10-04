"""Publish prepared terminal tools without changing active training or the handoff."""
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import paramiko

local = Path(__file__).resolve().parent
workspace = Path('C:/Users/gb')
assert not (local / 'terminal_tools_publication.json').exists()
previous = json.loads((local / 'launch_publication.json').read_bytes())
repos = [workspace / name for name in ('.codex_mcln_g0_20260905',
    '.codex_pvground_cs_20261002', '.codex_mcln_v99_internal_20260928')]
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies = [repo / doc for repo in repos] + [workspace / 'Desktop/document' / Path(doc).name]
assert all(hashlib.sha256(path.read_bytes()).hexdigest() == previous['handoff_sha256'] for path in copies)
for repo, head in zip(repos, previous['heads']):
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip() == head
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == previous['heads'][0]
names = ['prepare_terminal_tools.py', 'collect_terminal.py', 'analyze_terminal.py',
    'terminal_tool_preparation.json', 'terminal_tool_text_correction.json',
    'prepare_terminal_audit_request.py', 'terminal_audit_preparer_identity.json', Path(__file__).name]
prefix = 'refine-logs/pvground_boundary_distribution_20261004/'
payloads = {prefix + name: (local / name).read_bytes() for name in names}
for name in names:
    if name.endswith('.py'):
        ast.parse((local / name).read_text(encoding='utf-8'))
correction = json.loads((local / 'terminal_tool_text_correction.json').read_bytes())
assert correction['new_analyzer_sha256'] == hashlib.sha256((local / 'analyze_terminal.py').read_bytes()).hexdigest()
preparation = json.loads((local / 'terminal_tool_preparation.json').read_bytes())
assert preparation['files']['collect_terminal.py'] == hashlib.sha256((local / 'collect_terminal.py').read_bytes()).hexdigest()
assert all(not (repo / relative).exists() for repo in repos[:2] for relative in payloads)
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
    password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
remote = '/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(remote + '/' + doc, 'rb') as stream:
    assert hashlib.sha256(stream.read()).hexdigest() == previous['handoff_sha256']
for relative, raw in payloads.items():
    for repo in repos[:2]:
        (repo / relative).write_bytes(raw)
    with sftp.open(remote + '/' + relative, 'wx') as stream:
        stream.write(raw)
    with sftp.open(remote + '/' + relative, 'rb') as stream:
        assert stream.read() == raw
sftp.close()
client.close()
stamp = datetime.datetime.now().astimezone().isoformat()
heads = []
for repo in repos[:2]:
    with (repo / 'MANIFEST.md').open('a', encoding='utf-8') as stream:
        stream.write('\n- ' + stamp + ' boundary terminal collection/row analysis/fresh audit request tools prepared and published; no terminal execution or metrics, active training unchanged.\n')
    stage = ['MANIFEST.md', *payloads]
    changed = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage], stdout=subprocess.DEVNULL)
    subprocess.check_call(['git', '-C', str(repo), '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol',
        'diff', '--cached', '--check', '--', *stage])
    for relative, raw in payloads.items():
        assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + relative]) == raw
    subprocess.check_call(['git', '-C', str(repo), 'commit', '-q', '-m',
        'Prepare boundary terminal row analysis and fresh audit request tools'])
    heads.append(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip())
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
heads.append(previous['heads'][2])
subprocess.check_call(['git', '-C', str(repos[0]), '-c', 'http.version=HTTP/1.1', 'push', 'origin', 'HEAD:main'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == heads[0]
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), heads=heads,
    github_main=heads[0], handoff_sha256=previous['handoff_sha256'], handoff_bytes=previous['handoff_bytes'],
    handoff_unchanged=True, raw_payloads_exact=True, remote_payloads_exact=True, payload_count=len(payloads),
    execution_scope='PREPARED_LOCAL_AST_ONLY', terminal_collection_executed=False,
    terminal_analysis_executed=False, terminal_audit_launched=False,
    active_model_source_changed=False, training_restarted=False, goal_achieved=False)
(local / 'terminal_tools_publication.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
state_path = local / 'active_continuation_state.json'
state = json.loads(state_path.read_bytes())
state.update(latest_actual_terminal_tools_publication=record, latest_publication_heads=heads,
    goal_status='ACTIVE_UNMET', formal_metrics_available=False)
state_path.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record), flush=True)
