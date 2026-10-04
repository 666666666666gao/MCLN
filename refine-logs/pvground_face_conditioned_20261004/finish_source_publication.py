"""Finish the observed JSON CRLF index-conversion interruption, exactly once."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import paramiko


local = Path(__file__).resolve().parent
assert not (local / 'source_publication.json').exists()
previous = json.loads((local.parent / 'pvground_boundary_distribution_20261004/terminal_publication.json').read_bytes())
workspace = Path('C:/Users/gb')
repos = [workspace / '.codex_mcln_g0_20260905', workspace / '.codex_pvground_cs_20261002',
         workspace / '.codex_mcln_v99_internal_20260928']
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies = [repo / doc for repo in repos] + [workspace / 'Desktop/document' / Path(doc).name]
new = copies[0].read_bytes()
assert all(path.read_bytes() == new for path in copies)
assert hashlib.sha256(new[:previous['handoff_bytes']]).hexdigest() == previous['handoff_sha256']
assert new[previous['handoff_bytes']:].count('## 20.376.46'.encode()) == 1
prefix = 'refine-logs/pvground_face_conditioned_20261004/'
rule = b'\n# Preserve actual face decoder source and probe receipt bytes.\n' + prefix.encode() + b'** -text whitespace=cr-at-eol\n'
for repo, head in zip(repos, previous['heads']):
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip() == head
for repo in repos[:2]:
    path = repo / '.gitattributes'
    raw = path.read_bytes()
    assert prefix.encode() not in raw
    path.write_bytes(raw + rule)
first_manifest = (repos[0] / 'MANIFEST.md').read_text(encoding='utf-8')
marker = 'face-conditioned decoder unintegrated source draft; LOCAL_AST_ONLY, no new GPU result.'
assert first_manifest.count(marker) == 1
entry = next(line for line in first_manifest.splitlines() if marker in line)
second = repos[1] / 'MANIFEST.md'
assert marker not in second.read_text(encoding='utf-8')
with second.open('a', encoding='utf-8') as stream:
    stream.write('\n' + entry + '\n')
names = ('pvground_face_conditioned_box_refiner.py', 'IMPLEMENTATION_SCOPE.md',
         'LOCAL_AST_CHECK.json', 'publish_source_draft.py', 'finish_source_publication.py',
         'OBSERVED_NEWLINE_CORRECTION.json')
payloads = {prefix + name: (local / name).read_bytes() for name in names}
assert hashlib.sha256((local / names[0]).read_bytes()).hexdigest() == '4bb498f776c83a92ddc1218753cab49e8700527bc25bae5ec630d69fdf695bd3'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
remote = '/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(remote + '/' + doc, 'rb') as stream:
    assert stream.read() == new
for name in names[:4]:
    with sftp.open(remote + '/' + prefix + name, 'rb') as stream:
        assert stream.read() == payloads[prefix + name]
for relative, raw in payloads.items():
    for repo in repos[:2]:
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
for name in names[4:]:
    relative = prefix + name
    with sftp.open(remote + '/' + relative, 'wx') as stream:
        stream.write(payloads[relative])
    with sftp.open(remote + '/' + relative, 'rb') as stream:
        assert stream.read() == payloads[relative]
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
    subprocess.check_call(['git', '-C', str(repo), '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol',
                           'diff', '--cached', '--check', '--', *stage])
    for relative, raw in (payloads.items() if index < 2 else []):
        assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + relative]) == raw
    subprocess.check_call(['git', '-C', str(repo), 'commit', '-m', 'Record face-conditioned decoder draft with exact source evidence bytes'])
    heads.append(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip())
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
subprocess.check_call(['git', '-C', str(repos[0]), '-c', 'http.version=HTTP/1.1', 'push', 'origin', 'HEAD:main'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == heads[0]
for relative in (doc, *payloads):
    assert subprocess.check_output(['git', '-C', str(repos[0]), 'rev-parse', 'HEAD:' + relative]) == subprocess.check_output(['git', '-C', str(repos[1]), 'rev-parse', 'HEAD:' + relative])
guard = workspace / '.codex/tmp/sync_cs_handoff_remote_20260923.py'
guard_raw = guard.read_bytes()
assert guard_raw.count(previous['handoff_sha256'].encode()) == 1
digest = hashlib.sha256(new).hexdigest()
guard.write_bytes(guard_raw.replace(previous['handoff_sha256'].encode(), digest.encode()))
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), section='20.376.46',
              heads=heads, github_main=heads[0], handoff_sha256=digest, handoff_bytes=len(new),
              four_local_and_remote_equal=True, exact_committed_payloads=True, payload_count=len(payloads),
              observed_newline_correction='LOCAL_AST_CHECK13CRLF; scoped-text-rule and explicit renormalization',
              execution_scope='LOCAL_AST_ONLY', native_factory_integrated=False,
              gpu_forwards=0, optimizer_updates=0, accuracy_evidence=False,
              retained_metric_best='distribution_5616_4506', weights_created=0,
              weights_downloaded=0, weights_deleted=0, goal_achieved=False)
(local / 'source_publication.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
with (workspace / 'memory/2026-10-04.md').open('a', encoding='utf-8') as stream:
    stream.write(f'\nPVGround {record["time_cst"]} face decoder LOCAL_AST_ONLY draft published {heads[0]}, fourlocal+remote SHA{digest}; actual13CRLF index interruption corrected by scoped-text-rule/renormalization; no GPU claim. GoalACTIVE_UNMET.\n')
print(json.dumps(record), flush=True)
