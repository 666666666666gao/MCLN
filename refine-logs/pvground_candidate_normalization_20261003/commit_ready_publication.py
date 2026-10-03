"""Commit the already uploaded bundle after scoped Git renormalization."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import paramiko

workspace = Path(r'C:\Users\gb')
local = Path(__file__).parent
assert not (local / 'launch_publication.json').exists()
previous = json.loads((local.parent / 'pvground_candidate_consistency_20261003/complete_publication.json').read_bytes())
repos = [workspace / '.codex_mcln_g0_20260905', workspace / '.codex_pvground_cs_20261002',
         workspace / '.codex_mcln_v99_internal_20260928']
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
prefix = 'refine-logs/pvground_candidate_normalization_20261003/'
copies = [repo / doc for repo in repos] + [workspace / 'Desktop/document' / Path(doc).name]
pending = copies[0].read_bytes()
assert all(path.read_bytes() == pending for path in copies)
assert hashlib.sha256(pending[:previous['handoff_bytes']]).hexdigest() == previous['handoff_sha256']
for repo, head in zip(repos, previous['heads']):
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip() == head
now = datetime.datetime.now().astimezone().isoformat()
failure = dict(time_recorded_cst=now, script='finish_publication_exact.py', native_session=51718,
    exit_code=1, stage='first repository run.py staged-byte comparison, before any commit',
    cause='new byte-preserving attribute did not refresh an already staged file until explicit git add --renormalize',
    actual_fix_verified='scoped git add --renormalize; runner index bytes equal original; no file content changed',
    training_modified=False)
(local / 'publication_renormalize_failure.json').write_bytes((json.dumps(failure, indent=2) + '\n').encode())
payloads = {path.relative_to(repos[0]).as_posix(): local / path.relative_to(repos[0] / prefix)
            for path in (repos[0] / prefix).rglob('*') if path.is_file()}
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
remote = '/home/gb/new butd/butd_detr-main/MCLN-main'

def remote_bytes(path):
    with sftp.open(path, 'rb') as stream:
        return stream.read()

assert remote_bytes(remote + '/' + doc) == pending
for relative, path in payloads.items():
    assert all((repo / relative).read_bytes() == path.read_bytes() for repo in repos[:2])
    assert remote_bytes(remote + '/' + relative) == path.read_bytes()
for name in ('commit_ready_publication.py', 'publication_renormalize_failure.json'):
    relative = prefix + name
    path = local / name
    for repo in repos[:2]:
        target = repo / relative
        assert not target.exists()
        target.write_bytes(path.read_bytes())
    with sftp.open(remote + '/' + relative, 'wx') as stream:
        stream.write(path.read_bytes())
    assert remote_bytes(remote + '/' + relative) == path.read_bytes()
    payloads[relative] = path
new = pending + ('\n最终发布续接：新增目录属性后，旧索引尚未刷新而再次在提交前的run.py字节核对退出1；'
    '只对本次实验记录路径显式git add --renormalize，已实际核对runner索引字节与原件相同，文件内容未改。'
    '提交脚本保留上述失败记录，并在三个仓库提交与main推送前逐一核对全部payload原始字节。活动训练没有重启或修改。\n').encode('utf-8')
for path in copies:
    path.write_bytes(new)
with sftp.open(remote + '/' + doc, 'wb') as stream:
    stream.write(new)
assert remote_bytes(remote + '/' + doc) == new and all(path.read_bytes() == new for path in copies)
sftp.close()
client.close()
exempt = {prefix + name for name in ('EXPERIMENT_CODE_REVIEW.json', 'EXPERIMENT_CODE_REVIEW.md', 'implementation_diff.patch')}
heads = []
for index, repo in enumerate(repos):
    if index == 1:
        with (repo / 'MANIFEST.md').open('a', encoding='utf-8') as stream:
            stream.write('\n- ' + now + ' /experiment-bridge: ' + prefix + ' normalized candidate control and best-only own weight retention.\n')
    stage = [doc] + (['MANIFEST.md', *payloads] if index < 2 else [])
    status = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all']).decode()
    assert {line[3:] for line in status.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage])
    if index < 2:
        subprocess.check_call(['git', '-C', str(repo), 'add', '--renormalize', '--', prefix.rstrip('/')])
    subprocess.check_call(['git', '-C', str(repo), '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol',
        'diff', '--cached', '--check', '--', *[name for name in stage if name not in exempt]])
    for relative, path in (payloads.items() if index < 2 else []):
        assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + relative]) == path.read_bytes(), relative
    subprocess.check_call(['git', '-C', str(repo), 'commit', '-m', 'Run expanded-count candidate control and clean nonbest weight archives'])
    heads.append(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip())
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
subprocess.check_call(['git', '-C', str(repos[0]), 'push', 'origin', 'HEAD:main'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == heads[0]
for relative in (doc, *payloads):
    assert subprocess.check_output(['git', '-C', str(repos[0]), 'rev-parse', 'HEAD:' + relative]) == subprocess.check_output(['git', '-C', str(repos[1]), 'rev-parse', 'HEAD:' + relative])
digest = hashlib.sha256(new).hexdigest()
guard = workspace / '.codex/tmp/sync_cs_handoff_remote_20260923.py'
raw = guard.read_bytes()
assert raw.count(previous['handoff_sha256'].encode()) == 1
guard.write_bytes(raw.replace(previous['handoff_sha256'].encode(), digest.encode()))
receipt = dict(section='20.376.29', time_cst=datetime.datetime.now().astimezone().isoformat(), github_main=heads[0],
    heads=heads, handoff_sha256=digest, handoff_bytes=len(new), prior_prefix_unchanged=True,
    four_local_and_remote_equal=True, exact_committed_evidence=True, payload_count=len(payloads),
    Git_document_EOL_uses_existing_rules=True, GPU_preflight_pass=True, normalized_training_started=True,
    normalized_formal_results_available=False, scanrefer_target_pass=False, retained_best='original_g',
    review_independence='same-family', acceptance_status='provisional', publication_failures_preserved=True,
    own_local_model_weights_deleted=16, own_local_model_weight_bytes_deleted=3989174047)
(local / 'launch_publication.json').write_bytes((json.dumps(receipt, indent=2) + '\n').encode())
print(json.dumps(receipt))
