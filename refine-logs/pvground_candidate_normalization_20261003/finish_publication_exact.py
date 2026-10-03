"""Finish the prepared publication with byte-preserving experiment artifacts."""
import ast
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
copies = [repo / doc for repo in repos] + [workspace / 'Desktop/document' / Path(doc).name]
pending = copies[0].read_bytes()
assert all(path.read_bytes() == pending for path in copies)
assert hashlib.sha256(pending[:previous['handoff_bytes']]).hexdigest() == previous['handoff_sha256']
assert pending.count(b'### 20.376.29 ') == 1
for repo, head in zip(repos, previous['heads']):
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip() == head
eol = json.loads((local / 'publication_eol_mismatches.json').read_bytes())
assert len(eol['mismatches']) == 13 and all(row['only_eol_difference'] for row in eol['mismatches'])
tree = ast.parse((local / 'publish_launch.py').read_text(encoding='utf-8'))
names = ast.literal_eval(next(node.value for node in tree.body if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == 'names' for target in node.targets)))
prefix = 'refine-logs/pvground_candidate_normalization_20261003/'
names += ['resume_publication.py', 'publication_failure.json', 'local_superseded_cleanup_receipt.json', 'WEIGHT_RETENTION.md']
payloads = {prefix + name: local / name for name in names}
payloads.update({prefix + path.name: path for path in local.glob('observation_*.json')})
payloads.update({prefix + 'code_review_inputs/' + path.name: path for path in (local / 'code_review_inputs').iterdir()})
now = datetime.datetime.now().astimezone().isoformat()
failure = dict(time_recorded_cst=now, script='resume_publication.py', native_session=37538,
    exit_code=1, stage='first repository exact staged payload check, before any commit',
    actual_mismatches=eol['mismatches'], all_mismatches_only_EOL=True,
    model_or_training_modified=False,
    resolution='own experiment directory .gitattributes * -text preserves raw source/spec/receipts; CRLF-aware source whitespace lint; exact-byte checks remain mandatory')
(local / 'publication_stage_failure.json').write_bytes((json.dumps(failure, indent=2) + '\n').encode('utf-8'))
additions = ['.gitattributes', 'finish_publication_exact.py', 'publication_eol_mismatches.json', 'publication_stage_failure.json']
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
    raw = path.read_bytes()
    assert all((repo / relative).read_bytes() == raw for repo in repos[:2])
    assert remote_bytes(remote + '/' + relative) == raw
for name in additions:
    relative = prefix + name
    path = local / name
    raw = path.read_bytes()
    for repo in repos[:2]:
        target = repo / relative
        assert not target.exists()
        target.write_bytes(raw)
    with sftp.open(remote + '/' + relative, 'wx') as stream:
        stream.write(raw)
    assert remote_bytes(remote + '/' + relative) == raw
    payloads[relative] = path
new = pending + ('\n发布校验补充：第一续接脚本在提交前的Git字节核对中发现13份记录仅因CRLF→LF转换不一致，实际差异列表保留。'
    '单独完成脚本仅在本次实验refine-logs目录使用.gitattributes的* -text，保留原始源码、规格和回执字节，'
    '不改变根仓库换行配置或模型文件；源码空白检查允许实际CRLF行尾，三份合法原始证据仍不参加空白lint，完整字节检查继续执行。'
    '没有重新运行实验或先前上传阶段。唯一交接的工作副本、桌面及远端仍按原始字节一致，Git文档换行沿用既有规则，'
    '实验payload则逐一按Git blob与原件完全一致验收。\n').encode('utf-8')
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
            stream.write('\n- ' + now + ' /experiment-bridge: ' + prefix + ' normalization launched after native sanity; metric-best retention applied.\n')
    stage = [doc] + (['MANIFEST.md', *payloads] if index < 2 else [])
    status = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all']).decode()
    assert {line[3:] for line in status.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage])
    subprocess.check_call(['git', '-C', str(repo), '-c',
        'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol',
        'diff', '--cached', '--check', '--', *[name for name in stage if name not in exempt]])
    for relative, path in (payloads.items() if index < 2 else []):
        assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + relative]) == path.read_bytes(), relative
    subprocess.check_call(['git', '-C', str(repo), 'commit', '-m', 'Launch normalized candidate control and retain strongest model weights'])
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
receipt = dict(section='20.376.29', time_cst=datetime.datetime.now().astimezone().isoformat(),
    github_main=heads[0], heads=heads, handoff_sha256=digest, handoff_bytes=len(new),
    prior_prefix_unchanged=True, four_local_and_remote_equal=True, exact_committed_evidence=True,
    Git_document_EOL_uses_existing_rules=True, payload_count=len(payloads),
    GPU_preflight_pass=True, normalized_training_started=True, normalized_formal_results_available=False,
    scanrefer_target_pass=False, retained_best='original_g', review_independence='same-family',
    acceptance_status='provisional', publication_failures_preserved=True,
    own_local_model_weights_deleted=16, own_local_model_weight_bytes_deleted=3989174047)
(local / 'launch_publication.json').write_bytes((json.dumps(receipt, indent=2) + '\n').encode('utf-8'))
print(json.dumps(receipt))
