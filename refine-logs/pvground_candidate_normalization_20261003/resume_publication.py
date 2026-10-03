"""Finish the interrupted publication without rerunning uploads or training."""
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import paramiko

workspace = Path(r'C:\Users\gb')
local = Path(__file__).parent
assert not (local / 'launch_publication.json').exists()
previous = json.loads((local.parent / 'pvground_candidate_consistency_20261003/complete_publication.json').read_bytes())
cleanup = json.loads((local / 'local_superseded_cleanup_receipt.json').read_bytes())
assert cleanup['deleted_count'] == 14 and cleanup['model_weights_only']
repos = [workspace / '.codex_mcln_g0_20260905', workspace / '.codex_pvground_cs_20261002',
         workspace / '.codex_mcln_v99_internal_20260928']
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies = [repo / doc for repo in repos] + [workspace / 'Desktop/document' / Path(doc).name]
pending = copies[0].read_bytes()
assert all(path.read_bytes() == pending for path in copies)
old = pending[:previous['handoff_bytes']]
assert hashlib.sha256(old).hexdigest() == previous['handoff_sha256'] and pending.startswith(old)
assert pending.count(b'### 20.376.29 ') == 1
for repo, head in zip(repos, previous['heads']):
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip() == head

now = datetime.datetime.now().astimezone().isoformat()
failure = dict(time_recorded_cst=now, script='publish_launch.py', native_session=14991,
    exit_code=1, failed_command_exit_code=2, stage='first repository git diff --cached --check, before any commit',
    reason='original review MD/JSON have extra EOF blank lines; unified diff evidence has required blank context lines with a leading space',
    remote_and_four_local_document_already_uploaded=True, training_restarted=False,
    raw_review_and_diff_preserved=True,
    recovery_precheck_failure='first recovery precheck compared raw handoff SHA against Git-normalized EOL blob and exited before mutations; corrected to verify the exact published raw prefix length/SHA',
    recovery='resume only staging/commits/push; exclude these exact three evidence files from source whitespace lint, retain exact-byte checks')
(local / 'publication_failure.json').write_text(json.dumps(failure, indent=2) + '\n', encoding='utf-8')
source = ast.parse((local / 'publish_launch.py').read_text(encoding='utf-8'))
assignment = next(node for node in source.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == 'names' for target in node.targets))
names = ast.literal_eval(assignment.value)
prefix = 'refine-logs/pvground_candidate_normalization_20261003/'
payloads = {prefix + name: local / name for name in names}
payloads.update({prefix + path.name: path for path in local.glob('observation_*.json')})
payloads.update({prefix + 'code_review_inputs/' + path.name: path for path in (local / 'code_review_inputs').iterdir()})
additions = ['resume_publication.py', 'publication_failure.json', 'local_superseded_cleanup_receipt.json', 'WEIGHT_RETENTION.md']
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
new = pending + ('\n发布补记：原发布在第一仓库的Git空白检查退出1，尚未产生提交；'
    '触发项是原审查报告的EOF空行和统一差异文件的合法空白上下文。原始报告和差异字节保留，'
    '仅这三份证据文件不参加源码空白lint，仍做完整Git字节核对；其余源码和文档检查保持。'
    '独立续接脚本完成后续提交和推送，没有重跑预检、训练或初始上传。失败与续接凭证保留。\n\n'
    f'按用户最新最佳权重规则，继续核对既有完整归档SHA，仅清理14份列明的过时/非最佳本地.pth，共{int(cleanup["deleted_bytes"])}字节；'
    f'加上本轮先前两份，累计16份3989174047字节，C盘实际余量{cleanup["local_c_free_after"]}字节。'
    '相应远端文件此前已经清理；原G、必要父权重、V99链及活动恢复不动。'
    '非模型权重的初始数值trace.pt和全部实验记录保留；旧归档回执与新删除回执一并披露。'
    '后续失败终点不长期积累无用归档，当前活动实验未变。\n').encode('utf-8')
for path in copies:
    path.write_bytes(new)
with sftp.open(remote + '/' + doc, 'wb') as stream:
    stream.write(new)
assert remote_bytes(remote + '/' + doc) == new and all(path.read_bytes() == new for path in copies)
sftp.close()
client.close()
heads = []
exempt = {prefix + name for name in ('EXPERIMENT_CODE_REVIEW.json', 'EXPERIMENT_CODE_REVIEW.md', 'implementation_diff.patch')}
for index, repo in enumerate(repos):
    if index == 1:
        with (repo / 'MANIFEST.md').open('a', encoding='utf-8') as stream:
            stream.write('\n- ' + now + ' /experiment-bridge: ' + prefix + ' normalization launched after native sanity; metric-best retention applied.\n')
    stage = [doc] + (['MANIFEST.md', *payloads] if index < 2 else [])
    status = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all']).decode()
    assert {line[3:] for line in status.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage])
    subprocess.check_call(['git', '-C', str(repo), 'diff', '--cached', '--check', '--', *[name for name in stage if name not in exempt]])
    for relative, path in (payloads.items() if index < 2 else []):
        assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + relative]) == path.read_bytes()
    subprocess.check_call(['git', '-C', str(repo), 'commit', '-m', 'Launch normalized candidate control and clean superseded own weight archives'])
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
    payload_count=len(payloads), GPU_preflight_pass=True, normalized_training_started=True,
    normalized_formal_results_available=False, scanrefer_target_pass=False, retained_best='original_g',
    review_independence='same-family', acceptance_status='provisional',
    initial_publication_failed_then_resumed=True, raw_review_and_diff_preserved=True,
    own_local_model_weights_deleted=16, own_local_model_weight_bytes_deleted=3989174047)
(local / 'launch_publication.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps(receipt))
