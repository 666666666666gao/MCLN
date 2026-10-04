"""Publish the actual partial source gate. Never modify active training source."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import paramiko

local = Path(__file__).resolve().parent
assert not (local / 'review_publication.json').exists()
previous = json.loads((local / 'port_publication.json').read_bytes())
review = json.loads((local / 'READBACK_SOURCE_REVIEW.json').read_bytes())
call = json.loads((local / 'SOURCE_REVIEW_CALL.json').read_bytes())
assert call['status'] == 'ACTUALLY_COMPLETED' and review['verdict'] == 'PASS'
assert not review['blocking_findings'] and not review['required_source_patches']
assert review['execution_scope'] == 'SOURCE_ONLY'
for item in review['reviewed_files']:
    raw = Path(item['path']).read_bytes()
    assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
workspace = Path('C:/Users/gb')
repos = [workspace / '.codex_mcln_g0_20260905', workspace / '.codex_pvground_cs_20261002',
    workspace / '.codex_mcln_v99_internal_20260928']
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies = [repo / doc for repo in repos] + [workspace / 'Desktop/document' / Path(doc).name]
old = copies[0].read_bytes()
assert hashlib.sha256(old).hexdigest() == previous['handoff_sha256']
assert len(old) == previous['handoff_bytes'] and all(path.read_bytes() == old for path in copies)
for repo, head in zip(repos, previous['heads']):
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip() == head
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
guard = workspace / '.codex/tmp/sync_cs_handoff_remote_20260923.py'
guard_raw = guard.read_bytes()
assert guard_raw.count(previous['handoff_sha256'].encode()) == 1
prefix = 'refine-logs/pvground_geometry_readback_20261004/'
names = ['CODE_REVIEW_REQUEST.txt', 'SOURCE_REVIEW_REQUEST_BINDINGS.json', 'SOURCE_REVIEW_CALL.json',
    'READBACK_SOURCE_REVIEW.md', 'READBACK_SOURCE_REVIEW.json', 'ACTUAL_SOURCE_REVIEW_RESPONSE.txt',
    'finalize_source_review.py', 'publish_source_review.py']
payloads = {prefix + name: (local / name).read_bytes() for name in names}
assert all(not (repo / name).exists() for repo in repos[:2] for name in payloads)
stamp = datetime.datetime.now().astimezone().isoformat()
section = f'''

## 20.376.50 几何证据回写的独立源码评审（{stamp}）

已实际调用一次全新、无对话上下文的源码评审，要求Astra/max；实际后端型号与推理强度未独立见证，按same-family/provisional记录。评审核对18份既存文件的字节数和SHA，解析11份Python源码，结果为PASS/SOURCE_ONLY，未发现需要修复的源码缺陷。实际请求、绑定、完整报告与最终回复已保存；本地原始调用trace不提交仓库。

这次只审查独立回写单元、显式安装器、延后末层语义子头的两文件源码草稿与未使用的ScanRefer root bbs辅助函数。回写先读取六面分布和完整文本，再返回同一候选语义表示；末层原生语义子头仅在Mask与精修之后调用一次。没有独立质量排名、没有截断256候选，当前face训练源码未修改。没有批准未来模型工厂、完整runner或几何provider选择，也没有GPU等价性、恢复或精度结论。

未使用的bbs辅助函数与原生公式代数一致，但浮点归约顺序不同；未发现实测分数或排名差异，本轮未据此修改源码。接入真实runner之前仍须比较同一真实batch的全部256候选分数、排名和语义logit梯度。完整模型与优化器构建、真实两步检查及保存恢复均待执行，不将SOURCE_ONLY的PASS写成这些门槛已通过。

当前sole observer仍观察同一face训练；原G和5616/4506的当前最佳分布头权重链继续保护。新几何回写没有正式训练或新增REC结果，目标仍为ACTIVE_UNMET。
'''
new = old + section.encode('utf-8')
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
    password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
remote = '/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(remote + '/' + doc, 'rb') as stream:
    assert stream.read() == old
for relative, raw in payloads.items():
    for repo in repos[:2]:
        target = repo / relative
        target.write_bytes(raw)
    with sftp.open(remote + '/' + relative, 'wx') as stream:
        stream.write(raw)
    with sftp.open(remote + '/' + relative, 'rb') as stream:
        assert stream.read() == raw
for path in copies:
    path.write_bytes(new)
with sftp.open(remote + '/' + doc, 'wb') as stream:
    stream.write(new)
with sftp.open(remote + '/' + doc, 'rb') as stream:
    assert stream.read() == new
assert all(path.read_bytes() == new for path in copies)
sftp.close()
client.close()
heads = []
for index, repo in enumerate(repos):
    if index < 2:
        with (repo / 'MANIFEST.md').open('a', encoding='utf-8') as stream:
            stream.write(f'\n- {stamp} actual partial readback source review PASS/SOURCE_ONLY; 18 byte bindings, no required source patches, full factory/runner/runtime/accuracy pending.\n')
    stage = [doc] + (['MANIFEST.md', *payloads] if index < 2 else [])
    changed = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage])
    subprocess.check_call(['git', '-C', str(repo), '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol',
        'diff', '--cached', '--check', '--', *stage])
    for relative, raw in (payloads.items() if index < 2 else []):
        assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + relative]) == raw
    subprocess.check_call(['git', '-C', str(repo), 'commit', '-m', 'Record actual partial source review for boundary evidence readback'])
    heads.append(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip())
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
subprocess.check_call(['git', '-C', str(repos[0]), '-c', 'http.version=HTTP/1.1', 'push', 'origin', 'HEAD:main'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == heads[0]
digest = hashlib.sha256(new).hexdigest()
guard.write_bytes(guard_raw.replace(previous['handoff_sha256'].encode(), digest.encode()))
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), section='20.376.50',
    predecessor=str(local / 'port_publication.json'), heads=heads, github_main=heads[0],
    handoff_sha256=digest, handoff_bytes=len(new), four_local_and_remote_equal=True,
    exact_committed_payloads=True, payload_count=len(payloads), execution_scope='SOURCE_ONLY',
    reviewer_verdict='PASS', required_source_patches=0, runtime_checked=False, accuracy_result=False,
    active_face_fit_changed=False, retained_metric_best='plain_distribution_5616_4506', goal_achieved=False)
(local / 'review_publication.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
state_path = local.parent / 'pvground_face_conditioned_20261004/active_continuation_state.json'
state = json.loads(state_path.read_bytes())
state.update(latest_publication=str(local / 'review_publication.json'),
    github_publication_predecessor=str(local / 'review_publication.json'), published_heads=heads,
    handoff_sha256=digest, handoff_section='20.376.50')
state['next_actions'] = [item.replace('geometry_readback/source_publication.json', 'geometry_readback/review_publication.json')
    for item in state['next_actions']]
state_path.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
cursor = local.parent / 'pvground_face_conditioned_20261004/NEXT_CONTINUATION.md'
with cursor.open('a', encoding='utf-8') as stream:
    stream.write(f'\nLatest actual publication: {local / "review_publication.json"}; main {heads[0]}; doc50. Use this predecessor for terminal publication. Sole observer806/native84531 remains unchanged. Readback source-only review PASS; full factory/runner/runtime/accuracy pending.\n')
with (workspace / 'memory/2026-10-04.md').open('a', encoding='utf-8') as stream:
    stream.write(f'\nPVGround {record["time_cst"]}: actual partial readback source review PASS/SOURCE_ONLY published {heads[0]}, doc50 fourlocal+remote byteequal. No factory/GPU/accuracy approval; active face run unchanged; resume sole exec806 only. GoalACTIVE_UNMET.\n')
print(json.dumps(record), flush=True)
