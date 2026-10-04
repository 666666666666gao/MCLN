"""Publish the actual unintegrated face decoder after terminal publication."""
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
assert previous['full_pair_complete'] and previous['fresh_pair_audit_complete']
workspace = Path('C:/Users/gb')
repos = [workspace / '.codex_mcln_g0_20260905', workspace / '.codex_pvground_cs_20261002',
         workspace / '.codex_mcln_v99_internal_20260928']
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies = [repo / doc for repo in repos] + [workspace / 'Desktop/document' / Path(doc).name]
old = copies[0].read_bytes()
assert hashlib.sha256(old).hexdigest() == previous['handoff_sha256']
assert all(path.read_bytes() == old for path in copies)
for repo, head in zip(repos, previous['heads']):
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip() == head
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
guard = workspace / '.codex/tmp/sync_cs_handoff_remote_20260923.py'
guard_raw = guard.read_bytes()
assert guard_raw.count(previous['handoff_sha256'].encode()) == 1
names = ('pvground_face_conditioned_box_refiner.py', 'IMPLEMENTATION_SCOPE.md',
         'LOCAL_AST_CHECK.json', 'publish_source_draft.py')
prefix = 'refine-logs/pvground_face_conditioned_20261004/'
payloads = {prefix + name: (local / name).read_bytes() for name in names}
assert all(not (repo / relative).exists() for repo in repos[:2] for relative in payloads)
stamp = datetime.datetime.now().astimezone().isoformat()
section = f'''

## 20.376.46 六面条件化支撑解码器本地初稿（{stamp}）

在§20.376.45完整训练与审查记录之后，下一项解码器的独立源文件已经写出，但尚未接入真实PV模型工厂、配置、训练入口、GPU预检或正式训练。本次仅有LOCAL_AST_ONLY语法检查，0次原生模型构建、0次GPU前向、0次优化器更新，没有新的精度或参数量测量。不能把本地初稿当作已通过实验。

{(local / 'IMPLEMENTATION_SCOPE.md').read_text(encoding='utf-8').replace('# 六面条件化支撑解码：本地实现范围', '### 实际实现范围', 1)}

源文件与执行范围证据见refine-logs/pvground_face_conditioned_20261004；本轮度量最佳仍为已验证的普通分布头5616／4506，ScanRefer开发线及Nr3D／Sr3D目标仍未完成。原G、官方PV、必要V99依赖和当前最佳增量权重保持保留；本次不创建、下载或删除权重。
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
sftp.mkdir(remote + '/' + prefix.rstrip('/'))
for relative, raw in payloads.items():
    for repo in repos[:2]:
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    with sftp.open(remote + '/' + relative, 'wx') as stream:
        stream.write(raw)
    with sftp.open(remote + '/' + relative, 'rb') as stream:
        assert stream.read() == raw
for path in copies:
    path.write_bytes(new)
with sftp.open(remote + '/' + doc, 'wb') as stream:
    stream.write(new)
assert all(path.read_bytes() == new for path in copies)
with sftp.open(remote + '/' + doc, 'rb') as stream:
    assert stream.read() == new
sftp.close()
client.close()
heads = []
for index, repo in enumerate(repos):
    if index < 2:
        with (repo / 'MANIFEST.md').open('a', encoding='utf-8') as stream:
            stream.write(f'\n- {stamp} face-conditioned decoder unintegrated source draft; LOCAL_AST_ONLY, no new GPU result.\n')
    stage = [doc] + (['MANIFEST.md', *payloads] if index < 2 else [])
    changed = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage])
    subprocess.check_call(['git', '-C', str(repo), '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol',
                           'diff', '--cached', '--check', '--', *stage])
    for relative, raw in (payloads.items() if index < 2 else []):
        assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + relative]) == raw
    subprocess.check_call(['git', '-C', str(repo), 'commit', '-m', 'Record unintegrated face-conditioned support decoder draft'])
    heads.append(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip())
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
subprocess.check_call(['git', '-C', str(repos[0]), '-c', 'http.version=HTTP/1.1', 'push', 'origin', 'HEAD:main'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == heads[0]
for relative in (doc, *payloads):
    assert subprocess.check_output(['git', '-C', str(repos[0]), 'rev-parse', 'HEAD:' + relative]) == subprocess.check_output(['git', '-C', str(repos[1]), 'rev-parse', 'HEAD:' + relative])
digest = hashlib.sha256(new).hexdigest()
guard.write_bytes(guard_raw.replace(previous['handoff_sha256'].encode(), digest.encode()))
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), section='20.376.46',
              heads=heads, github_main=heads[0], handoff_sha256=digest, handoff_bytes=len(new),
              four_local_and_remote_equal=True, exact_committed_payloads=True, payload_count=len(payloads),
              execution_scope='LOCAL_AST_ONLY', native_factory_integrated=False,
              gpu_forwards=0, optimizer_updates=0, accuracy_evidence=False,
              retained_metric_best='distribution_5616_4506', weights_created=0,
              weights_downloaded=0, weights_deleted=0, goal_achieved=False)
(local / 'source_publication.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
with (workspace / 'memory/2026-10-04.md').open('a', encoding='utf-8') as stream:
    stream.write(f'\nPVGround {record["time_cst"]} face decoder LOCAL_AST_ONLY source draft published {heads[0]}, fourlocal+remote SHA{digest}; not integrated or GPU-tested. GoalACTIVE_UNMET.\n')
print(json.dumps(record), flush=True)
