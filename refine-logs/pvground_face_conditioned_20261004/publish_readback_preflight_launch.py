"""Publish actual terminal records and isolated source preparation; no invented PASS."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import paramiko

local = Path(__file__).resolve().parent
readback = local.parent / 'pvground_geometry_readback_20261004'
assert not (local / 'readback_launch_publication.json').exists()
previous = json.loads((local / 'terminal_audit_publication.json').read_bytes())
intake = json.loads((local / 'complete/INTAKE.json').read_bytes())
summary = json.loads((local / 'analysis/SUMMARY.json').read_bytes())
assert intake['controller_exit'] == 0 and not intake['controller_alive']
assert intake['status']['status'] == 'complete'
assert summary['stages']['formal']['bbs']['rec_hits25'] == 5615
assert summary['stages']['formal']['bbs']['rec_hits50'] == 4496
assert not summary['scanrefer_target_pass']
assert intake['status']['retained_best']['bbs_hits50'] == 4506
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
face_prefix = 'refine-logs/pvground_face_conditioned_20261004/'
readback_prefix = 'refine-logs/pvground_geometry_readback_20261004/'
audit = json.loads((local / 'analysis/EXPERIMENT_AUDIT.json').read_bytes())
assert audit['verdict'] == 'WARN' and not audit['blocking_issues']
assert audit['primary_files_read'] == 70
review = json.loads((readback / 'READBACK_FULL_SOURCE_REVIEW.json').read_bytes())
launch = json.loads((readback / 'readback_preflight_launch.json').read_bytes())
assert review['verdict'] == 'PASS' and not review['blocking_findings']
assert review['execution_scope'] == 'SOURCE_ONLY'
assert launch['execution_status'] == 'LAUNCHED_NOT_COMPLETED'
payloads = {}
for name in ('READBACK_FULL_SOURCE_REVIEW.md', 'READBACK_FULL_SOURCE_REVIEW.json',
    'ACTUAL_FULL_SOURCE_REVIEW_RESPONSE.txt', 'FULL_SOURCE_REVIEW_CALL.json',
    'finalize_full_source_review.py', 'readback_preflight_launch.json',
    'readback_preflight_resource_check.json', 'readback_remote_source_receipt.json',
    'SEALED_READBACK_SOURCE_PORT.json', 'evidence_hidden_preflight_spec.json',
    'evidence_visible_preflight_spec.json', 'wait_readback_preflight_authorized.py',
    'collect_readback_preflight.py', 'prepare_launch_publication.py'):
    payloads[readback_prefix + name] = (readback / name).read_bytes()
payloads[face_prefix + 'publish_readback_preflight_launch.py'] = (local / 'publish_readback_preflight_launch.py').read_bytes()
assert not any('/.aris/' in name or name.endswith('.pth') for name in payloads)
stamp = datetime.datetime.now().astimezone().isoformat()
section = f'''

## 20.376.53 固定4506几何回写完整预检源码审查通过并实际启动（{stamp}）

完整32份来源已由新的Codex原生审查员实际直接读取，SOURCE_ONLY PASS，无正确性阻断或非阻断项，same-family/provisional。审查员可选本地AST调用因Python启动器失败未执行；主端此前已实际执行Python3.7 AST检查，二者不混淆。PASS仅覆盖CPU构建和两步预检源码，不是CPU/GPU实测通过、正式训练授权见证或精度结果。实际最终响应与私有trace已保存；旧部分源码审查不替代这次完整审查。

预检控制器实际于{launch['time_cst']}启动，PID {launch['process'].split()[0]}，根目录为{launch['root']}。先evidence_hidden，后evidence_visible，每组CPU严格构建后仅做两次真实GPU更新。模型使用已保留的4506平铺分布头及原G/官方PV父链，父模型与几何保持冻结/eval，新回写单元学习。两组只改变44维证据是否可见，全256候选、完整文本、相同参数及原生唯一评分保留。源覆盖位于全新隔离目录，原已验证源码未改；sealed源SHA为{launch['sealed_source']['source_port_sha256']}。

启动时实测GPU无计算进程，旧控制器已关闭，数据盘剩余{launch['resources']['directory_free_bytes']}字节、系统盘{launch['resources']['system_free_bytes']}字节，均只代表启动快照。预检不创建磁盘权重。唯一只读观察器已安排首次22:25 CST附近、之后240秒复查，不重复SSH轮询、不重启任务。此发布尚未读取实际预检终态；不能把已启动写成两组通过或正式训练已经开始。保留最佳5616/4506及必要父链，ScanRefer目标5615/4754仍未达，无新Nr/Sr结果。
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
new_directories = []
for directory in new_directories:
    sftp.mkdir(remote + '/' + directory)
for relative, raw in payloads.items():
    for repo in repos[:2]:
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    with sftp.open(remote + '/' + relative, 'wb') as stream:
        stream.write(raw)
    with sftp.open(remote + '/' + relative, 'rb') as stream:
        assert stream.read() == raw
for path in copies:
    path.write_bytes(new)
with sftp.open(remote + '/' + doc, 'wb') as stream:
    stream.write(new)
with sftp.open(remote + '/' + doc, 'rb') as stream:
    assert stream.read() == new
sftp.close()
client.close()
assert all(path.read_bytes() == new for path in copies)
heads = []
for index, repo in enumerate(repos):
    if index < 2:
        with (repo / 'MANIFEST.md').open('a', encoding='utf-8') as stream:
            stream.write(f'\n- {stamp} face actual formal5615/4496, currentbest4506 protected; nonbest904318-byte weight removed; CPU recount and fresh integrity audit WARN/no blockers complete; readback full source gate passed; actual preflight pending. Full source gate passed; actual readback preflight launched, runtime terminal pending.\n')
    stage = [doc] + (['MANIFEST.md', '.gitattributes', *payloads] if index < 2 else [])
    changed = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage])
    subprocess.check_call(['git', '-C', str(repo), '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol',
        'diff', '--cached', '--check', '--', *stage])
    for relative, raw in (payloads.items() if index < 2 else []):
        assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + relative]) == raw
    subprocess.check_call(['git', '-C', str(repo), 'commit', '-m', 'Record passed full readback source gate and actual two-arm preflight launch'])
    heads.append(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip())
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
subprocess.check_call(['git', '-C', str(repos[0]), '-c', 'http.version=HTTP/1.1', 'push', 'origin', 'HEAD:main'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == heads[0]
digest = hashlib.sha256(new).hexdigest()
guard.write_bytes(guard_raw.replace(previous['handoff_sha256'].encode(), digest.encode()))
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), section='20.376.53',
    predecessor=str(local / 'terminal_audit_publication.json'), heads=heads, github_main=heads[0],
    handoff_sha256=digest, handoff_bytes=len(new), four_local_and_remote_equal=True,
    exact_committed_payloads=True, payload_count=len(payloads), formal_bbs=[5615,4496],
    retained_metric_best=[5616,4506], terminal_audit_pending_at_publication=False, terminal_integrity_verdict='WARN', terminal_integrity_blockers=0, full_readback_source_review_pending=False, full_readback_source_review='PASS_SOURCE_ONLY', actual_preflight_launched=True,
    revised_readback_factory_constructed=False, revised_readback_gpu_checked=False, goal_achieved=False)
(local / 'readback_launch_publication.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
state_path = local / 'active_continuation_state.json'
state = json.loads(state_path.read_bytes())
state.update(latest_publication=str(local / 'readback_launch_publication.json'),
    github_publication_predecessor=str(local / 'readback_launch_publication.json'), published_heads=heads,
    handoff_sha256=digest, handoff_section='20.376.53')
state_path.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
with (local / 'NEXT_CONTINUATION.md').open('a', encoding='utf-8') as stream:
    stream.write(f'\nLatest actual publication: {local / "readback_launch_publication.json"}; main{heads[0]}; doc53. Actual terminal audit WARN/no blockers; full source gate PASS_SOURCE_ONLY; actual preflight launched, terminal pending. Use this doc/repo predecessor.\n')
with (workspace / 'memory/2026-10-04.md').open('a', encoding='utf-8') as stream:
    stream.write(f'\nPVGround {record["time_cst"]} face terminal5615/4496+CPU evidence and revised readback/factory drafts published {heads[0]}, doc53 fourlocal+remote raw equal. Actual terminal audit WARN/no blockers, full32-file source gate PASS; actual preflight launched, terminal pending; no CPU/GPU/accuracy pass. Retain4506+parents, no live training. GoalACTIVE_UNMET.\n')
print(json.dumps(record), flush=True)
