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
assert not (local / 'terminal_audit_publication.json').exists()
previous = json.loads((local / 'terminal_pending_publication.json').read_bytes())
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
call = json.loads((readback / 'FULL_SOURCE_REVIEW_CALL.json').read_bytes())
assert call['status'] == 'ACTUALLY_CALLED_RUNNING'
payloads = {}
for path in sorted((local / 'analysis').glob('EXPERIMENT_AUDIT*')):
    payloads[face_prefix + 'analysis/' + path.name] = path.read_bytes()
for name in ('ACTUAL_TERMINAL_REVIEWER_CALL.json', 'ACTUAL_TERMINAL_REVIEWER_RESPONSE.txt',
             'finalize_terminal_audit.py', 'publish_terminal_audit.py', 'prepare_audit_publication.py'):
    payloads[face_prefix + name] = (local / name).read_bytes()
for name in ('run_readback_preflight.py', 'readback_preflight_controller.py',
    'create_remote_readback_source.py', 'launch_readback_preflight_authorized.py',
    'prepare_preflight_bundle.py', 'EXPERIMENT_PLAN_READBACK.md',
    'evidence_hidden_preflight_template.json', 'evidence_visible_preflight_template.json',
    'PREFLIGHT_BUNDLE_SOURCE_CHECK.json', 'prepare_full_source_review.py',
    'FULL_SOURCE_REVIEW_REQUEST.txt', 'FULL_SOURCE_REVIEW_BINDINGS.json',
    'FULL_SOURCE_AST_CHECK.json', 'FULL_SOURCE_REVIEW_CALL.json', 'record_full_source_call.py'):
    payloads[readback_prefix + name] = (readback / name).read_bytes()
for path in sorted((readback / 'runtime_bundle').iterdir()):
    payloads[readback_prefix + 'runtime_bundle/' + path.name] = path.read_bytes()
assert all(len(raw) < 100000000 for raw in payloads.values())
assert not any('/.aris/' in name or name.endswith('.pth') for name in payloads)
stamp = datetime.datetime.now().astimezone().isoformat()
section = f'''

## 20.376.52 六面终态独立审计完成与固定最佳几何回写预检源码（{stamp}）

新的experiment-audit已经实际完成：WARN、无阻断项，same-family/provisional；A/B/C/F为PASS，D/E为WARN。独立读取70份主文件、50287条NDJSON并完成CPU重算，正式last/bbs仍为face5615/4496、flat5616/4506，目标5615/4754未达。WARN涉及已继承但未调用的辅助函数、单seed开发验证范围、两架构容量不同、跨进程数值差异，以及Mask/oracle仅保存标量复核、未重放原始Mask与全候选框。未发现GT伪造、指标拼接或不存在的结果。没有因此新增训练或重评估；精度与权重清理仍以§51真实终态为准。

下一项选择保留的4506平铺分布头作为几何提供者，固定官方PV、原G及几何头参数与eval状态；仅新回写单元学习。两组保持相同参数、完整文本、Query、六面角色和256候选，仅改变44维几何证据可见/置零。原生最终语义子头在精修后只调用一次，Mask与对比投影保留原路径。现已补齐实际CPU/两步GPU预检caller、严格三段权重工厂、孤立源覆盖、顺序控制器和部署入口；新32份完整来源审查已实际调用，当前仍待终态。旧部分源码PASS不覆盖新字节；此处只有Python3.7语法与源码准备，未CPU构建、未GPU更新、未启动正式回写训练，更没有新精度结果。

预检将检查零残差原生一致、框/Mask/对比冻结、语义头单次调用、实际bbs分数/排序/梯度、独立末层CE+G梯度、两步更新与内存模型/优化器恢复。不会创建临时磁盘权重。未来正式对照预算须同时记录有效batch与更新次数：29778条fit各一次、batch8，3722个完整batch加尾批2，共3723次更新；改变有效batch会改变同遍历下更新次数，不以学习率缩放代替预算核对。当前无Nr/Sr、教师、质量损失或部署双源结果。
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
new_directories = [readback_prefix + 'runtime_bundle']
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
            stream.write(f'\n- {stamp} face actual formal5615/4496, currentbest4506 protected; nonbest904318-byte weight removed; CPU recount and fresh integrity audit WARN/no blockers complete; readback full source gate pending. Revised readback/factory AST-only, full source/runtime gate pending.\n')
    stage = [doc] + (['MANIFEST.md', '.gitattributes', *payloads] if index < 2 else [])
    changed = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage])
    subprocess.check_call(['git', '-C', str(repo), '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol',
        'diff', '--cached', '--check', '--', *stage])
    for relative, raw in (payloads.items() if index < 2 else []):
        assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + relative]) == raw
    subprocess.check_call(['git', '-C', str(repo), 'commit', '-m', 'Record completed face integrity audit and full readback preflight source'])
    heads.append(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip())
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
subprocess.check_call(['git', '-C', str(repos[0]), '-c', 'http.version=HTTP/1.1', 'push', 'origin', 'HEAD:main'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == heads[0]
digest = hashlib.sha256(new).hexdigest()
guard.write_bytes(guard_raw.replace(previous['handoff_sha256'].encode(), digest.encode()))
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), section='20.376.52',
    predecessor=str(local / 'terminal_pending_publication.json'), heads=heads, github_main=heads[0],
    handoff_sha256=digest, handoff_bytes=len(new), four_local_and_remote_equal=True,
    exact_committed_payloads=True, payload_count=len(payloads), formal_bbs=[5615,4496],
    retained_metric_best=[5616,4506], terminal_audit_pending_at_publication=False, terminal_integrity_verdict='WARN', terminal_integrity_blockers=0, full_readback_source_review_pending=True,
    revised_readback_factory_constructed=False, revised_readback_gpu_checked=False, goal_achieved=False)
(local / 'terminal_audit_publication.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
state_path = local / 'active_continuation_state.json'
state = json.loads(state_path.read_bytes())
state.update(latest_publication=str(local / 'terminal_audit_publication.json'),
    github_publication_predecessor=str(local / 'terminal_audit_publication.json'), published_heads=heads,
    handoff_sha256=digest, handoff_section='20.376.52')
state_path.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
with (local / 'NEXT_CONTINUATION.md').open('a', encoding='utf-8') as stream:
    stream.write(f'\nLatest actual publication: {local / "terminal_audit_publication.json"}; main{heads[0]}; doc52. Actual terminal audit WARN/no blockers; full readback source review pending, revised readback/factory unconstructed. Use this doc/repo predecessor.\n')
with (workspace / 'memory/2026-10-04.md').open('a', encoding='utf-8') as stream:
    stream.write(f'\nPVGround {record["time_cst"]} face terminal5615/4496+CPU evidence and revised readback/factory drafts published {heads[0]}, doc52 fourlocal+remote raw equal. Actual terminal audit WARN/no blockers, full32-file readback source gate pending; no CPU/GPU/accuracy pass. Retain4506+parents, no live training. GoalACTIVE_UNMET.\n')
print(json.dumps(record), flush=True)
