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
assert not (local / 'terminal_pending_publication.json').exists()
previous = json.loads((readback / 'review_publication.json').read_bytes())
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
payloads = {}
for relative, item in intake['files'].items():
    raw = (local / 'complete' / relative).read_bytes()
    assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
    payloads[face_prefix + 'complete/' + relative] = raw
for name in ('complete/INTAKE.json', 'analysis/SUMMARY.json', 'analysis/REPORT.md',
    'collect_terminal.py', 'analyze_terminal.py', 'prepare_terminal_audit_request.py',
    'TERMINAL_AUDIT_REQUEST.txt', 'ACTUAL_TERMINAL_REVIEWER_CALL.json',
    'record_terminal_audit_call.py', 'wait.json', 'publish_terminal_pending_audit.py'):
    payloads[face_prefix + name] = (local / name).read_bytes()
for name in ('pvground_boundary_evidence_readback.py', 'install_boundary_evidence_readback.py',
    'native_root_bbs.py', 'readback_preflight_checks.py', 'readback_model_factory.py',
    'READBACK_PREFLIGHT_SOURCE_SCOPE.md', 'READBACK_PREFLIGHT_SOURCE_CHECK.json',
    'check_preflight_revision.py'):
    payloads[readback_prefix + name] = (readback / name).read_bytes()
assert all(len(raw) < 100000000 for raw in payloads.values())
assert not any('/.aris/' in name or name.endswith('.pth') for name in payloads)
stamp = datetime.datetime.now().astimezone().isoformat()
formal = summary['stages']['formal']['bbs']
effect = formal['same_query_refinement']
section = f'''

## 20.376.51 六面条件化训练终态与回写预检准备（{stamp}）

face实验实际于20:08:33 CST封闭，控制器与原观察器均正常退出。原G固定/eval，仅64737参数、25项新增头状态训练；29778条fit各一次，batch8/tail2，共3723次更新。完整9508条原生last/bbs为5615/4496（59.0555%/47.2865%），比当前最佳5616/4506少1/10，比原G5615/4495为0/+1，未达到5615/4754目标。它与456102参数的平铺分布头同时改变了架构和容量，不能将差值单独归因于方向条件。没有新回写、质量监督、教师或Nr/Sr结果。

终态已收集38份文本/逐行文件，未下载权重。CPU重新计算选中框IoU与REC计数无阈值变化；同Query严格精修前后为{effect['coarse_hits50']}/4496，修复{effect['repairs50']}、破坏{effect['damages50']}。原始Mask和全候选框未独立重放，Mask与oracle仅重数原生保存值。新的独立experiment-audit已实际调用，same-family/provisional；本节发布时最终审计仍待返回，不写成审计PASS。

控制器依据原生主指标核验后，已删除本轮904318字节的非最佳terminal，SHA为d09be40e67b19ba2ff4fa50eec9ff4e912f32c1478bf80372d0a098a16044428，没有创建本地失败权重归档。保留4506的分布头及其必要原G/官方PV父链，V99链未动。收集时数据盘余量为{intake['directory_free_bytes']}字节，属于该时刻快照。

后续以当前4506几何模型固定框和Mask，准备同容量的几何证据可见/置零回写对照。两组保留完整文本、Query、六面角色和唯一原生语义头；只改变44维几何输入是否可见，保留全部256候选。bbs辅助函数改为原生逐map归约顺序；此前未测出实际分数错误。新预检辅助代码检查语义头单次调用、几何/Mask/对比输出保持、真实bbs分数/排名/梯度，以及单独最后层CE+G的任务梯度。

独立工厂草稿写明官方PV→原G→4506分布头的严格加载顺序，冻结这条链后安装回写单元。工厂仅通过Python3.7语法检查，尚未CPU构建、部署、GPU预检、保存恢复或训练；回写96672参数/23项仍是源码算数，未真实测量。先前源码PASS只覆盖旧字节，修订后的完整工厂/runner仍须新源码评审。当前没有新的优化器任务在GPU上运行，不把这些源码准备写成性能增益。
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
new_directories = sorted({str(Path(name).parent).replace('\\', '/') for name in payloads
    if name.startswith(face_prefix + 'complete/') or name.startswith(face_prefix + 'analysis/')},
    key=lambda name: (name.count('/'), name))
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
        with (repo / '.gitattributes').open('a', encoding='utf-8') as stream:
            stream.write('\n# Preserve the exact collected face evidence, including historical log whitespace.\n' +
                face_prefix + 'complete/** -text whitespace=cr-at-eol,-blank-at-eol,-blank-at-eof\n')
        with (repo / 'MANIFEST.md').open('a', encoding='utf-8') as stream:
            stream.write(f'\n- {stamp} face actual formal5615/4496, currentbest4506 protected; nonbest904318-byte weight removed; CPU recount complete, fresh integrity audit pending. Revised readback/factory AST-only, full source/runtime gate pending.\n')
    stage = [doc] + (['MANIFEST.md', '.gitattributes', *payloads] if index < 2 else [])
    changed = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage])
    subprocess.check_call(['git', '-C', str(repo), '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol',
        'diff', '--cached', '--check', '--', *stage])
    for relative, raw in (payloads.items() if index < 2 else []):
        assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + relative]) == raw
    subprocess.check_call(['git', '-C', str(repo), 'commit', '-m', 'Record face boundary terminal and prepare isolated geometry readback checks'])
    heads.append(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip())
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
subprocess.check_call(['git', '-C', str(repos[0]), '-c', 'http.version=HTTP/1.1', 'push', 'origin', 'HEAD:main'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == heads[0]
digest = hashlib.sha256(new).hexdigest()
guard.write_bytes(guard_raw.replace(previous['handoff_sha256'].encode(), digest.encode()))
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), section='20.376.51',
    predecessor=str(readback / 'review_publication.json'), heads=heads, github_main=heads[0],
    handoff_sha256=digest, handoff_bytes=len(new), four_local_and_remote_equal=True,
    exact_committed_payloads=True, payload_count=len(payloads), formal_bbs=[5615,4496],
    retained_metric_best=[5616,4506], terminal_audit_pending_at_publication=True,
    revised_readback_factory_constructed=False, revised_readback_gpu_checked=False, goal_achieved=False)
(local / 'terminal_pending_publication.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
state_path = local / 'active_continuation_state.json'
state = json.loads(state_path.read_bytes())
state.update(latest_publication=str(local / 'terminal_pending_publication.json'),
    github_publication_predecessor=str(local / 'terminal_pending_publication.json'), published_heads=heads,
    handoff_sha256=digest, handoff_section='20.376.51')
state_path.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
with (local / 'NEXT_CONTINUATION.md').open('a', encoding='utf-8') as stream:
    stream.write(f'\nLatest actual publication: {local / "terminal_pending_publication.json"}; main{heads[0]}; doc51. Audit pending at this snapshot, revised readback/factory unconstructed. Use this doc/repo predecessor.\n')
with (workspace / 'memory/2026-10-04.md').open('a', encoding='utf-8') as stream:
    stream.write(f'\nPVGround {record["time_cst"]} face terminal5615/4496+CPU evidence and revised readback/factory drafts published {heads[0]}, doc51 fourlocal+remote raw equal. Fresh terminal audit pending; no revised readback CPU/GPU/accuracy pass. Retain4506+parents, no live training. GoalACTIVE_UNMET.\n')
print(json.dumps(record), flush=True)
