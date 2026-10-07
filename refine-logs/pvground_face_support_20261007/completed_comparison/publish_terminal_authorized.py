"""Publish this closed paired result and actual standing-authorized cleanup."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

import paramiko


root = Path(__file__).resolve().parent
receipt_path = root / 'terminal_publication.json'
assert not receipt_path.exists()
review = json.loads((root / 'PUBLISH_TERMINAL_SOURCE_REVIEW.json').read_bytes())
assert review['execution_scope'] == 'SOURCE_ONLY'
assert review['verdict'] in ('PASS', 'WARN') and not review['blocking_findings']
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
summary = json.loads((root / 'analysis/SUMMARY.json').read_bytes())
audit = json.loads((root / 'analysis/EXPERIMENT_AUDIT.json').read_bytes())
claim = json.loads((root / 'analysis/RESULT_TO_CLAIM.json').read_bytes())
wait = json.loads((root / 'fit_wait.json').read_bytes())
retention = json.loads((root / 'weight_retention.json').read_bytes())
archive_path = root / 'archived_array_cleanup_receipt.json'
archive = json.loads(archive_path.read_bytes())
policy = json.loads((root / 'CLEANUP_POLICY.json').read_bytes())
assert audit['fresh_context'] and audit['verdict'] in ('PASS', 'WARN') and not audit['blocking_findings']
assert claim['claim_supported'] == 'no'
assert summary['metric_best_candidate']['arm'] == 'protected_geometry_parent'
assert retention['retained_best']['hits'] == [5598, 4848]
assert retention['status'] == 'CLOSED_NONBEST_WEIGHTS_REMOVED' and len(retention['deleted']) == 2
assert retention['released_bytes'] == 11172106 and retention['original_G_preserved']
assert retention['official_PV_paths_touched'] == retention['V99_paths_touched'] == 0
assert retention['fresh_audit_sha256'] == hashlib.sha256((root / 'analysis/EXPERIMENT_AUDIT.json').read_bytes()).hexdigest()
assert archive['status'] == 'STANDING_AUTHORIZED_ARCHIVED_ARRAYS_REMOVED'
assert archive['deleted_count'] == 2378 and archive['released_file_bytes'] == 330047051
assert archive['local_archive_preserved'] and archive['best_weights_touched'] == archive['text_logs_touched'] == archive['datasets_touched'] == 0
assert policy['future_repeated_approval_required'] is False
assert wait['observer_closed'] and wait['terminal']['exitcode'] == 0 and not wait['terminal']['controller_alive']
assert wait['terminal']['status']['completed_modes'] == ['initial_formal', 'train', 'formal']
assert summary['independent_iou_values'] == 19472384
for row in summary['table']:
    if row['stage'] == 'formal':
        assert row['optimizer_updates'] == 3723 and [row['rec_hits25'], row['rec_hits50']] == [5593, 4832]
previous = json.loads((root / 'fit_start_publication.json').read_bytes())
assert previous['section'] == '20.376.95'
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
stamp = datetime.datetime.now().astimezone().isoformat()
prefix = 'refine-logs/pvground_face_support_20261007/completed_comparison/'
section = f"""

## 20.376.96 六面区域读取配对实验终态、独立复核与自动清理（{stamp}）

本轮controller于{wait['terminal']['status']['finished_cst']}实际完成initial_formal、train、formal三阶段，退出0；每臂29778条fit各一次、3723次更新，固定seed2027、B8/累积1、LR1e-5、WD5e-4、clip0.1。相同冻结PV/G父模型的一次前向提供共同Query、原生bbs、Mask与参考，两臂独立训练同容量456102参数几何头；控制为中心及六个面中心的近邻读取，策略为bounded-face矩形距离选成员。后者不等于均匀覆盖整面、固定薄层或显式前景/背景平衡。无新评分、教师、父模型更新或多seed。

完整9508条正式验证：初始两臂均5598/4848（58.8767%/50.9886%）；训练后face_center、face_region均5593/4832（58.8241%/50.8204%），逐表达两阈值命中集合相同，臂间修复/破坏均0。相对同次前向参考，每臂@0.25修复4、破坏9，严格修复13、破坏29，净变化−5/−16。两臂实际框数值不同，不能解释成未接入或相同权重。6887条作者预训练见过场景的内部留出初始6149/5667，终点控制6147/5665、策略6146/5666，其−1/+1不构成正式验证增益。所选Mask仍5812/5133，mIoU47.107628%，为保存IoU重计，未重新生成raw masks。

实际归档2421份文件；分析器一次CPU重算19,472,384个float64 IoU，独立fresh reviewer再从2378份NPZ和日志核验，阈值翻转及oracle标签差异均0。审查WARN、blocking findings=0、确定性证据PASS；RESULT_TO_CLAIM为no。审查是same-family/provisional，实际backend/effort身份未认证。两份训练终点已分别严格CPU恢复声明的sampler、10个几何状态及Adam状态；最佳父模型沿用未改变的1304-state完整CPU恢复见证，无模型/优化器重放。

Full256严格GT覆盖参考8163、训练后两臂8130，是诊断上界而非部署成绩。跨阶段非选中参考存在444个坐标元素/36条表达的漂移，validity有12位/8条表达变化；所选共同字段无漂移。故不能将全部跨阶段Full256差值单独归因于训练。当前实现及配方未产生增量，不计为有效模块，不继续无界延长同一实验。

依用户长期授权，闭合审查和归档核验后，于{archive['time_cst']}实际删除两份非最佳终点权重（{retention['released_bytes']}字节）及2378份远端已归档NPZ（{int(archive['released_file_bytes'])}字节），合计341219157字节；本地NPZ归档保留。最佳5598/4848权重、必要官方PV/原G/V99依赖、数据集、文本结果和日志保留。数据盘实际空闲由{archive['free_bytes_before']}变为{archive['free_bytes_after']}字节；文件字节数与文件系统变化分别报告，不重复请求这类清理审批。

本轮实际运行约5小时21分，超过最初3小时45分估计；首次观察按既定15:20:55之后进行，后续间隔240秒，无提前查询或重复observer。后续同形任务以实际阶段耗时估计，在预计结束前几分钟首次检查，必要时180–300秒间隔。当前没有活动NN、observer或新训练。

完整目标仍未完成：同一个完整模型必须超过59.1%/50.1%（至少5620/4764），当前最佳5598/4848，@0.25尚缺22个命中；还需三个经直接对照证明有效的贡献。之后固定完整方法，再用各自作者预训练权重公平初始化、独立训练Sr3D/Nr3D；本轮没有新增跨基准结果。下一阶段先针对现有参考的真实支撑/边界错误取得证据，不将已失败的普通回写、质量排序或相同读取头重新命名为新模块。
"""
new = old + section.encode('utf-8')
assert new.startswith(old) and new.count(b'## 20.376.96 ') == 1
names = ['fit_wait.json', 'analysis/SUMMARY.json', 'analysis/EXPERIMENT_AUDIT.json',
         'analysis/EXPERIMENT_AUDIT.md', 'analysis/RESULT_TO_CLAIM.json', 'analysis/RESULT_TO_CLAIM.md',
         'analysis/TERMINAL_AUDITOR_CHECKS.json', 'analysis/terminal_auditor_checks.py',
         'closed_weight_inspection.json', 'weight_retention.json', 'CLEANUP_POLICY.json',
         'postrun/analyze_face_support_formal.py', 'postrun/ANALYSIS_SOURCE_REVIEW.json',
         'postrun/ANALYSIS_SOURCE_REVIEW.md', 'postrun/inspect_closed_face_weights.py',
         'postrun/inspect_closed_face_weights_authorized.py', 'postrun/cleanup_closed_face_pair.py',
         'postrun/cleanup_closed_face_pair_authorized.py', 'postrun/CLOSED_CONSUMERS_SOURCE_REVIEW.json',
         'postrun/CLOSED_CONSUMERS_SOURCE_REVIEW.md', 'publish_terminal_authorized.py',
         'PUBLISH_TERMINAL_SOURCE_REVIEW.json', 'PUBLISH_TERMINAL_SOURCE_REVIEW.md']
payloads = {prefix + name: (root / name).read_bytes() for name in names}
compact = {key: value for key, value in archive.items() if key != 'deleted_files'}
compact['full_receipt_sha256'] = hashlib.sha256(archive_path.read_bytes()).hexdigest()
payloads[prefix + 'ARCHIVED_ARRAY_CLEANUP_RESULT.json'] = (json.dumps(compact, indent=2) + '\n').encode()
payloads[prefix + '.gitattributes'] = b'** -text\n'
assert all(not name.endswith(('.pt', '.pth', '.npz')) and '.aris' not in name for name in payloads)
client = paramiko.SSHClient(); client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
spec = json.loads((root / 'pair_spec.json').read_bytes())
code = '''import base64,hashlib,json,sys
from pathlib import Path
project=Path(sys.argv[1]);b=json.load(sys.stdin);doc=project/b['doc'];old=doc.read_bytes()
assert hashlib.sha256(old).hexdigest()==b['old_sha256']
evidence=(project/b['prefix']).resolve()
assert str(evidence)=='/root/autodl-tmp/mcln_published_evidence_20261005/pvground_face_support_20261007/completed_comparison'
assert not evidence.exists()
for name,encoded in b['files'].items():
    assert name.startswith(b['prefix']) and not name.endswith(('.pt','.pth','.npz')) and '.aris' not in name
    p=project/name;assert evidence in p.resolve().parents;p.parent.mkdir(parents=True,exist_ok=True)
    raw=base64.b64decode(encoded)
    with p.open('xb') as f:f.write(raw)
    assert p.read_bytes()==raw
new=base64.b64decode(b['new_doc']);assert new.startswith(old);doc.write_bytes(new)
assert doc.read_bytes()==new
print(json.dumps(dict(files=len(b['files']),handoff_sha256=hashlib.sha256(new).hexdigest())))
'''
bundle = dict(doc=doc, prefix=prefix, old_sha256=previous['handoff_sha256'], new_doc=base64.b64encode(new).decode(),
              files={name: base64.b64encode(raw).decode() for name, raw in payloads.items()})
stdin, stdout, stderr = client.exec_command(shlex.join([spec['runtime'] + '/venv/bin/python', '-B', '-c', code,
    '/home/gb/new butd/butd_detr-main/MCLN-main']), timeout=180)
stdin.write(json.dumps(bundle).encode()); stdin.flush(); stdin.channel.shutdown_write()
raw = stdout.read(); assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
remote = json.loads(raw); client.close()
for name, raw in payloads.items():
    for repo in repos[:2]:
        path = repo / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
for path in copies:
    path.write_bytes(new)
heads = []
for index, repo in enumerate(repos):
    stage = [doc]
    if index < 2:
        with (repo / 'MANIFEST.md').open('a', encoding='utf-8') as stream:
            stream.write('\n- ' + stamp + ' Closed paired face-support ablation: both5593/4832, protected5598/4848 retained; actual nonbest weights and archived arrays cleaned.\n')
        stage += ['MANIFEST.md', *payloads]
    changed = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage], stderr=subprocess.DEVNULL)
    subprocess.check_call(['git', '-C', str(repo), '-c', 'core.whitespace=cr-at-eol,-blank-at-eof', 'diff', '--cached', '--check', '--', *stage])
    if index < 2:
        assert all(subprocess.check_output(['git', '-C', str(repo), 'show', ':' + name]) == raw for name, raw in payloads.items())
    assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + doc]).startswith(
        subprocess.check_output(['git', '-C', str(repo), 'show', previous['heads'][index] + ':' + doc]))
    subprocess.check_call(['git', '-C', str(repo), 'commit', '--quiet', '-m', 'Record closed face support comparison and authorized cleanup'])
    heads.append(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip())
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
subprocess.check_call(['git', '-C', str(repos[0]), '-c', 'http.version=HTTP/1.1', 'push', 'origin', 'HEAD:main'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == heads[0]
digest = hashlib.sha256(new).hexdigest()
assert digest == remote['handoff_sha256'] and all(path.read_bytes() == new for path in copies)
guard = workspace / '.codex/tmp/sync_cs_handoff_remote_20260923.py'; guard_raw = guard.read_bytes()
assert guard_raw.count(previous['handoff_sha256'].encode()) == 1
guard.write_bytes(guard_raw.replace(previous['handoff_sha256'].encode(), digest.encode()))
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), section='20.376.96', heads=heads,
    handoff_sha256=digest, four_local_and_remote_equal=True, github_main=heads[0], payload_count=len(payloads),
    execution_scope='CLOSED_FORMAL_COMPARISON_AND_ACTUAL_CLEANUP', raw_npz_published=False, weights_published=False,
    retained_best_hits=[5598, 4848], formal_hits_per_arm=[5593, 4832], weights_deleted=2, arrays_deleted=2378,
    released_file_bytes=341219157, new_training_started=False, overall_goal_complete=False)
receipt_path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
state_path = root.parent / 'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state = json.loads(state_path.read_bytes())
state.update(time_cst=record['time_cst'], latest_publication=str(receipt_path), handoff_section=record['section'],
    handoff_sha256=digest, published_heads=heads, status='FACE_SUPPORT_CLOSED_NEGATIVE_AND_CLEANED',
    owned_gpu_job_active=False, overall_goal_complete=False, face_support_terminal_publication_complete=True,
    face_support_terminal_publication=str(receipt_path), face_support_terminal_audit_pending=False,
    face_support_terminal_audit_verdict='WARN', face_support_result_to_claim='no',
    face_support_inspection_complete=True, face_support_weight_retention_complete=True,
    face_support_array_cleanup_complete=True, full_goal_status='ACTIVE_UNMET',
    next_action='Preserve5598/4848; do not rerun this negative pair. Obtain targeted actual support/boundary evidence before the next method design; same complete model5620/4764 and3effective contributions precede Sr/Nr.')
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record), flush=True)
