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
historical_faces = json.loads((root / 'HISTORICAL_FACE_SUMMARY.json').read_bytes())
independent = json.loads((root / 'analysis/INDEPENDENT_CPU_AUDIT.json').read_bytes())
audit = json.loads((root / 'analysis/ACTUAL_EVIDENCE_REVIEW.json').read_bytes())
wait = json.loads((root / 'wait.json').read_bytes())
archive_path = root / 'array_cleanup_receipt.json'
archive = json.loads(archive_path.read_bytes())
policy = json.loads((root.parent / 'pvground_face_support_20261007/CLEANUP_POLICY.json').read_bytes())
assert audit['fresh_context'] and audit['execution_scope'] == 'ACTUAL_ARTIFACTS'
assert audit['verdict'] in ('PASS', 'WARN') and not audit['blocking_findings']
assert summary['cases'] == 191 and summary['optimizer_updates'] == summary['weights_created'] == 0
assert not summary['new_overall_accuracy_claim'] and not summary['effective_module_claim']
assert summary['retained_best_hits'] == [5598, 4848]
assert historical_faces['cases'] == 191 and historical_faces['neural_forwards'] == 0
assert not historical_faces['current_predicted_masks_used'] and not historical_faces['old_query_or_foreground_recovered']
assert not historical_faces['new_overall_accuracy_claim'] and not historical_faces['effective_module_claim']
assert archive['status'] == 'STANDING_AUTHORIZED_CLOSED_DIAGNOSTIC_ARRAYS_REMOVED'
assert archive['deleted_count'] == 191 and archive['local_archive_preserved']
assert archive['best_weights_touched'] == archive['datasets_touched'] == archive['text_logs_touched'] == 0
assert policy['future_repeated_approval_required'] is False
assert wait['observer_closed'] and not wait['terminal']['controller_alive'] and wait['terminal']['exitcode'] == 0
previous = json.loads((root.parent / 'pvground_face_support_20261007/terminal_publication.json').read_bytes())
assert previous['section'] == '20.376.96'
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
prefix = 'refine-logs/pvground_support_boundary_cases_20261007/completed_diagnosis/'
over = summary['groups']['overextended/deployed']
missing = summary['groups']['missing_gt_extent/deployed']
old_over = historical_faces['groups']['overextended']
old_missing = historical_faces['groups']['missing_gt_extent']
section = f"""

## 20.376.97 已选参考错误的真实成员与六面支撑诊断、归档和自动清理（{stamp}）

本轮仅复查历史缓存中191条原框@0.25正确而Mask参考错误的开发表达，描述性分组为128条过度扩展、63条GT范围覆盖不足。它们不是完整验证集发生率、新训练划分或推理资格。保持原B8上下文，共161次真实模型前向、1288条上下文输入；没有优化器更新、新权重、重排或删除低分候选，全部256候选保留。GT只在模型前向后用于离线成员标注与几何诊断。controller于{wait['terminal']['status']['finished_cst']}退出0，实际耗时{wait['terminal']['status']['elapsed_seconds']:.3f}秒。

归档保存真实50000点XYZ、超点成员ID、目标成员标签、原生Text/Query/融合Mask及有效前景、实际参考与粗框、256原生分数。保留当前部署Query，以及本次前向中与历史索引相同的数字槽位；本次有{summary['current_vs_historical_selected_query_changes']}条选择变化。真实导入的GumbelSampling即使eval也无条件调用F.gumbel_softmax；本轮只执行161个选中批、开始前一次reset_rng，没有推进旧完整9508序列中被跳过批的随机状态。因此所谓historical角色仅是本次新前向的历史数字索引，不是已恢复的历史Query预测、支撑或物理实例对应。保留B8上下文和点身份不能消除这个限制。128/63分组只是历史错误表达队列，本轮不能建立它们的历史逐Query因果解释，也不据此归因原5598/4848的具体失败。CPU在全部191份新切片上重算六面成员极值、超点目标纯度、向外误差和框IoU，并核对保存的逐行证据。全XYZRGB输入身份由实际收集阶段核对；本地XYZ切片不构成全输入独立再生见证。

独立CPU检查确认：历史数字槽的粗框在{independent['archived_historical_coarse_box_changed_rows']}/191条均有变化，参考框在{independent['archived_historical_reference_box_changed_rows']}/191条变化；所选编号未变的{independent['archived_unchanged_selected_query_rows']}条中，{independent['archived_unchanged_selected_query_but_coarse_changed']}条粗框也变化。因此不以编号相同宣称预测恢复。该检查实际执行9550000个成员GT标签、382个角色切片和2286个六面记录；不调用模型或读取权重。

当前部署角色，128条过扩展组中：{over['any_pure_background_extreme_rows']}条至少一面极值来自纯背景超点，{over['any_mixed_extreme_rows']}条至少一面含混合超点，{over['any_background_extremal_point_rows']}条有背景成员极值；过扩展面{over['excessive_faces']}个，其中背景成员极值的过扩展面{over['excessive_background_extreme_faces']}个。63条范围不足组对应计数为{missing['any_pure_background_extreme_rows']}、{missing['any_mixed_extreme_rows']}、{missing['any_background_extremal_point_rows']}条，缺失面{missing['missing_faces']}个。纯背景、混合及背景成员统计可能重叠，不能相加；混合超点不等于其全部成员错误。无效参考明确记录，无新增推理fallback。目标实际成员范围与标注框范围分别保存，不把二者视为恒等。

另外完成不依赖新前向Mask的旧框坐标复核：使用历史缓存参考框的六面坐标，在相同真实输入点中找出轴向坐标相容的全部成员，容差2e-6米。只有全部可能极值成员均为背景时才判背景确定；只有全部可能来源超点均为纯背景时才判纯背景来源确定，目标与背景坐标重合保留为含糊。旧128条过扩展案例中，{old_over['only_background_extreme_rows']}条至少一面仅有背景成员可能，{old_over['all_possible_sources_pure_background_rows']}条至少一面所有可能来源均为纯背景超点，共{old_over['only_background_extreme_faces']}个背景确定面；{old_over['any_possible_mixed_source_rows']}条存在可能混合来源，但不证明旧Mask选中了该混合超点。63条缺范围案例中，{old_missing['gt_missing_faces']}个缺失GT面全部有已观测目标成员在其外侧。该分析未恢复旧Query或前景集合，也未使用新预测Mask；它只证明旧面坐标的保守成员来源和已观测目标超出范围，不给出新推理规则或涨点承诺。两次历史数字槽粗框复现断言实际失败，最大差2.7123厘米，红色检查保留，不作为已修复问题或模型精度失败。

独立fresh审查为{audit['verdict']}，blocking findings=0；same-family/provisional，实际backend/effort身份未认证。本诊断没有新的9508条总体成绩，没有验证有效的新模块，也不把191个GT筛选错误换算成潜在涨点。当前保留最佳仍5598/4848（58.8767%/50.9886%）。

原单文件串行SFTP传输实测22份NPZ耗时约14分钟，故仅停止该传输进程并改为读取预取，已有完整文件按原字节/SHA核验后复用；原模型诊断不重跑。续传最终全部191份数组归档成功。依用户“清理，以后不用我审批无用的权重这些”的长期授权，于{archive['time_cst']}实际删除仅这191份远端临时NPZ，释放{archive['released_file_bytes']}文件字节；数据盘实际空闲由{archive['free_bytes_before']}变为{archive['free_bytes_after']}字节。本地完整数组、最佳权重、必要PV/G/V99依赖、数据集、文本日志和结果保留；本轮生成及删除权重均0。

完整目标仍为同一完整ScanRefer模型两项超过59.1%/50.1%（至少5620/4764），并有三个经直接控制证明有效的贡献。当前@0.25仍差22个命中，三项贡献尚未成立；之后固定完整方法，使用各自作者预训练权重公平初始化，独立训练Sr3D/Nr3D。固定seed2027，不做多seed。下一项修改需依据真实支撑缺失或背景极值证据确定最小控制，不恢复双源排名，不无条件丢弃错误或低分候选。
"""
new = old + section.encode('utf-8')
assert new.startswith(old) and new.count(b'## 20.376.97 ') == 1
names = ['EXPERIMENT_PLAN.md', 'prepare_inputs.py', 'diagnostic_spec.json', 'case_manifest.json',
    'support_evidence.py', 'collect_support_cases.py', 'controller.py', 'launch.json',
    'SOURCE_REVIEW.json', 'SOURCE_REVIEW.md', 'ANALYSIS_SOURCE_REVIEW.json', 'ANALYSIS_SOURCE_REVIEW.md',
    'CLEANUP_SOURCE_REVIEW.json', 'CLEANUP_SOURCE_REVIEW.md', 'wait.json',
    'TRANSFER_SLOW_DIAGNOSIS.json', 'transfer_stop.json', 'resume_closed_collection_authorized.py',
    'TRANSFER_RESUME_SOURCE_REVIEW.json', 'TRANSFER_RESUME_SOURCE_REVIEW.md',
    'complete/receipt.json', 'complete/imports.json', 'complete/load.json',
    'analyze_closed.py', 'analysis/SUMMARY.json', 'analysis/ACTUAL_EVIDENCE_REVIEW.json',
    'analysis/ACTUAL_EVIDENCE_REVIEW.md', 'cleanup_archived_authorized.py',
    'PRELIMINARY_ACTUAL_EVIDENCE_REVIEW.json', 'PRELIMINARY_ACTUAL_EVIDENCE_REVIEW.md',
    'PRELIMINARY_CPU_CHECK.json', 'audit_actual_support_independent.py',
    'historical_face_provenance.py', 'analyze_historical_faces.py', 'check_historical_face_provenance.py',
    'HISTORICAL_FACE_LOCAL_CHECK.json', 'HISTORICAL_FACE_SOURCE_REVIEW.json', 'HISTORICAL_FACE_SOURCE_REVIEW.md',
    'HISTORICAL_FACE_SUMMARY.json', 'HISTORICAL_FACE_ROWS.jsonl', 'HISTORICAL_FACE_ONE_CASE.json',
    'HISTORICAL_RECOVERY_RED_CHECK.json', 'check_historical_slot_recovery.py',
    'analysis/INDEPENDENT_CPU_AUDIT.json', 'analysis/INDEPENDENT_HISTORICAL_FACE_AUDIT.json',
    'audit_historical_coordinates_independent.py',
    'publish_terminal_authorized.py', 'PUBLISH_TERMINAL_SOURCE_REVIEW.json', 'PUBLISH_TERMINAL_SOURCE_REVIEW.md']
payloads = {prefix + name: (root / name).read_bytes() for name in names}
compact = {key: value for key, value in archive.items() if key != 'deleted_files'}
compact['full_receipt_sha256'] = hashlib.sha256(archive_path.read_bytes()).hexdigest()
payloads[prefix + 'ARCHIVED_ARRAY_CLEANUP_RESULT.json'] = (json.dumps(compact, indent=2) + '\n').encode()
payloads[prefix + '.gitattributes'] = b'** -text\n'
assert all(not name.endswith(('.pt', '.pth', '.npz')) and '.aris' not in name for name in payloads)
client = paramiko.SSHClient(); client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
spec = json.loads((root / 'diagnostic_spec.json').read_bytes())
code = '''import base64,hashlib,json,sys
from pathlib import Path
project=Path(sys.argv[1]);b=json.load(sys.stdin);doc=project/b['doc'];old=doc.read_bytes()
assert hashlib.sha256(old).hexdigest()==b['old_sha256']
evidence=(project/b['prefix']).resolve()
assert str(evidence)=='/root/autodl-tmp/mcln_published_evidence_20261005/pvground_support_boundary_cases_20261007/completed_diagnosis'
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
            stream.write('\n- ' + stamp + ' Closed191-case support/boundary diagnosis; protected5598/4848 retained; verified archived temporary arrays cleaned.\n')
        stage += ['MANIFEST.md', *payloads]
    changed = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage], stderr=subprocess.DEVNULL)
    subprocess.check_call(['git', '-C', str(repo), '-c', 'core.whitespace=cr-at-eol,-blank-at-eof', 'diff', '--cached', '--check', '--', *stage])
    if index < 2:
        assert all(subprocess.check_output(['git', '-C', str(repo), 'show', ':' + name]) == raw for name, raw in payloads.items())
    assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + doc]).startswith(
        subprocess.check_output(['git', '-C', str(repo), 'show', previous['heads'][index] + ':' + doc]))
    subprocess.check_call(['git', '-C', str(repo), 'commit', '--quiet', '-m', 'Record actual support boundary diagnosis and authorized cleanup'])
    heads.append(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip())
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
subprocess.check_call(['git', '-C', str(repos[0]), '-c', 'http.version=HTTP/1.1', 'push', 'origin', 'HEAD:main'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == heads[0]
digest = hashlib.sha256(new).hexdigest()
assert digest == remote['handoff_sha256'] and all(path.read_bytes() == new for path in copies)
guard = workspace / '.codex/tmp/sync_cs_handoff_remote_20260923.py'; guard_raw = guard.read_bytes()
assert guard_raw.count(previous['handoff_sha256'].encode()) == 1
guard.write_bytes(guard_raw.replace(previous['handoff_sha256'].encode(), digest.encode()))
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), section='20.376.97', heads=heads,
    handoff_sha256=digest, four_local_and_remote_equal=True, github_main=heads[0], payload_count=len(payloads),
    execution_scope='ACTUAL_TARGETED_SUPPORT_DIAGNOSIS_AND_CLEANUP', raw_npz_published=False, weights_published=False,
    retained_best_hits=[5598,4848], diagnostic_cases=191, optimizer_updates=0, weights_created=0,
    arrays_deleted=191, released_file_bytes=archive['released_file_bytes'], overall_goal_complete=False)
receipt_path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
state_path = root.parent / 'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state = json.loads(state_path.read_bytes())
state.update(time_cst=record['time_cst'], latest_publication=str(receipt_path), handoff_section=record['section'],
    handoff_sha256=digest, published_heads=heads, status='SUPPORT_BOUNDARY_DIAGNOSIS_CLOSED_AND_CLEANED',
    owned_gpu_job_active=False, overall_goal_complete=False, full_goal_status='ACTIVE_UNMET',
    support_boundary_collection_complete=True, support_boundary_analysis_complete=True,
    historical_face_coordinate_analysis_complete=True,
    publisher_needs_scientific_limit_revision=False,
    support_boundary_array_cleanup_complete=True, support_boundary_terminal_publication_complete=True,
    next_action='Preserve5598/4848 and complete support evidence. Use actual191 member/face findings to define the smallest direct intervention; Scan5620/4764 and3effective contributions precede author-init Sr/Nr.')
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record), flush=True)
