"""Publish the completed hidden control; leave the visible arm and observer intact."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import paramiko

local = Path(__file__).resolve().parent
formal = local / 'formal_draft'
face = local.parent / 'pvground_face_conditioned_20261004'
receipt_path = local / 'hidden_control_publication_20261005.json'
assert not receipt_path.exists()
previous_path = local / 'terminal_analysis_preparation_publication.json'
previous = json.loads(previous_path.read_bytes())
observation_path = formal / 'observation_08.json'
observation = json.loads(observation_path.read_bytes())
assert observation['status']['stage'] == 'evidence_visible/train'
assert observation['controller_alive'] and observation['exitcode'] is None
completed = observation['status']['completed']
assert [(item['arm'], item['mode']) for item in completed] == [('evidence_hidden', 'train'), ('evidence_hidden', 'formal')]
retention_observed = observation['status']['last_weight_retention']
assert retention_observed['cpu_bbs_hits25'] == 5616 and retention_observed['cpu_bbs_hits50'] == 4475
assert retention_observed['cpu_box_threshold_changes'] == 0
workspace = Path('C:/Users/gb')
repos = [workspace / '.codex_mcln_g0_20260905', workspace / '.codex_pvground_cs_20261002', workspace / '.codex_mcln_v99_internal_20260928']
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies = [repo / doc for repo in repos] + [workspace / 'Desktop/document' / Path(doc).name]
old = copies[0].read_bytes()
assert hashlib.sha256(old).hexdigest() == previous['handoff_sha256']
assert all(path.read_bytes() == old for path in copies)
assert b'## 20.376.56 ' not in old
for repo, head in zip(repos, previous['heads']):
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip() == head
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
guard = workspace / '.codex/tmp/sync_cs_handoff_remote_20260923.py'
guard_raw = guard.read_bytes()
assert guard_raw.count(previous['handoff_sha256'].encode()) == 1
output = local / 'hidden_complete'
assert not output.exists()
output.mkdir()
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
remote_project = '/home/gb/new butd/butd_detr-main/MCLN-main'
remote_arm = '/root/autodl-tmp/pvground_readback_fit_20261004/evidence_hidden'
with sftp.open(remote_project + '/' + doc, 'rb') as stream:
    assert stream.read() == old
names = {'fit_receipt.json': 'receipt.json', 'formal_receipt.json': 'formal/receipt.json',
    'formal_restore.json': 'formal_restore.json', 'weight_retention.json': 'weight_retention.json'}
raws = {}
for name, relative in names.items():
    with sftp.open(remote_arm + '/' + relative, 'rb') as stream:
        raw = stream.read()
    (output / name).write_bytes(raw)
    raws[name] = raw
fit = json.loads(raws['fit_receipt.json'])
evaluation = json.loads(raws['formal_receipt.json'])
restore = json.loads(raws['formal_restore.json'])
retention = json.loads(raws['weight_retention.json'])
assert retention == retention_observed
assert fit['status'] == 'complete' and fit['training_steps'] == 3723 and fit['fit_rows'] == 29778
assert fit['fit_seen_exactly_once'] and fit['frozen_parent_states_exact'] and not fit['use_geometry_evidence']
assert fit['readback_parameters'] == 96672 and fit['readback_state_tensors'] == 23
assert fit['physical_batch'] == fit['effective_batch'] == 8 and fit['last_batch_rows'] == 2
assert evaluation['status'] == 'pass' and evaluation['rows'] == evaluation['formal_rows'] == 9508
assert evaluation['primary_mode'] == 'bbs' and evaluation['primary_threshold'] == .5
metric = evaluation['metrics']['bbs']
assert metric['rec_hits25'] == 5616 and metric['rec_hits50'] == 4475
assert restore['status'] == 'pass' and restore['strict_model_restore'] and restore['restored_steps'] == 3723
assert fit['terminal_sha256'] == restore['terminal_sha256'] == retention['terminal_sha256']
assert evaluation['rows_sha256'] == retention['formal_rows_sha256']
assert retention['required_parent_chain_preserved'] and not retention['local_weight_archive_created']
assert len(retention['deleted']) == 1 and retention['deleted'][0]['bytes'] == 1284112
remaining = sftp.listdir(remote_arm)
assert 'terminal.pth' not in remaining and 'latest.pth' not in remaining
direct = evaluation['fixed_frame_readback_effect']
stamp = datetime.datetime.now().astimezone().isoformat()
summary = dict(time_cst=stamp, status='HIDDEN_CONTROL_COMPLETE_VISIBLE_ARM_RUNNING',
    formal_rows=9508, primary_mode='bbs', primary_threshold=.5, metrics=metric,
    acc25_percent=metric['rec_hits25'] / 9508 * 100, acc50_percent=metric['rec_hits50'] / 9508 * 100,
    delta_vs_frozen4506=dict(hits25=0, hits50=-31), fixed_frame_readback_effect=direct,
    training_steps=3723, fit_rows=29778, effective_batch=8, last_batch_rows=2,
    completed_stages=completed, observation=str(observation_path), retention=retention,
    actual_receipt_sha256={name: hashlib.sha256(raw).hexdigest() for name, raw in raws.items()},
    independent_terminal_audit_complete=False, comparison_pair_complete=False,
    visible_arm_accuracy_available=False, weights_downloaded=0, new_inference_or_optimizer_updates=0,
    active_training_source_changed=False, observer_session=66222, goal_achieved=False)
summary_raw = (json.dumps(summary, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
(output / 'SUMMARY.json').write_bytes(summary_raw)
prefix = 'refine-logs/pvground_geometry_readback_20261004/'
payloads = {prefix + 'formal_draft/observation_08.json': observation_path.read_bytes(),
    prefix + 'hidden_complete/SUMMARY.json': summary_raw,
    prefix + Path(__file__).name: Path(__file__).read_bytes()}
payloads.update({prefix + 'hidden_complete/' + name: raw for name, raw in raws.items()})
section = f'''

## 20.376.56 回读的隐藏几何证据控制已完成；可见证据组继续运行（{stamp}）

结构、共同起点与固定训练协议沿用§20.376.54。本节仅追加第一组的真实完成结果；两组对照尚未结束，不能据此宣布新增几何回读有效或无效。

`evidence_hidden`为相同容量的回读控制：保留原Query、完整文本和六面角色，只将新增44维几何证据置零。96672个R参数／23项状态从相同初始化训练，官方PV、原G和4506几何父模型保持冻结／eval。实际29778条fit各一次，batch／有效batch8、尾批2，完成3723次更新。训练及6887条模块留出阶段于{completed[0]['finished_cst']}结束，耗时{completed[0]['seconds']:.3f}秒；该6887条不是正式验证。

独立进程的完整9508条正式验证于{completed[1]['finished_cst']}结束，耗时{completed[1]['seconds']:.3f}秒。原生`last/bbs`为**59.0660%／47.0656%〔5616／4475〕**，相对冻结4506父模型为**0／−31命中**。当前受保护最佳仍为5616／4506（59.0660%／47.3917%）；50%开发线仍差248个严格命中。原生Mask为{metric['mask_hits25'] / 9508 * 100:.4f}%／{metric['mask_hits50'] / 9508 * 100:.4f}%／{metric['mask_miou']:.4f}% mIoU，均来自本组同一原生Query，未拼接其他模型。

同一次前向、固定候选框下的缓存原语义头回放：@0.25修复{direct['0.25']['fixes']}／破坏{direct['0.25']['damages']}（净{direct['0.25']['net']:+d}）；@0.50修复{direct['0.5']['fixes']}／破坏{direct['0.5']['damages']}（净{direct['0.5']['net']:+d}）。这只是本检查点R直接选择作用的诊断，不是独立训练消融。独立CPU从实际9508行框坐标重算得5616／4475，两个阈值判断差异为0；最终跨组行级分析与新鲜完整性审查仍待两组闭合后执行。

按用户持续授权，在实际模型／优化器恢复及CPU计数核验后，已删除本组非最佳`evidence_hidden/terminal.pth`，1284112字节，SHA256 `{retention['terminal_sha256']}`；本次取回收据时再次确认远端该组无terminal／latest权重。本地未归档权重，4506最佳及全部必需重建父链保留。`evidence_visible`已于{observation['status']['stage_started_cst']}启动，尚无其完整指标；不改变活动训练源码、不重启控制器、继续使用原观察器66222每240秒检查。预计第二组整段结束约03:43 CST，仅按第一组实测耗时估计，不作为完成证据。

实际证据位于`{prefix}hidden_complete/`及`formal_draft/observation_08.json`。本次仅读取已完成第一组的四份收据并发布，没有执行新前向或优化器更新，没有下载权重。目标仍未达成，Nr3D／Sr3D新结构训练及三个有效贡献也尚未成立。
'''
new = old + section.encode('utf-8')
assert new.startswith(old) and new.count(b'## 20.376.56 ') == 1
sftp.mkdir(remote_project + '/' + prefix + 'hidden_complete')
for repo in repos[:2]:
    (repo / prefix / 'hidden_complete').mkdir()
for relative, raw in payloads.items():
    for repo in repos[:2]:
        (repo / relative).write_bytes(raw)
    with sftp.open(remote_project + '/' + relative, 'wb') as stream:
        stream.write(raw)
    with sftp.open(remote_project + '/' + relative, 'rb') as stream:
        assert stream.read() == raw
for path in copies:
    path.write_bytes(new)
with sftp.open(remote_project + '/' + doc, 'wb') as stream:
    stream.write(new)
with sftp.open(remote_project + '/' + doc, 'rb') as stream:
    assert stream.read() == new
sftp.close()
client.close()
heads = []
for index, repo in enumerate(repos):
    stage = [doc]
    if index < 2:
        with (repo / 'MANIFEST.md').open('a', encoding='utf-8') as stream:
            stream.write('\n- ' + stamp + ' actual hidden readback control9508 native5616/4475; CPU threshold differences0, inferior1284112-byte delta removed, visible arm running; no full pair claim.\n')
        stage += ['MANIFEST.md', *payloads]
    changed = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage])
    subprocess.check_call(['git', '-C', str(repo), 'diff', '--cached', '--check', '--', *stage])
    for relative, raw in (payloads.items() if index < 2 else []):
        assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + relative]) == raw
    subprocess.check_call(['git', '-C', str(repo), 'commit', '--quiet', '-m', 'Record completed hidden readback control and verified nonbest retention'])
    heads.append(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip())
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
subprocess.check_call(['git', '-C', str(repos[0]), '-c', 'http.version=HTTP/1.1', 'push', 'origin', 'HEAD:main'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == heads[0]
digest = hashlib.sha256(new).hexdigest()
assert all(path.read_bytes() == new for path in copies)
guard.write_bytes(guard_raw.replace(previous['handoff_sha256'].encode(), digest.encode()))
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), section='20.376.56',
    predecessor=str(previous_path), heads=heads, github_main=heads[0], handoff_sha256=digest,
    handoff_bytes=len(new), four_local_and_remote_equal=True, exact_committed_payloads=True,
    payload_count=len(payloads), hidden_control_formal_hits=[5616, 4475], comparison_pair_complete=False,
    visible_arm_accuracy_available=False, weights_downloaded=0, inference_or_optimizer_replayed=False,
    observer_session=66222, goal_achieved=False)
receipt_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
state_path = face / 'active_continuation_state.json'
state = json.loads(state_path.read_bytes())
state.update(time_cst=record['time_cst'], status='READBACK_HIDDEN_COMPLETE_VISIBLE_ARM_RUNNING',
    latest_publication=str(receipt_path), published_heads=heads, handoff_sha256=digest,
    handoff_section='20.376.56', handoff_bytes=len(new), actual_formal_observation=str(observation_path),
    completed_stages=completed, observed_phase='evidence_visible/train', observed_rows=None,
    formal_result_available=True, formal_result_arm='evidence_hidden', formal_hits25=5616, formal_hits50=4475,
    formal_pair_complete=False, formal_complete=False, visible_arm_accuracy_available=False,
    hidden_complete_summary=str(output / 'SUMMARY.json'), last_weight_retention=retention,
    last_goal_turn_classification='PROGRESS_COMPLETED_HIDDEN_CONTROL_PUBLISHED',
    directory_free_bytes_at_observation=observation['directory_free_bytes'],
    system_free_bytes_at_observation=observation['system_free_bytes'],
    controller_alive_observation_cst=observation['time_cst'])
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
for path in [face / 'NEXT_CONTINUATION.md', local / 'NEXT_CONTINUATION.md']:
    body = path.read_text(encoding='utf-8')
    start = body.index('Latest actual publication:')
    end = body.index('\n\nProtected best', start)
    body = body[:start] + ('Latest actual publication: ' + str(receipt_path) + '\nMain ' + heads[0] + '; section20.376.56, ' + str(len(new)) + 'bytes; SHA256 ' + digest + '. Fourlocal+remote exact, actual hidden control receipts published. Never rerun completed publishers.') + body[end:]
    lines = body.splitlines()
    cursor_indexes = [i for i, line in enumerate(lines) if line.startswith('ONLY native observer')]
    assert len(cursor_indexes) == 1
    lines[cursor_indexes[0]] = 'ONLY native observer66222 is live. Actual observation01:30:58 CST confirms hidden formal complete01:29:27, native5616/4475, CPU thresholddifferences0 and negative1284112B delta already removed. Visible arm process526600 started01:29:31 CST; no visible accuracy. Controller alive at observation, later240s; estimated full pair end~03:43 CST, not observed. Required4506/G/officialPV parents preserved. Actual witness: ' + str(observation_path)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
assert (face / 'NEXT_CONTINUATION.md').read_bytes() == (local / 'NEXT_CONTINUATION.md').read_bytes()
with (workspace / 'memory/2026-10-05.md').open('a', encoding='utf-8') as stream:
    stream.write('\nPVGround ' + record['time_cst'] + ': actual hidden control complete9508 native5616/4475 (strict-31 vs4506), fixedframe replay and masks recorded in hidden_complete actualreceipts, CPU thresholddiff0. Inferior1284112B delta already removed/no localweightarchive; visible started01:29:31, sole66222 remains. Publishedmain' + heads[0] + '/doc56/fourlocal+remoteexact; goalACTIVE_UNMET.\n')
print(json.dumps(record), flush=True)
