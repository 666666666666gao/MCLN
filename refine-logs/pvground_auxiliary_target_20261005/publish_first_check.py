"""Append the actual closed sanity and fit launch without claiming accuracy."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

import paramiko


root = Path(__file__).resolve().parent
state_path = root.parent / 'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state = json.loads(state_path.read_bytes())
previous = json.loads((root / 'retention_tools_publication.json').read_bytes())
assert previous['section'] == '20.376.72'
assert not (root / 'first_check_publication.json').exists()
for name in ('SOURCE_REVIEW.json', 'LAUNCH_REVIEW.json'):
    review = json.loads((root / name).read_bytes())
    assert review['verdict'] == 'PASS' and not review['blocking_findings']
    for item in review['reviewed_files']:
        assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
wait = json.loads((root / 'preflight_wait.json').read_bytes())
assert wait['observer_closed'] and not wait['controller_alive']
assert wait['exitcode'] == 0 and wait['status']['status'] == 'complete'
proofs = {arm: json.loads((root / 'preflight_complete' / arm / 'preflight.json').read_bytes())
          for arm in ('control', 'member_target')}
for proof in proofs.values():
    assert proof['status'] == 'pass' and proof['optimizer_steps'] == 2
    assert proof['weight_files_created'] == 0 and proof['native_data_and_member_target_exact']
    assert proof['all_parent_and_R_states_exact'] and proof['isolated_extra_geometry_gradient_verified']
    assert proof['optimizer_exact_check']['all_keys_moments_steps_and_groups_exact']
launch = json.loads((root / 'fit_launch.json').read_bytes())
observer = json.loads((root / 'FIT_OBSERVER_HANDLE.json').read_bytes())
assert launch['status'] == 'TWO_ARM_FIT_LAUNCHED_NOT_COMPLETED'
assert launch['protected_best_hits'] == [5614, 4509]
assert observer['native_session_id'] == 38531 and not observer['observer_closed']
assert (root / 'fit_observer_started.json').exists() and not (root / 'fit_wait.json').exists()

repos = [Path(r'C:\Users\gb\.codex_mcln_g0_20260905'),
         Path(r'C:\Users\gb\.codex_pvground_cs_20261002'),
         Path(r'C:\Users\gb\.codex_mcln_v99_internal_20260928')]
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies = [repo / doc for repo in repos] + [Path(r'C:\Users\gb\Desktop\document') / Path(doc).name]
old = copies[0].read_bytes()
assert all(path.read_bytes() == old for path in copies)
assert hashlib.sha256(old).hexdigest() == previous['handoff_sha256']
assert b'## 20.376.73 ' not in old


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args]).decode().strip()


for repo, head in zip(repos, previous['heads']):
    assert git(repo, 'rev-parse', 'HEAD') == head and not git(repo, 'status', '--porcelain')
stamp = datetime.datetime.now().astimezone().isoformat()
relocation = json.loads((root / 'PUBLICATION_STORAGE_RELOCATION.json').read_bytes())
assert relocation['status'] == 'PUBLISHED_EVIDENCE_RELOCATED_LOGICAL_PATH_PRESERVED'
assert relocation['source_and_destination_file_sizes_and_sha256_exact']
assert relocation['records_deleted'] == relocation['weights_deleted'] == 0
review = json.loads((root / 'RETENTION_REVIEW.json').read_bytes())
assert review['verdict']=='PASS' and not review['blocking_findings'] and review['execution_scope']=='SOURCE_ONLY'
assert all(hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'] for item in review['reviewed_files'])
assert json.loads((root / 'RETENTION_REVIEW_CALL.json').read_bytes())['result_received']
progress = json.loads((root / 'SCHEDULED_PROGRESS_185943.json').read_bytes())
estimate = json.loads((root / 'FIRST_CHECK_ESTIMATE.json').read_bytes())
assert progress['controller_alive'] and progress['protected_best_sha256_exact']
assert progress['progress']['last_logged_step']==3723 and not estimate['formal_complete']
assert hashlib.sha256((root / 'SCHEDULED_PROGRESS_185943.json').read_bytes()).hexdigest()==estimate['basis_progress_sha256']
section = f'''

## 20.376.73 首查确认control更新完成，终点留出仍评估中；结束时间按实测修订（{stamp}）

唯一观察器于18:54:14实际首查，controller630343及control子进程630346存活，状态为control/train，尚无completed_run。随后{progress['time_cst']}做一次与该窗口对应的只读进度／资源核对，没有模型或优化器重放，也没有启动第二观察器。

这次实际日志已完整写出3723条更新记录，最后step3723、尾batch2；训练累计4453.7007秒，最后512步平均1.1847秒。control的terminal.pth已生成5585925字节，原4509头5585861字节的SHA仍与规格一致。这里仅确认更新和保存发生，不将未评估新权重替换4509。

已完成的initial模块留出为6887条、6172／5609 REC命中，耗时996.1125秒；它是预训练见过场景的模块留出，不是9508正式验证。检查时terminal模块留出仅写出4286／6887条，没有terminal完整回执，也没有本轮formal完整成绩。当前正式最好仍5614／4509，不使用局部评估数字更新成绩。

该次资源快照为A100显存6979／40960MiB、利用率12%，系统盘余量467582976字节、数据盘1774354432字节。只描述检查时刻，不由该快照推断全训练峰值或服务器之后状态。

新估计使用实际4453.7007秒训练、996.1125秒initial评估、约450.13秒重建／初始准备，以及上一轮完整9508实测评估成本并按本轮速度缩放。control formal预计{estimate['control_formal_end_estimate_cst']}附近完成，两组预计{estimate['pair_end_estimate_cst']}附近闭合；下次人工结果核对安排{estimate['next_manual_outcome_check_cst']}附近。均为阶段估计，不是已完成结果；相较原21:44估计，按这次实测调整到约22:03。

既有观察器38531继续240秒间隔，不修改活动实验源码、预算或原启动记录。待control完整formal回执后再报告9508数字；两组都闭合、CPU重算及终态审查完成后才依§20.376.72严格主指标晋级／清理规则处理权重。仍保留全部256候选、唯一last/bbs和同Query框／Mask，目标ACTIVE_UNMET。原始首查、进度和时间估计证据见refine-logs/pvground_auxiliary_target_20261005/。
'''
new = old + section.encode('utf-8')
prefix = 'refine-logs/pvground_auxiliary_target_20261005/'
payloads = {prefix + file.relative_to(root).as_posix(): file.read_bytes()
            for file in root.rglob('*')
            if file.is_file() and file.suffix in ('.py', '.json', '.md', '.log', '.exit')}
payloads[prefix + '.gitattributes'] = b'** -text\n'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
project = '/home/gb/new butd/butd_detr-main/MCLN-main'
for name, raw in payloads.items():
    for repo in repos[:2]:
        destination = repo / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    parent = name.rpartition('/')[0]
    parts = parent.split('/')
    for index in range(1, len(parts) + 1):
        folder = '/'.join(parts[:index])
        upper, _, leaf = folder.rpartition('/')
        if leaf not in sftp.listdir(project + '/' + upper):
            sftp.mkdir(project + '/' + folder)
    with sftp.open(project + '/' + name, 'wb') as stream:
        stream.write(raw)
    with sftp.open(project + '/' + name, 'rb') as stream:
        assert stream.read() == raw
for path in copies:
    path.write_bytes(new)
with sftp.open(project + '/' + doc, 'wb') as stream:
    stream.write(new)
with sftp.open(project + '/' + doc, 'rb') as stream:
    assert stream.read() == new
sftp.close()
client.close()

heads = []
reports = [name for name in payloads if Path(name).name in
           ('SOURCE_REVIEW.json', 'SOURCE_REVIEW.md', 'LAUNCH_REVIEW.json', 'LAUNCH_REVIEW.md', 'RETENTION_REVIEW.json', 'RETENTION_REVIEW.md')]
for index, repo in enumerate(repos):
    stage = [doc]
    if index < 2:
        with (repo / 'MANIFEST.md').open('a', encoding='utf-8') as stream:
            stream.write('\n- ' + stamp + ' Actual scheduled check confirmed control3723updates complete, terminal holdout4286rows; no formal result. Revised pair estimate22:03.\n')
        stage.append('MANIFEST.md')
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage], stderr=subprocess.DEVNULL)
    if index < 2:
        subprocess.check_call(['git', '-C', str(repo), '-c', 'core.autocrlf=false',
                               'add', '-f', '--', *payloads], stderr=subprocess.DEVNULL)
        stage += list(payloads)
    excludes = [':(exclude)' + name for name in reports] if index < 2 else []
    subprocess.check_call(['git', '-C', str(repo), '-c', 'core.whitespace=cr-at-eol',
                           'diff', '--cached', '--check', '--', *stage, *excludes])
    subprocess.check_call(['git', '-C', str(repo), '-c', 'core.whitespace=cr-at-eol,-blank-at-eof',
                           'diff', '--cached', '--check', '--', *stage])
    if index < 2:
        for name, raw in payloads.items():
            assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + name]) == raw
    assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + doc]) == new.replace(b'\r\n', b'\n')
    subprocess.check_call(['git', '-C', str(repo), 'commit', '--quiet', '-m',
                           'Record actual first progress check and revise auxiliary fit estimate'])
    heads.append(git(repo, 'rev-parse', 'HEAD'))
    assert not git(repo, 'status', '--porcelain')
subprocess.check_call(['git', '-C', str(repos[0]), '-c', 'http.version=HTTP/1.1',
                       'push', 'origin', 'HEAD:main'])
assert git(repos[0], 'ls-remote', 'origin', 'refs/heads/main').split()[0] == heads[0]

digest = hashlib.sha256(new).hexdigest()
guard = Path(r'C:\Users\gb\.codex\tmp\sync_cs_handoff_remote_20260923.py')
raw = guard.read_bytes()
assert raw.count(previous['handoff_sha256'].encode()) == 1
guard.write_bytes(raw.replace(previous['handoff_sha256'].encode(), digest.encode()))
record = dict(previous)
record.update(time_cst=stamp, section='20.376.73', heads=heads, github_main=heads[0],
              handoff_bytes=len(new), handoff_sha256=digest, four_local_and_remote_equal=True,
              payload_count=len(payloads), status='ACTUAL_SCHEDULED_CONTROL_PROGRESS_AND_REVISED_ESTIMATE_PUBLISHED',
              new_accuracy_result=False, preflight_passed=True, formal_fit_launched=True,
              formal_fit_started_cst=launch['time_cst'], observer_native_session=observer['native_session_id'])
record.update(publication_storage_relocation=relocation,closed_result_tools_executed=False,retention_tools_executed=False)
record.update(actual_scheduled_progress=progress,revised_stage_estimate=estimate)
(root / 'first_check_publication.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
current = json.loads(state_path.read_bytes())
current.update(latest_publication=str(root / 'first_check_publication.json'), published_heads=heads,
               handoff_section='20.376.73', handoff_sha256=digest,
               current_goal_turn_classification='PROGRESS_ACTUAL_SCHEDULED_CHECK_AND_TIME_ESTIMATE')
state_path.write_text(json.dumps(current, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
with Path(r'C:\Users\gb\memory\2026-10-05.md').open('a', encoding='utf-8') as stream:
    stream.write('\nPVGround ' + stamp + ': Actual18:54 observer/18:59 read-only check: control updates3723 complete and terminal saved, terminal holdout4286/6887, no formal9508 result. Best4509SHA exact; system467.6MB/data1.774GB. Revised control formal19:36:54, pair22:02:51, manual next19:34:54, same observer38531/240s. Doc73 Main ' + heads[0] + ', ACTIVE_UNMET.\n')
print(json.dumps({key: record[key] for key in ('time_cst', 'section', 'heads', 'handoff_sha256',
      'four_local_and_remote_equal', 'formal_fit_started_cst', 'formal_fit_launched')}), flush=True)
