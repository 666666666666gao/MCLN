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
previous = json.loads((root / 'first_check_publication.json').read_bytes())
assert previous['section'] == '20.376.73'
assert not (root / 'control_closed_publication.json').exists()
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
assert b'## 20.376.74 ' not in old


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
intake=json.loads((root/'CONTROL_CLOSED_INTAKE.json').read_bytes())
recount=json.loads((root/'CONTROL_CPU_RECOUNT.json').read_bytes())
control_estimate=json.loads((root/'CONTROL_COMPLETE_ESTIMATE.json').read_bytes())
assert intake['formal_rows']==recount['formal_rows']==9508
assert recount['control']['cpu_box_threshold_changes']==0
assert [recount['control'][k] for k in ('rec_hits25','rec_hits50')]==[5616,4511]
assert not recount['full_pair_complete'] and not recount['best_weight_promoted']
section=f'''

## 20.376.74 native_gt控制组正式结束，CPU完整重算5616／4511；member_gt仍待结果（{stamp}）

control正式9508条评估实际于{intake['formal_finished_cst']}结束，train／formal退出码均0；19:34:27既有观察器已见control/train、control/formal均闭合，member_target/train子进程636595启动。收集8份已结束控制组文件，其中完整formal rows为7623290字节、SHA256=ec73b6acb9691d6c591136c5e5a34e80a8d7d716bedef714ca960bbbd324716c；不下载权重或重放模型。

该组从4509头出发再训练29778条各一次／3723更新，仅训练456102参数既有几何头；父模型及零R状态保持、完整模型与优化器恢复回执通过。初始6887模块留出为6172／5609，终点6177／5613；两者为预训练见过场景的模块留出，不与9508正式数字混用。

| 同一检查点原生last/bbs，9508条 | Acc@0.25 | Acc@0.50 |
|---|---:|---:|
| 保留的4509父模型 | 59.0450%〔5614〕 | 47.4232%〔4509〕 |
| 本轮native_gt继续训练控制 | 59.0660%〔5616〕 | 47.4443%〔4511〕 |

完整9508行经原CPU函数核对Box／GT阈值，GPU与CPU命中翻转0。对4509父模型，@0.25修复4／破坏2、净+2；@0.50修复18／破坏16、净+2；最终选中Query改变0。当前控制组同Query内部粗框4495到最终框4511，修复42／破坏26、净+16；这是本模型内部精修作用，不能将+16当成相对4509或独立baseline增益。

该检查点Mask完整三项为5812／9508、5133／9508及47.107628% mIoU，原始回执与逐行标量均保留；CPU独立重算的是框阈值，Mask和候选覆盖仍按已有标量核对，不称原始点级Mask重新计算。本轮暂只有native_gt控制结果，member_gt策略尚无终态，不能用这+2声称扰动前成员目标更有效或稳定显著涨点。

当前4509保护权重继续保留，没有晋级或删除；待策略组完整9508、两组CPU核对及fresh实际终态审查，再依严格Acc@0.50改进规则确定保留头。现有8文件恢复／退出见证通过，仍不等于整组terminal integrity审查完成。新正式模型候选4511距50%线4754尚差243，距V99同一行4797差286；这只是候选指标差，不重新定义总目标。

依据实际控制组从启动到formal结束8385.1874秒，策略组完整结束新估计{control_estimate['member_formal_end_estimate_cst']}，下一次人工结果核对{control_estimate['next_manual_outcome_check_cst']}附近；均为估计，GT资格数量可能改变吞吐。既有观察器38531持续240秒周期，无新观察器、模型、教师或推理排名；活动配置保持。完整行、CPU核对及估计证据在refine-logs/pvground_auxiliary_target_20261005/，目标ACTIVE_UNMET。
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
            stream.write('\n- ' + stamp + ' Native-target control full9508 closed and CPU-recounted5616/4511, parent delta+2/+2; member results pending, no promotion/cleanup.\n')
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
                           'Record closed native-target control and independent full-row recount'])
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
record.update(time_cst=stamp, section='20.376.74', heads=heads, github_main=heads[0],
              handoff_bytes=len(new), handoff_sha256=digest, four_local_and_remote_equal=True,
              payload_count=len(payloads), status='CLOSED_NATIVE_CONTROL_FULL_ROWS_PUBLISHED_MEMBER_PENDING',
              new_accuracy_result=False, preflight_passed=True, formal_fit_launched=True,
              formal_fit_started_cst=launch['time_cst'], observer_native_session=observer['native_session_id'])
record.update(publication_storage_relocation=relocation,closed_result_tools_executed=False,retention_tools_executed=False)
record.update(actual_scheduled_progress=progress,revised_stage_estimate=estimate)
record.update(control_formal_hits=[5616,4511],control_formal_finished_cst=intake['formal_finished_cst'],control_CPU_recount_pass=True,member_result_unobserved=True,full_pair_complete=False,revised_member_estimate=control_estimate)
(root / 'control_closed_publication.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
current = json.loads(state_path.read_bytes())
current.update(latest_publication=str(root / 'control_closed_publication.json'), published_heads=heads,
               handoff_section='20.376.74', handoff_sha256=digest,
               current_goal_turn_classification='PROGRESS_CLOSED_CONTROL_FULL_ROWS_COLLECTED_AND_RECOUNTED')
state_path.write_text(json.dumps(current, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
with Path(r'C:\Users\gb\memory\2026-10-05.md').open('a', encoding='utf-8') as stream:
    stream.write('\nPVGround ' + stamp + ': Native_gt control formal actually closed19:30:38, full9508 CPUrecount5616/4511 vs4509parent+2/+2, strict18repair16damage, selectedQchanges0. 8closed files including7.623MBrows collected, restore/exitpass; no parent promotion/delete, memberstrategy pending. Memberfinish estimate21:50:24, nextmanual21:48:24; same observer38531. Doc74 Main ' + heads[0] + ', ACTIVE_UNMET.\n')
print(json.dumps({key: record[key] for key in ('time_cst', 'section', 'heads', 'handoff_sha256',
      'four_local_and_remote_equal', 'formal_fit_started_cst', 'formal_fit_launched')}), flush=True)
