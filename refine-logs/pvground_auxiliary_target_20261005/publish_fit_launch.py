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
previous = json.loads((root / 'source_publication.json').read_bytes())
assert previous['section'] == '20.376.69'
assert not (root / 'fit_launch_publication.json').exists()
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
assert b'## 20.376.70 ' not in old


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args]).decode().strip()


for repo, head in zip(repos, previous['heads']):
    assert git(repo, 'rev-parse', 'HEAD') == head and not git(repo, 'status', '--porcelain')
stamp = datetime.datetime.now().astimezone().isoformat()
section = f'''

## 20.376.70 辅助定位目标坐标对照：两组真实预检结束，正式同预算训练已启动（{stamp}）

两组GPU预检实际于{wait['status']['finished_cst']}结束，controller及两组退出码均为0；各完成2次优化更新，没有生成磁盘权重。21份原始回执共98022字节已完整收集。实际输入与已闭合CPU面板的采样点、原生扰动GT及扰动前成员框一致；冻结父模型与零R状态、隔离额外定位梯度、缓存上游bbs和Mask保持、模型与优化器内存保存恢复均通过。control两步额外候选数146／143，member_target为122／140；超出分布节点的面数分别120／126和45／40。这是单batch工程见证，不是完整验证精度或全训练显存测量。

启动就绪审查实读上述闭合回执及启动器，PASS、阻塞0；审查仍标记same-family/provisional、backend unattested，不称外部独立审查。正式fit实际于{launch['time_cst']}启动，controller PID={observer['controller_pid']}，screen={launch['screen']}。启动时GPU空闲，数据盘可用{launch['resources']['data_free_bytes']}字节，已满足实际保存余量门槛。这里记录已启动，不推断后续更新或终态。

两组从当前完整4509几何头及相同PV／原G父权重重新加载，创建新优化器，不承接预检状态。仅训练原有456102参数、10状态几何头，父模型、Mask、语义路径及零R冻结。共同监督为native＋G＋原匹配DFL及权重1的额外几何项；control额外项使用原生独立扰动root框，member_target使用原生50000点scan表示中独立GT框扰动前的标注成员范围。选择的坐标定义同时用于额外Box资格与L1／GIoU／DFL目标，属于辅助几何资格和目标定义的整体对照。候选自身Query及融合Mask的训练GT资格、原匹配排除和每表达再batch平均不变；原生GT、匹配、原损失、G标签、对象输入及正式评估不改。

每组29778条fit输入各一次，batch8、累积1、3723次更新、尾batch2，seed2027、lr1e-5、WD0.0005、clip0.1。几何头从累计7446更新出发，正式终点累计11169；原G适配历史另计。初始及终点6887条模块留出与9508条正式开发验证分别记录。推理保留全部256候选、唯一last/bbs及同Query框／Mask，没有GT资格输入、候选删除、教师或第二套排名。

唯一只读观察器已启动，native session={observer['native_session_id']}。依据上一轮实际两组耗时估算，本次首查{observer['first_expected_observation_cst']}，预计两组闭合约{observer['estimated_pair_finish_cst']}，首查之后240秒间隔；估计不等于服务器当前进度，不额外启动重复轮询。当前保留最好正式结果仍为5614／4509，即59.0450%／47.4232%；本次尚无新精度结果。距离严格50%线4754仍差245条，不保证本轮达到目标。

闭合后同时比较本轮control和4509强起点，并保留完整9508计数、修复／破坏、候选几何及Mask代价；结果审查后及时删除本次非最佳终点，原G／官方PV和受保护V99权重保持。实际源码、两组spec、审查、原始预检、启动和单观察器证据见refine-logs/pvground_auxiliary_target_20261005/。主线仍为PV-Ground，研究目标ACTIVE_UNMET；空间参考与渐进边界设计等待本轮实际证据，不在活动训练中修改规则。
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
           ('SOURCE_REVIEW.json', 'SOURCE_REVIEW.md', 'LAUNCH_REVIEW.json', 'LAUNCH_REVIEW.md')]
for index, repo in enumerate(repos):
    stage = [doc]
    if index < 2:
        with (repo / 'MANIFEST.md').open('a', encoding='utf-8') as stream:
            stream.write('\n- ' + stamp + ' Auxiliary-target two-step sanity closed PASS; formal native/member pair launched from4509, no new accuracy yet.\n')
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
                           'Record closed auxiliary-target sanity and actual formal fit launch'])
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
record.update(time_cst=stamp, section='20.376.70', heads=heads, github_main=heads[0],
              handoff_bytes=len(new), handoff_sha256=digest, four_local_and_remote_equal=True,
              payload_count=len(payloads), status='CLOSED_SANITY_AND_REAL_FORMAL_LAUNCH_PUBLISHED',
              new_accuracy_result=False, preflight_passed=True, formal_fit_launched=True,
              formal_fit_started_cst=launch['time_cst'], observer_native_session=observer['native_session_id'])
(root / 'fit_launch_publication.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
current = json.loads(state_path.read_bytes())
current.update(latest_publication=str(root / 'fit_launch_publication.json'), published_heads=heads,
               handoff_section='20.376.70', handoff_sha256=digest,
               current_goal_turn_classification='PROGRESS_REAL_AUXILIARY_TARGET_FORMAL_FIT')
state_path.write_text(json.dumps(current, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
with Path(r'C:\Users\gb\memory\2026-10-05.md').open('a', encoding='utf-8') as stream:
    stream.write('\nPVGround ' + stamp + ': Auxiliary native/member target preflight closed PASS2steps/arm, no saved weights; formal pair actually launched17:10:53 from4509, each3723updates, nativeGT protocol untouched. Sole observer38531, first18:54:13 then240s, pairestimate21:44:13. Doc70 and raw evidence published. Best4509 retained; ACTIVE_UNMET. Main ' + heads[0] + '.\n')
print(json.dumps({key: record[key] for key in ('time_cst', 'section', 'heads', 'handoff_sha256',
      'four_local_and_remote_equal', 'formal_fit_started_cst', 'formal_fit_launched')}), flush=True)
