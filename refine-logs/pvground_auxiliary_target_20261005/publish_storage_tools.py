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
previous = json.loads((root / 'fit_launch_publication.json').read_bytes())
assert previous['section'] == '20.376.70'
assert not (root / 'storage_tools_publication.json').exists()
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
assert b'## 20.376.71 ' not in old


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args]).decode().strip()


for repo, head in zip(repos, previous['heads']):
    assert git(repo, 'rev-parse', 'HEAD') == head and not git(repo, 'status', '--porcelain')
stamp = datetime.datetime.now().astimezone().isoformat()
relocation = json.loads((root / 'PUBLICATION_STORAGE_RELOCATION.json').read_bytes())
assert relocation['status'] == 'PUBLISHED_EVIDENCE_RELOCATED_LOGICAL_PATH_PRESERVED'
assert relocation['source_and_destination_file_sizes_and_sha256_exact']
assert relocation['records_deleted'] == relocation['weights_deleted'] == 0
section = f'''

## 20.376.71 实验记录迁到数据盘，终态收集与非最佳权重清理工具已准备（{stamp}）

启动时系统盘仅约43MiB可用，存储核对确认refine-logs实际占用约420MiB且位于系统盘。{relocation['time_cst']}已将整个静态发布证据目录迁至{relocation['data_location']}，原路径{relocation['logical_path']}通过目录链接保持访问。迁移前后4355项文件／链接及428491229字节实际文件内容均核对一致，6个已有绝对链接保留；没有删除实验记录或权重，也没有训练／优化器重放。首次迁移检查因发现已有链接而在任何移动前停止；按真实链接状态修正后完成上述迁移。

此次释放系统盘439902208字节；实际可用空间由44732416增至484634624字节，数据盘余量1787027456字节。这里是存储操作时的测量，不推断之后余量。运行中的正式训练目录、PV／原G／4509权重及模型源码未迁移。后续SFTP发布仍使用原refine-logs路径，证据写入数据盘。

已复用上轮只读工具准备本轮闭合结果收集与CPU重算：collect_formal_authorized.py、geometry_result_metrics.py、analyze_closed_formal.py。新重算比较control／member_target与4509父模型，核对两组权重1、实际目标模式、29778条相同行序和3723更新，终点累计11169；同时记录9508实际命中、修复／破坏、Mask及候选覆盖，不下载权重或重放模型。工具尚未执行，没有新增精度结果。

闭合后的retain_metric_best.py及授权入口已准备，尚未审查或执行。必须待两组完整评估、CPU核对、终态审查和清理源码审查通过，确认完整10状态及模型／优化器恢复回执后，才按Acc@0.50优先的实际指标保留最好、删除本轮已结束非最佳头；不碰原G／官方PV／V99，不新建负结果权重归档。当前仍保留5614／4509强起点。

正式fit及唯一观察器沿用§20.376.70实际启动记录：controller630343、native观察器38531、18:54:13首查、之后240秒间隔，预计21:44:13附近闭合。存储操作没有新增GPU进程查询或提前训练轮询，也没有更改训练配置。本次没有formal终态或新REC结果，目标继续ACTIVE_UNMET。存储回执与上述源码位于refine-logs/pvground_auxiliary_target_20261005/。
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
            stream.write('\n- ' + stamp + ' Published evidence moved to data disk with exact contents and logical paths; closed-result and retention tools prepared, not executed.\n')
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
                           'Preserve published evidence on data disk and prepare closed-result tools'])
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
record.update(time_cst=stamp, section='20.376.71', heads=heads, github_main=heads[0],
              handoff_bytes=len(new), handoff_sha256=digest, four_local_and_remote_equal=True,
              payload_count=len(payloads), status='STORAGE_RELOCATION_AND_CLOSED_TOOLS_PREPARATION_PUBLISHED',
              new_accuracy_result=False, preflight_passed=True, formal_fit_launched=True,
              formal_fit_started_cst=launch['time_cst'], observer_native_session=observer['native_session_id'])
record.update(publication_storage_relocation=relocation,closed_result_tools_executed=False,retention_tools_executed=False)
(root / 'storage_tools_publication.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
current = json.loads(state_path.read_bytes())
current.update(latest_publication=str(root / 'storage_tools_publication.json'), published_heads=heads,
               handoff_section='20.376.71', handoff_sha256=digest,
               current_goal_turn_classification='PROGRESS_STORAGE_RELOCATION_AND_CLOSED_TOOLS')
state_path.write_text(json.dumps(current, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
with Path(r'C:\Users\gb\memory\2026-10-05.md').open('a', encoding='utf-8') as stream:
    stream.write('\nPVGround ' + stamp + ': Published evidence relocated from system disk to /root/autodl-tmp/mcln_published_evidence_20261005 with original refine-logs symlink, 4355entries/428491229bytes exact and6absolute links preserved. System free44.7MB to484.6MB, no weights or records deleted; first attempt stopped before moving on existing-link assertion. Closed result/retention tools prepared but notexecuted; cleanup still needs actual terminal and source review. Doc71 Main ' + heads[0] + ', ACTIVE_UNMET; same live observer38531 waits18:54.\n')
print(json.dumps({key: record[key] for key in ('time_cst', 'section', 'heads', 'handoff_sha256',
      'four_local_and_remote_equal', 'formal_fit_started_cst', 'formal_fit_launched')}), flush=True)
