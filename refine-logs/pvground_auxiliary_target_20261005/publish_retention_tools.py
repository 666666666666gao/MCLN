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
previous = json.loads((root / 'storage_tools_publication.json').read_bytes())
assert previous['section'] == '20.376.71'
assert not (root / 'retention_tools_publication.json').exists()
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
assert b'## 20.376.72 ' not in old


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
section = f'''

## 20.376.72 终态工具源码审查通过，主指标持平时保留4509（{stamp}）

终态收集、CPU重算和限定路径清理工具完成SOURCE_ONLY审查，PASS、阻塞0，24份当前文件SHA核对；同一既有审查上下文的follow-up、same-family/provisional、backend unattested，不称fresh或外部独立。审查核对了4509父rows的实际9508重算、control／member_target目标模式、两组额外权重1及完整头恢复条件。本轮结果尚未闭合，这不是本轮精度或终态审查。

审查发现原重算继承的(hits50,hits25,tie-parent)排序与本轮计划“严格Acc@0.50提高才替换4509”不一致。已改为(hits50,tie-parent,hits25)，对应生成器同步；精确3路径清理脚本在任何unlink前也核验赢家是原父模型，或其严格命中大于4509。训练、匹配、损失和正式评价代码没有变化，未执行任何本轮权重删除；修正发生在本轮结果出现之前。

收集器、原CPU度量及授权入口保留上轮实现；新两生成器与产物一致。清理仍需实际两组闭合、9508回执、CPU重算及另一次真实终态审查，不以源码PASS直接开始删除。完整10状态替代头、原G／官方PV／V99保护和不归档负权重规则继续执行。当前最佳仍5614／4509，尚无新增REC结果；既有观察器38531按18:54:13首查和240秒后续间隔继续等待。源码、审查和修正绑定见refine-logs/pvground_auxiliary_target_20261005/。
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
            stream.write('\n- ' + stamp + ' Closed-result/retention source review PASS24; strict primary improvement required for replacing4509; no current result or weight cleanup.\n')
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
                           'Require strict primary gain before replacing retained geometry best'])
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
record.update(time_cst=stamp, section='20.376.72', heads=heads, github_main=heads[0],
              handoff_bytes=len(new), handoff_sha256=digest, four_local_and_remote_equal=True,
              payload_count=len(payloads), status='REVIEWED_STRICT_PROMOTION_AND_CLOSED_TOOLS_PUBLISHED',
              new_accuracy_result=False, preflight_passed=True, formal_fit_launched=True,
              formal_fit_started_cst=launch['time_cst'], observer_native_session=observer['native_session_id'])
record.update(publication_storage_relocation=relocation,closed_result_tools_executed=False,retention_tools_executed=False)
(root / 'retention_tools_publication.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
current = json.loads(state_path.read_bytes())
current.update(latest_publication=str(root / 'retention_tools_publication.json'), published_heads=heads,
               handoff_section='20.376.72', handoff_sha256=digest,
               current_goal_turn_classification='PROGRESS_REVIEWED_STRICT_PROMOTION_CORRECTION')
state_path.write_text(json.dumps(current, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
with Path(r'C:\Users\gb\memory\2026-10-05.md').open('a', encoding='utf-8') as stream:
    stream.write('\nPVGround ' + stamp + ': Closed-result and retention tool source review PASS24, same-context/same-family provisional. Fixed real plan mismatch in CPU rank to hits50,tie-parent,hits25 and strict-gain guard before anyunlink; no current result or cleanup. Doc72 Main ' + heads[0] + ', ACTIVE_UNMET, observer38531 first18:54.\n')
print(json.dumps({key: record[key] for key in ('time_cst', 'section', 'heads', 'handoff_sha256',
      'four_local_and_remote_equal', 'formal_fit_started_cst', 'formal_fit_launched')}), flush=True)
