"""Publish the actual authorized cleanup and started fit, without NN queries."""
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
receipt_path = root / 'fit_start_publication.json'
assert not receipt_path.exists()
review = json.loads((root / 'PUBLISH_FIT_START_SOURCE_REVIEW.json').read_bytes())
assert review['execution_scope'] == 'SOURCE_ONLY'
assert review['verdict'] in ('PASS', 'WARN') and not review['blocking_findings']
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
fit_review = json.loads((root / 'FIT_SOURCE_REVIEW_TIMING_REV2.json').read_bytes())
assert fit_review['execution_scope'] == 'SOURCE_ONLY' and fit_review['verdict'] == 'PASS'
assert not fit_review['blocking_findings']
for item in fit_review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
launch = json.loads((root / 'fit_launch.json').read_bytes())
observer = json.loads((root / 'fit_observer_started.json').read_bytes())
cleanup = json.loads((root / 'CLEANUP_RESULT.json').read_bytes())
policy = json.loads((root / 'CLEANUP_POLICY.json').read_bytes())
old_root = root.parent / 'pvground_reference_keep_20261006'
full_cleanup = json.loads((old_root / 'archived_array_cleanup_receipt.json').read_bytes())
approval = json.loads((old_root / 'archived_array_cleanup_authorization.json').read_bytes())
assert approval['approved'] and approval['actual_user_reply'] == '清理，以后不用我审批无用的权重这些'
assert hashlib.sha256((old_root / 'archived_array_cleanup_receipt.json').read_bytes()).hexdigest() == cleanup['full_receipt_sha256']
assert full_cleanup['deleted_count'] == cleanup['deleted_count'] == 4756
assert full_cleanup['released_file_bytes'] == cleanup['released_file_bytes'] == 505731567
assert full_cleanup['deleted_manifest_sha256'] == approval['manifest_sha256']
assert full_cleanup['user_authorization_sha256'] == hashlib.sha256((old_root / 'archived_array_cleanup_authorization.json').read_bytes()).hexdigest()
assert cleanup['status'] == 'APPROVED_ARCHIVED_CURRENT_ARRAYS_REMOVED'
assert cleanup['local_archive_preserved'] and cleanup['best_weights_touched'] == 0
assert not policy['future_repeated_approval_required']
assert launch['status'] == 'PAIRED_FACE_FIT_LAUNCHED_NOT_COMPLETED'
assert launch['controller_pid'] == 827075 and launch['accuracy_result'] is False
assert launch['resources']['data_free_bytes'] >= launch['resources']['required_reserve_bytes']
assert launch['updates_per_arm'] == 3723 and launch['first_check_seconds'] == 13200
assert observer['first_check_seconds'] == 13200 and observer['poll_seconds'] == 240
assert observer['observer_local_pid'] == 13304
assert datetime.datetime.fromisoformat(observer['first_observation_cst']) == (
    datetime.datetime.fromisoformat(launch['time_cst']) + datetime.timedelta(seconds=13200))
previous_path = root / 'completed_preflight_publication.json'
previous = json.loads(previous_path.read_bytes())
assert previous['section'] == '20.376.94'
state_path = root.parent / 'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state = json.loads(state_path.read_bytes())
assert Path(state['latest_publication']).resolve() == previous_path.resolve()
assert state['face_support_fit_actual_started'] and state['face_support_fit_observer_native_session'] == 43488
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
finish = (datetime.datetime.fromisoformat(launch['time_cst']) + datetime.timedelta(seconds=launch['estimated_seconds'])).isoformat()
prefix = 'refine-logs/pvground_face_support_20261007/fit_start/'
section = f"""

## 20.376.95 已授权归档数组清理，六面观测同前向正式对照启动（{stamp}）

用户明确指示“清理，以后不用我审批无用的权重这些”。在我们产生的已结束非最佳权重、已有完整核验本地副本的临时候选数组范围内，后续直接清理，不再重复请求许可；最佳权重、必要PV/原G/V99依赖、活动恢复状态、数据集及实验结果记录保留。这一授权不扩大到无关文件。当前按已审查入口、清单及实际授权记录，于{cleanup['time_cst']}仅删除reference_keep_20261006四个声明目录中的4756份NPZ，文件合计505731567字节，完整本地归档保留；权重、日志、数据集触及数均为0。实际文件系统空闲由{cleanup['free_bytes_before']}增至{cleanup['free_bytes_after']}字节，不将文件字节总数等同于文件系统变化。

空间阻塞已解除。{launch['time_cst']}正式controller827075启动（screen pvg_face_support_fit_20261007）；启动时保存预算{launch['resources']['required_reserve_bytes']}字节、数据盘可用{launch['resources']['data_free_bytes']}字节，GPU空闲且M0已闭合。沿用§20.376.93—94的结构、同一冻结父模型前向、两个独立456102参数几何头与优化器：face_center原近邻对face_region矩形距离选点，各29778条fit一次、3723更新，B8/累积1/seed2027。重新加载保护的5598/4848与新优化器，不承接两步预检；尚无本轮新正式精度或有效模块结论。

训练入口定时修正已SOURCE_ONLY审查59实际文件PASS/0blocking（same-family/provisional，实际后端身份未认证），仅修正观察时间与审查报告绑定，没有变更NN、输入、监督或预算。唯一观察器native43488/Windows13304于{observer['time_cst']}启动，首次{observer['first_observation_cst']}，预计完整结束{finish}，之后必要时每240秒检查；预计时刻不是结果。该观察器在首次时间前仅本地等待，闭合后自动归档；本次发布不查询活动训练或提前收取结果。后续沿用已记录的ScanRefer双阈值、三项有效贡献及Sr3D/Nr3D顺序，见§20.376.94。
"""
new = old + section.encode('utf-8')
assert new.startswith(old) and new.count(b'## 20.376.95 ') == 1
names = ['fit_launch.json', 'fit_resource_check.json', 'fit_observer_started.json',
         'FIT_SOURCE_REVIEW_TIMING_REV2.json', 'FIT_SOURCE_REVIEW_TIMING_REV2.md',
         'launch_fit_authorized.py', 'observe_fit_authorized.py', 'collect_closed_fit_authorized.py',
         'CLEANUP_RESULT.json', 'CLEANUP_POLICY.json', 'publish_fit_start_and_cleanup_authorized.py',
         'PUBLISH_FIT_START_SOURCE_REVIEW.json', 'PUBLISH_FIT_START_SOURCE_REVIEW.md']
payloads = {prefix + name: (root / name).read_bytes() for name in names}
payloads[prefix + '.gitattributes'] = b'** -text\n'
assert all(not name.endswith(('.pt', '.pth', '.npz')) and '.aris' not in name for name in payloads)
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
spec = json.loads((root / 'pair_spec.json').read_bytes())
code = '''import base64,hashlib,json,sys
from pathlib import Path
project=Path(sys.argv[1]);b=json.load(sys.stdin);doc=project/b['doc'];old=doc.read_bytes()
assert hashlib.sha256(old).hexdigest()==b['old_sha256']
evidence=(project/b['prefix']).resolve()
assert str(evidence)=='/root/autodl-tmp/mcln_published_evidence_20261005/pvground_face_support_20261007/fit_start'
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
            stream.write('\n- ' + stamp + ' Authorized archived-array cleanup completed and shared-parent face-member formal fit started; no new accuracy yet.\n')
        stage += ['MANIFEST.md', *payloads]
    changed = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage], stderr=subprocess.DEVNULL)
    subprocess.check_call(['git', '-C', str(repo), '-c', 'core.whitespace=cr-at-eol,-blank-at-eof', 'diff', '--cached', '--check', '--', *stage])
    if index < 2:
        assert all(subprocess.check_output(['git', '-C', str(repo), 'show', ':' + name]) == raw for name, raw in payloads.items())
    assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + doc]).startswith(
        subprocess.check_output(['git', '-C', str(repo), 'show', previous['heads'][index] + ':' + doc]))
    subprocess.check_call(['git', '-C', str(repo), 'commit', '--quiet', '-m', 'Start paired face support fit after authorized archive cleanup'])
    heads.append(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip())
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
subprocess.check_call(['git', '-C', str(repos[0]), '-c', 'http.version=HTTP/1.1', 'push', 'origin', 'HEAD:main'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == heads[0]
digest = hashlib.sha256(new).hexdigest()
assert digest == remote['handoff_sha256'] and all(path.read_bytes() == new for path in copies)
guard = workspace / '.codex/tmp/sync_cs_handoff_remote_20260923.py'; guard_raw = guard.read_bytes()
assert guard_raw.count(previous['handoff_sha256'].encode()) == 1
guard.write_bytes(guard_raw.replace(previous['handoff_sha256'].encode(), digest.encode()))
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), section='20.376.95', heads=heads,
    handoff_sha256=digest, four_local_and_remote_equal=True, github_main=heads[0], payload_count=len(payloads),
    execution_scope='ACTUAL_ARCHIVE_CLEANUP_AND_FIT_START', new_formal_result=False,
    raw_npz_published=False, weights_published=False, formal_fit_started=True,
    controller_pid=launch['controller_pid'], observer_native_session=43488, observer_local_pid=13304,
    first_observation_cst=observer['first_observation_cst'], estimated_finish_cst=finish)
receipt_path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
state.update(time_cst=record['time_cst'], latest_publication=str(receipt_path), handoff_section=record['section'],
    handoff_sha256=digest, published_heads=heads, owned_gpu_job_active=True, overall_goal_complete=False)
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record), flush=True)
