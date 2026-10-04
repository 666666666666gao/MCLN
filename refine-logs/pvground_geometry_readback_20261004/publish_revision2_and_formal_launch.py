"""Publish actual failed/preflight receipts and the reviewed formal launch once."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import paramiko

local = Path(__file__).resolve().parent
revision = local / 'revision2'
formal = local / 'formal_draft'
face = local.parent / 'pvground_face_conditioned_20261004'
receipt_path = local / 'revision2_formal_launch_publication.json'
assert not receipt_path.exists()
previous = json.loads((face / 'readback_launch_publication.json').read_bytes())
proof = json.loads((revision / 'ACTUAL_PREFLIGHT_SUMMARY.json').read_bytes())
assert proof['status'] == 'ACTUAL_BOTH_ARMS_TWO_UPDATE_PASS'
review = json.loads((formal / 'READBACK_FORMAL_SOURCE_REVIEW.json').read_bytes())
assert review['verdict'] == 'PASS' and not review['blocking_findings']
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
launch = json.loads((formal / 'launch.json').read_bytes())
assert launch['updates_per_arm'] == 3723 and launch['effective_batch'] == 8
assert not launch['accuracy_result'] and not launch['completed']
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
prefix = 'refine-logs/pvground_geometry_readback_20261004/'
payloads = {}
for name in ('READBACK_PREFLIGHT_FAILURE_REVIEW.md', 'READBACK_PREFLIGHT_FAILURE_REVIEW.json',
             'readback_preflight_wait.json', 'publish_revision2_and_formal_launch.py'):
    payloads[prefix + name] = (local / name).read_bytes()
for source_root in (revision, formal):
    for path in sorted(source_root.iterdir()):
        if path.is_file() and path.suffix in ('.py', '.json', '.md', '.txt', '.pyfrag'):
            payloads[prefix + path.relative_to(local).as_posix()] = path.read_bytes()
    for path in sorted((source_root / 'runtime_bundle').glob('*.py')):
        payloads[prefix + path.relative_to(local).as_posix()] = path.read_bytes()
for path in sorted((revision / 'source_preview/PV-Ground/models').glob('*.py')):
    payloads[prefix + path.relative_to(local).as_posix()] = path.read_bytes()
for source_root in (local, revision):
    intake_path = source_root / 'complete_preflight/INTAKE.json'
    intake = json.loads(intake_path.read_bytes())
    assert intake['downloaded_weight_files'] == 0
    payloads[prefix + intake_path.relative_to(local).as_posix()] = intake_path.read_bytes()
    for name, identity in intake['files'].items():
        path = source_root / 'complete_preflight' / name
        raw = path.read_bytes()
        assert len(raw) == identity['bytes'] and hashlib.sha256(raw).hexdigest() == identity['sha256']
        payloads[prefix + path.relative_to(local).as_posix()] = raw
assert not any('/.aris/' in name or name.endswith(('.pth', '.pt')) for name in payloads)
stamp = datetime.datetime.now().astimezone().isoformat()
first = datetime.datetime.fromisoformat(launch['time_cst']) + datetime.timedelta(seconds=launch['first_check_seconds'])
eta = datetime.datetime.fromisoformat(launch['time_cst']) + datetime.timedelta(seconds=launch['estimate_seconds'])
section = f'''

## 20.376.54 回读预检失败边界、修正后实际通过及固定4506几何的正式对照启动（{stamp}）

此前§53记录的是原预检已启动的历史快照。原预检随后在两个独立完整前向的last_center逐位相等断言处失败：CPU工厂严格加载成功，GPU尚未优化更新，0次更新、0个新权重、无正式精度。失败日志及新鲜失败审查保留。修正后的V2仍使用相同模型计算，改为同一次前向在R前后严格比较几何、Mask和对比输出；原输入字典也核对不变，没有采用宽容差判定通过。关闭R的重复完整前向诊断确实观察到既有浮点差异，但具体算子原因没有被证实，不解释数个百分点精度差距。

V2来源审查为34文件SOURCE_ONLY PASS。实际控制器已于{proof['actual_terminal_cst']}完成，两组CPU工厂均实际构建96672参数/23项新状态；evidence_hidden及evidence_visible各完成2次真实GPU更新，共4次。零残差Query和缓存原生语义头回放严格相同；真实部署末层语义子头只调用一次，同帧几何/Mask/对比输出保持严格相同。冻结父参数及持久状态、独立末层CE+G输出梯度、更新后内部梯度、内存模型delta和完整AdamW恢复均通过。只收集28份日志/JSON/退出证据，0份权重；这些是工程预检结果，不是9508条正式精度结果。

正式caller、控制器、原生PV评估接口及清理策略另外经过新鲜42文件SOURCE_ONLY PASS，无阻断项，same-family/provisional；要求的Astra/max后端没有独立证实。审查实际CPU复算历史父模型9508行得到5616/4506、阈值差异0，不能写成R的新成绩。审查源码PASS与执行者观察到的V2实际运行PASS分别记录。

通过两项门槛后，正式对照控制器实际于{launch['time_cst']}启动，PID {launch['process'].split()[0]}，根目录`{launch['root']}`，screen `{launch['screen']}`。启动证据确认控制器存在，不代表此时已完成正式优化或评估。两组依次evidence_hidden/evidence_visible，从同一官方PV、原G及已训练4506平铺边界分布权重构建，冻结全部父参数和eval状态，只训练96672参数回读单元。两组容量、完整文本、原Query、六面角色相同，只有44维几何证据在编码前隐藏/可见不同。保留全部256候选、唯一原生last/bbs及同Query的Mask；没有新质量损失、教师、扩展对比正例或部署双源。

预算分别为seed2027、物理/有效batch8、29778条fit各遍历一次，3722个完整batch加末批2条，3723次优化更新；fresh AdamW学习率1e-5、weight decay5e-4、clip0.1。6887条预训练见过场景的模块留出与9508条正式开发验证分别记录。原生native+G监督不改，最终部署框保持冻结；评估额外缓存回放R之前的语义输入，仅用于同帧固定框诊断，并不是未训练R的模型消融，也不是第二套部署排名。

唯一正式只读观察器已启动，首次预计{first.isoformat()}，随后每240秒检查；总耗时估计约4.5小时、预计{eta.isoformat()}附近结束，依据此前实际单组fit6561秒及正式评估1478秒。这些是时间估计，不是结果。启动前实际数据盘空闲{launch['resources']['directory_free_bytes']}字节、系统盘{launch['resources']['system_free_bytes']}字节。活动组只保留原子替换latest恢复状态；完成新鲜进程模型/优化器恢复、9508原生评估和CPU阈值复算后删除本实验非最佳权重，不下载负结果权重归档。当前最佳5616/4506及其官方PV/原G重建父链必须保留，即使R后续提高也不能误删依赖父权重。

此时没有R正式精度，ScanRefer目标5615/4754仍未达到，距严格门槛248条。后续看可见证据对同容量隐藏控制与当前最佳的净修复/破坏、同框回读作用及全256候选空间；不预先声称三个创新成立或Nr/Sr有效。旧预检/源码审查/发布工具不重新执行，旧失败原始记录不覆盖。
'''
new = old + section.encode('utf-8')
assert new.startswith(old) and new.count(b'## 20.376.54 ') == 1
attributes = '\n# Preserve actual readback diagnostic and source evidence bytes.\n'
attributes += prefix + 'complete_preflight/** -text whitespace=cr-at-eol,-blank-at-eol,-blank-at-eof\n'
attributes += prefix + 'revision2/complete_preflight/** -text whitespace=cr-at-eol,-blank-at-eol,-blank-at-eof\n'
attributes += prefix + 'revision2/source_preview/** -text whitespace=cr-at-eol,-blank-at-eol,-blank-at-eof\n'
for relative, raw in payloads.items():
    if raw.endswith((b'\n\n', b'\r\n\r\n')) and '/complete_preflight/' not in relative:
        attributes += relative + ' -text whitespace=cr-at-eol,-blank-at-eof\n'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
remote = '/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(remote + '/' + doc, 'rb') as stream:
    assert stream.read() == old
new_dirs = set()
for relative in payloads:
    parent = Path(relative).parent
    while parent.as_posix() != prefix.rstrip('/'):
        new_dirs.add(parent.as_posix())
        parent = parent.parent
for directory in sorted(new_dirs, key=lambda name: (name.count('/'), name)):
    sftp.mkdir(remote + '/' + directory)
for repo in repos[:2]:
    with (repo / '.gitattributes').open('ab') as stream:
        stream.write(attributes.encode('utf-8'))
    for relative, raw in payloads.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    with (repo / 'MANIFEST.md').open('a', encoding='utf-8') as stream:
        stream.write(f'\n- {stamp} actual V1 preflight failure0updates retained; corrected V2 both2-update GPU PASS; fresh42-file formal SOURCE_ONLY PASS; frozen4506 hidden/visible formal controller launched, no accuracy result.\n')
for relative, raw in payloads.items():
    with sftp.open(remote + '/' + relative, 'wb') as stream:
        stream.write(raw)
    with sftp.open(remote + '/' + relative, 'rb') as stream:
        assert stream.read() == raw
with sftp.open(remote + '/.gitattributes', 'wb') as stream:
    stream.write((repos[0] / '.gitattributes').read_bytes())
with sftp.open(remote + '/.gitattributes', 'rb') as stream:
    assert stream.read() == (repos[0] / '.gitattributes').read_bytes()
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
    stage = [doc] + (['MANIFEST.md', '.gitattributes', *payloads] if index < 2 else [])
    changed = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage])
    subprocess.check_call(['git', '-C', str(repo), 'diff', '--cached', '--check', '--', *stage])
    for relative, raw in (payloads.items() if index < 2 else []):
        assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + relative]) == raw
    subprocess.check_call(['git', '-C', str(repo), 'commit', '-m', 'Record actual readback preflight closeout and frozen geometry formal launch'])
    heads.append(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip())
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
subprocess.check_call(['git', '-C', str(repos[0]), '-c', 'http.version=HTTP/1.1', 'push', 'origin', 'HEAD:main'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == heads[0]
digest = hashlib.sha256(new).hexdigest()
guard.write_bytes(guard_raw.replace(previous['handoff_sha256'].encode(), digest.encode()))
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), section='20.376.54',
    predecessor=str(face / 'readback_launch_publication.json'), heads=heads, github_main=heads[0],
    handoff_sha256=digest, handoff_bytes=len(new), four_local_and_remote_equal=True,
    exact_committed_payloads=True, payload_count=len(payloads), old_preflight_updates=0,
    revision2_preflight_updates=4, revision2_runtime_pass=True, formal_source_verdict='PASS_SOURCE_ONLY',
    formal_controller_launched=True, formal_accuracy_result=False, retained_metric_best=[5616,4506], goal_achieved=False)
receipt_path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
state_path = face / 'active_continuation_state.json'
state = json.loads(state_path.read_bytes())
state.update(time_cst=record['time_cst'], status='READBACK_FROZEN_GEOMETRY_FORMAL_PAIR_LIVE',
    latest_publication=str(receipt_path), github_publication_predecessor=str(receipt_path),
    published_heads=heads, handoff_sha256=digest, handoff_section='20.376.54',
    remote_fit=launch['root'], remote_controller_pid=int(launch['process'].split()[0]),
    readback_formal_training_started=True, readback_formal_result_available=False,
    native_readback_formal_launcher_closed=72439, native_observer_session_id=66222,
    sole_live_native_session_id=66222, native_observer_closed=False,
    observer_first_check=first.isoformat(), observer_poll_seconds=240,
    next_native_observer_check=first.isoformat(), controller_alive_at_launch=True,
    observer_resume_rule='Only66222 is live; first01:02:12CST October5, then240sec. No early SSH/relaunch.',
    next_actions=['Resume only original66222 near its scheduled ETA; no duplicate SSH observer.',
                  'Collect formal receipts once controller and original observer close; no weight archive.',
                  'Analyse native9508 bbs/repairs-damages/fixedframe effect and retain verified best with parents.'])
state_path.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
cursor = f'\nLatestactualpublication {receipt_path}, main{heads[0]}, doc54 fourlocal+remote exact. Formalcontroller512699 launched23:15:52CST; soleobserver66222 first{first.isoformat()},later240s; no Rformalaccuracy. V2actualboth2-stepPASS; reviewer42SOURCEPASS. Closed72439/24708/50701 neverresume. Protected4506+parents; goalACTIVE_UNMET.\n'
for path in (face / 'NEXT_CONTINUATION.md', local / 'NEXT_CONTINUATION.md'):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(cursor)
with (workspace / 'memory/2026-10-04.md').open('a', encoding='utf-8') as stream:
    stream.write('\nPVGround ' + record['time_cst'] + cursor)
print(json.dumps(record), flush=True)
