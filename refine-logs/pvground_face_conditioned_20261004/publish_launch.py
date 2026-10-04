"""Publish only the actual source gate, closed sanity and launched bounded fit."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import paramiko

local = Path(__file__).resolve().parent
assert not (local / 'launch_publication.json').exists()
previous = json.loads((local / 'source_publication.json').read_bytes())
review = json.loads((local / 'EXPERIMENT_CODE_REVIEW.json').read_bytes())
assert review['verdict'] in ('PASS', 'WARN') and not review['blocking_findings']
assert review['execution_scope'] == 'SOURCE_ONLY'
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
intake = json.loads((local / 'complete_preflight/INTAKE.json').read_bytes())
proof = json.loads((local / 'complete_preflight/face_conditioned/preflight.json').read_bytes())
launch = json.loads((local / 'launch.json').read_bytes())
assert intake['status']['status'] == 'complete' and intake['controller_exit'] == 0
assert proof['status'] == 'pass' and proof['optimizer_steps'] == 2
assert proof['head_parameters'] == 64737 and proof['head_architecture'] == 'face_conditioned'
assert proof['original_g_state_unchanged'] and proof['same_cached_inputs_zero_head_common_floor_exact']
assert len(proof['steps'][1]['p3_gradients']) == 25
assert all(value > 0 for value in proof['steps'][1]['p3_gradients'].values())
assert launch['formal_training_started'] and not launch['formal_result_available']
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
names = ['run_face_fit.py', 'controller.py', 'preflight_controller.py',
    'launch_preflight_authorized.py', 'launch_formal_authorized.py',
    'observe_preflight_authorized.py', 'wait_preflight_authorized.py', 'collect_preflight_terminal.py',
    'wait_fit_authorized.py', 'collect_terminal.py', 'EXPERIMENT_PLAN.md', 'EXPERIMENT_TRACKER.md',
    'face_preflight_spec.json', 'face_fit_spec.json', 'integration_preparation.json',
    'lifecycle_preparation.json', 'terminal_tools_preparation.json', 'CODE_REVIEW_REQUEST.txt',
    'EXPERIMENT_CODE_REVIEW.md', 'EXPERIMENT_CODE_REVIEW.json', 'ACTUAL_CODE_REVIEWER_RESPONSE.txt',
    'IMPLEMENTATION_SCOPE.md', 'PREFLIGHT_ANALYSIS.json', 'record_source_request.py',
    'record_source_result.py', 'record_review_followups.py', 'OBSERVED_RANK_RECORD_CORRECTION.json',
    'analyze_preflight.py', 'analyze_terminal.py',
    'preflight_launch.json', 'preflight_resource_check.json', 'launch.json',
    'idea-stage/docs/research_contract.md', 'publish_launch.py']
names += json.loads((local / 'integration_preparation.json').read_bytes())['modules']
names += [path.relative_to(local).as_posix() for path in (local / 'complete_preflight').rglob('*') if path.is_file()]
names += [path.relative_to(local).as_posix() for path in (local / '.aris/traces/experiment-bridge').rglob('*') if path.is_file()]
assert len(names) == len(set(names))
prefix = 'refine-logs/pvground_face_conditioned_20261004/'
payloads = {prefix + name: (local / name).read_bytes() for name in names}
stamp = datetime.datetime.now().astimezone().isoformat()
section = f'''

## 20.376.47 六面条件化解码：真实入口、两步GPU预检与固定预算启动（{stamp}）

§20.376.46记录的本地草稿已在隔离真实PV入口完成集成。新头实测64737参数、25项可训练状态；普通平面拼接分布对照为456102参数/10项。输入仍为原G的完整融合Mask范围与7×16局部成员，每个面读取对应轴32桶、自己的16个局部成员及中心16个成员。六面顺序x−/y−/z−/x+/y+/z+对应原局部槽2/4/6/1/3/5，共用33位置的分布解码及原生最后层匹配DFL/7。没有新增质量评分、教师、Top-k裁剪或测试GT门控，六面内部状态尚未回写最终语义头。结构与容量共同改变，不将差值解释为纯注意力效果。

实际新鲜源码审查为{review['verdict']}，0阻断；使用Astra/max/fork-none，同模型家族、暂定独立性，未独立验证后端SKU。源码审查与GPU工程证据分开保存。GPU预检在现有环境完成1个真实batch8的2次优化器更新、4次原生调用：缓存相同输入下均匀零偏移与共同尺寸下限粗框逐元素一致；单独定位项和DFL均到output.weight产生非零梯度；第2步25项新头梯度全非零；原G参数与运行状态未变；模型和AdamW实际内存序列化/恢复一致。预检分配峰值{proof['peak_allocated_bytes']}字节、保留峰值{proof['peak_reserved_bytes']}字节，序列化{proof['serialization_bytes']}字节；没有预检磁盘权重或正式准确率。

真实固定预算已启动：{launch['time_cst']}，{launch['root']}，控制器{launch['process'].split()[0]}。共同原G重新加载，新增头初始化，不从训练后的4506分布终点续训；上游参数/buffer和eval状态固定，只有新头学习。seed2027、物理/有效batch8、无累积、29778条fit各一次、尾batch2、3723次更新、重新初始化AdamW，LR1e−5/WD5e−4/clip0.1。保持6887模块留出及9508正式开发验证、原生last/bbs主Acc@0.50、对应Mask、修复/破坏、完整256候选覆盖和DFL饱和记录。此前跨进程非逐位限制保留。

当前真实最优仍为5616/4506（59.0660%/47.3917%），严格50%目标还差248条。新训练尚未终态，没有新增精度结果。只有完整9508、实际恢复、CPU重算和SHA确认后，才按4506门槛保留更优新终点/删除精确已归属的非最佳权重；原G、官方PV及必要V99依赖继续保护，不新增失败权重本地归档。启动时数据盘可用{launch['resources']['free_bytes']}字节、系统盘{launch['resources']['system_free_bytes']}字节；这只是启动快照。唯一观察器按约完成时间查看，再按实际吞吐估计并至少240秒间隔，不增加重复轮询。ScanRefer目标及后续独立Nr3D/Sr3D均未完成。
'''
new = old + section.encode('utf-8')
client = paramiko.SSHClient(); client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
    password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
remote = '/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(remote + '/' + doc, 'rb') as stream:
    assert stream.read() == old
made = set()
for relative, raw in payloads.items():
    if relative in (prefix+'pvground_face_conditioned_box_refiner.py', prefix+'IMPLEMENTATION_SCOPE.md'):
        assert (repos[0]/relative).read_bytes() == (repos[1]/relative).read_bytes()
        with sftp.open(remote+'/'+relative, 'rb') as stream:
            assert stream.read() == (repos[0]/relative).read_bytes()
    else:
        assert all(not (repo/relative).exists() for repo in repos[:2])
    for repo in repos[:2]:
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    parts = relative.split('/')
    for index in range(3, len(parts)):
        directory = remote + '/' + '/'.join(parts[:index])
        # Create each declared new nested directory exactly once.
        if directory not in made:
            sftp.mkdir(directory)
            made.add(directory)
    with sftp.open(remote + '/' + relative, 'wb') as stream:
        stream.write(raw)
    with sftp.open(remote + '/' + relative, 'rb') as stream:
        assert stream.read() == raw
for path in copies:
    path.write_bytes(new)
with sftp.open(remote + '/' + doc, 'wb') as stream:
    stream.write(new)
assert all(path.read_bytes() == new for path in copies)
with sftp.open(remote + '/' + doc, 'rb') as stream:
    assert stream.read() == new
sftp.close(); client.close()
heads = []
for index, repo in enumerate(repos):
    if index < 2:
        with (repo / 'MANIFEST.md').open('a', encoding='utf-8') as stream:
            stream.write(f'\n- {stamp} face-conditioned decoder actual two-update sanity passed; bounded original-G-frozen fit launched, no terminal accuracy yet.\n')
    stage = [doc] + (['MANIFEST.md', *payloads] if index < 2 else [])
    changed = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage])
    subprocess.check_call(['git', '-C', str(repo), '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--cached', '--check', '--', *stage])
    for relative, raw in (payloads.items() if index < 2 else []):
        assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + relative]) == raw
    subprocess.check_call(['git', '-C', str(repo), 'commit', '-m', 'Launch frozen-G face-conditioned boundary decoder after real sanity'])
    heads.append(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip())
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
subprocess.check_call(['git', '-C', str(repos[0]), '-c', 'http.version=HTTP/1.1', 'push', 'origin', 'HEAD:main'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == heads[0]
digest = hashlib.sha256(new).hexdigest()
guard.write_bytes(guard_raw.replace(previous['handoff_sha256'].encode(), digest.encode()))
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), section='20.376.47',
    heads=heads, github_main=heads[0], handoff_sha256=digest, handoff_bytes=len(new),
    four_local_and_remote_equal=True, exact_committed_payloads=True, payload_count=len(payloads),
    native_preflight_complete=True, real_probe_updates=2, fit_started=True,
    formal_terminal_accuracy_available=False, retained_metric_best='plain_distribution_5616_4506', goal_achieved=False)
(local / 'launch_publication.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
with (workspace / 'memory/2026-10-04.md').open('a', encoding='utf-8') as stream:
    stream.write(f'\nPVGround {record["time_cst"]} face-conditioned real two-step sanity and bounded frozenG fit published {heads[0]}; goalACTIVE_UNMET, no new formal result. ParentG/current4506 best protected.\n')
print(json.dumps(record), flush=True)
