"""Publish the actual closed engineering checks and launched bounded comparison."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import paramiko

local = Path(__file__).resolve().parent
assert not (local / 'launch_publication.json').exists()
workspace = Path('C:/Users/gb')
previous = json.loads((workspace / '.codex/tmp/pvground_range_head_only_20261004/terminal_publication.json').read_bytes())
review = json.loads((local / 'EXPERIMENT_CODE_REVIEW.json').read_bytes())
assert review['verdict'] == 'PASS' and not review['blocking_findings']
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
engineering = json.loads((local / 'PREFLIGHT_ANALYSIS.json').read_bytes())
intake = json.loads((local / 'complete_preflight/INTAKE.json').read_bytes())
assert engineering['status'] == 'ACTUAL_ENGINEERING_PASS'
assert intake['status']['status'] == 'complete' and intake['controller_exit'] == 0
launch = json.loads((local / 'launch.json').read_bytes())
assert launch['formal_training_started'] and not launch['formal_result_available']
wait = json.loads((local / 'wait.json').read_bytes())
assert wait['status'] == 'waiting' and not wait['observations']
repos = [workspace / name for name in ('.codex_mcln_g0_20260905',
    '.codex_pvground_cs_20261002', '.codex_mcln_v99_internal_20260928')]
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies = [repo / doc for repo in repos] + [workspace / 'Desktop/document' / Path(doc).name]
old = copies[0].read_bytes()
assert hashlib.sha256(old).hexdigest() == previous['handoff_sha256']
assert all(path.read_bytes() == old for path in copies)
for repo, head in zip(repos, previous['heads']):
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip() == head
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == previous['heads'][0]
guard = workspace / '.codex/tmp/sync_cs_handoff_remote_20260923.py'
guard_raw = guard.read_bytes()
assert guard_raw.count(previous['handoff_sha256'].encode()) == 1
stamp = datetime.datetime.now().astimezone().isoformat()
residual = engineering['arms']['residual']
distribution = engineering['arms']['distribution']
section = f'''

## 20.376.43 固定原G的六面边界分布对照：真实预检完成、正式配对启动（{stamp}）

§20.376.42完整范围head-only对照已形成负结果：whole严格4456比local4447多9，但仍比原G4495少39，不能将完整范围统计拼接宣称为验证有效的主模块。下一项从受保护原G重新加载，未复用退役的范围终点；只检验相同whole范围输入下的普通残差与六面分布表达。不是再次扩大P2或改变对比项分母。

### 本轮实际变化与预算

两组输入均为1302维：Query288＋固定7×16局部聚合896＋粗尺寸3＋场景归一化粗框6＋完整预测融合Mask范围109。全部256候选保留，G核心参数与持久buffer固定、eval；每组只训练新增头10项张量。residual为{residual['head_parameters']}参数、普通中心／尺寸残差；distribution为{distribution['head_parameters']}参数、六面各33个非均匀偏移位置（±4范围、近零更细），增加独立DFL均值项／7。参数量、参数化和监督同时变化，不能声称是纯DFL或纯结构的独立因果增量。这是D-FINE式边界表示的直接三维适配控制，尚不是面Query逐方向读取支撑的完整目标模块。

最终匹配使用原生7层criterion中的last真实Hungarian对应，保留全部已匹配GT职责；额外分布项对matched boxes×6面求均值，新增1/7系数与G／对比项分母分开。原生回归与GIoU读取唯一实际精修框。两组都在forward对参考与最终尺寸采用1e-6下限；预检原粗尺寸共有{residual['initial_raw_size_below_floor']}个元素低于下限，属于batch8×256×3全部候选元素统计，不能冒称已选答案错误数，也不声称对未经处理的负粗尺寸逐位一致。

每组固定seed2027、物理／有效batch8、29778条fit各一次、3723次更新（尾batch2）、LR1e-5、WD5e-4、clip0.1、fresh AdamW。初始／终点6887为作者预训练见过场景的模块留出；正式9508为长期开发验证。第二组实际loader行顺序对第一组检查，跨进程起点数值差异如实记录。没有教师、最终质量回写、P2、扩展多正例、双源排名或GT推理门控；bbs评分和原生Mask仍保留。

### 实际源码审查与GPU预检

新鲜源码审查PASS、0阻断、0未解决warning；实际task /root/pvg_boundary_distribution_source_review，gpt-6-astra／max／forknone，same-family／provisional，未独立验证backend SKU。审查覆盖48份当前文件；预检不由源码审查代替。唯一review建议是原始边界目标在clamp前显式检查finite，已用两行最小修正并重新审查；没有更改损失、匹配、输入或预算。control_tools_preparation中早期spec SHA保留为历史准备记录，后续有限目标修正及四份实际spec明确当前身份。

真实GPU预检controller471105于{intake['status']['finished_cst']}完成、exit0；两组各batch8、2次可丢弃更新、4次完整native forward，总4更新／8前向。零头与同缓存、共同尺寸协议的原G结果精确一致；原G状态不变；第二步10项头张量都有非零梯度；新增delta与AdamW keys／moments／steps／groups实际内存恢复一致。预检没有保存磁盘权重，也没有正式精度评估。

residual的原生定位输出梯度为{residual['native_output_gradient']}；distribution为{distribution['native_output_gradient']}，单独边界项输出梯度为{distribution['edge_alone_output_gradient']}。分布项loss为{distribution['boundary_loss']}，每步实际{distribution['matched_faces']}面目标、范围外{distribution['targets_outside']}；这些只有一个真实batch，不替代全训练分布检查。分布组峰值allocated {distribution['peak_allocated_bytes']}字节、reserved {distribution['peak_reserved_bytes']}字节，属于固定上游的本次预检，不能直接据此扩大batch或线性调整LR。实际serialized delta＋optimizer为{distribution['serialization_bytes']}字节。

### 实际正式启动与保留策略

两组已于{launch['time_cst']}由controller471894、screen {launch['screen']}串行启动：residual/train→residual/formal→distribution/train→distribution/formal，远端{launch['root']}。这是启动见证；此发布没有任何一组正式终态或9508精度，不能写成边界分布已经涨点。

实际启动前数据盘可用{launch['resources']['free_bytes']}字节，所需小delta与临时副本＋日志预算{launch['resources']['required_bytes']}字节。原G保留为指标最佳；每组必要小恢复终点仅保留到真实恢复、9508评估、CPU所选框阈值重算、父SHA和终点SHA核对完成，然后及时退役自有非最佳权重，不再创建失败本地归档。官方PV及必要V99链保护。§42已实际删除两份自有非最佳共9842058字节，不是只计划清理。

唯一只读观察器46724按上一完整固定G阶段约6495秒估计，首次计划{wait['next_scheduled_cst']}；后续按实际进度靠近阶段结束检查，最低间隔240秒，避免不断SSH查询。预算含义同时记录样本遍历和更新次数，没有只按batch缩放LR。

证据位于refine-logs/pvground_boundary_distribution_20261004：实际源码、spec、计划／contract、固定官方参考、48文件review及真实请求／响应、complete_preflight原始接线证据、PREFLIGHT_ANALYSIS和launch。当前完整ScanRefer目标5615／4754及后续Nr3D／Sr3D目标仍未达到；三模块贡献均须后续直接实验建立。
'''
new = old + section.encode('utf-8')
prefix = 'refine-logs/pvground_boundary_distribution_20261004/'
payloads = {}
for directory in ('complete_preflight', 'references', 'idea-stage'):
    for path in sorted((local / directory).rglob('*')):
        if path.is_file():
            payloads[prefix + path.relative_to(local).as_posix()] = path.read_bytes()
trace = local / '.aris/traces/experiment-bridge/2026-10-04_boundary_source_run01'
for path in sorted(trace.iterdir()):
    if path.is_file():
        payloads[prefix + 'code_review_trace/' + path.name] = path.read_bytes()
for path in sorted(local.iterdir()):
    if path.is_file() and path.suffix in ('.py', '.json', '.md', '.txt') and path.name not in (
            'active_continuation_state.json', 'launch_publication.json'):
        payloads[prefix + path.name] = path.read_bytes()
assert all(not (repo / relative).exists() for repo in repos[:2] for relative in payloads)
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
    password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
remote = '/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(remote + '/' + doc, 'rb') as stream:
    assert stream.read() == old
directories = sorted({str(Path(relative).parent).replace('\\', '/') for relative in payloads},
    key=lambda item: (item.count('/'), item))
created = {'refine-logs'}
for directory in directories:
    parts = directory.split('/')
    for count in range(1, len(parts) + 1):
        current = '/'.join(parts[:count])
        if current not in created:
            sftp.mkdir(remote + '/' + current)
            created.add(current)
for relative, raw in payloads.items():
    for repo in repos[:2]:
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    with sftp.open(remote + '/' + relative, 'wx') as stream:
        stream.write(raw)
    with sftp.open(remote + '/' + relative, 'rb') as stream:
        assert stream.read() == raw
for path in copies:
    path.write_bytes(new)
with sftp.open(remote + '/' + doc, 'wb') as stream:
    stream.write(new)
with sftp.open(remote + '/' + doc, 'rb') as stream:
    assert stream.read() == new
sftp.close()
client.close()
heads = []
for index, repo in enumerate(repos):
    if index < 2:
        with (repo / 'MANIFEST.md').open('a', encoding='utf-8') as stream:
            stream.write('\n- ' + stamp + ' frozen original-G boundary residual/distribution: actual two-arm GPU preflight pass; bounded serial comparison launched, no formal metrics yet.\n')
    stage = [doc] + (['MANIFEST.md', *payloads] if index < 2 else [])
    changed = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage])
    owned_check = [doc, 'MANIFEST.md'] + [p for p in payloads if
        p.startswith(prefix) and '/' not in p[len(prefix):]] if index < 2 else [doc]
    subprocess.check_call(['git', '-C', str(repo), '-c',
        'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol',
        'diff', '--cached', '--check', '--', *owned_check])
    for relative, raw in (payloads.items() if index < 2 else []):
        assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + relative]) == raw
    subprocess.check_call(['git', '-C', str(repo), 'commit', '-m',
        'Record real boundary preflight and launch fixed G representation comparison'])
    heads.append(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip())
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
subprocess.check_call(['git', '-C', str(repos[0]), '-c', 'http.version=HTTP/1.1',
    'push', 'origin', 'HEAD:main'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == heads[0]
assert all(path.read_bytes() == new for path in copies)
for relative in (doc, *payloads):
    assert subprocess.check_output(['git', '-C', str(repos[0]), 'rev-parse', 'HEAD:' + relative]) == subprocess.check_output(['git', '-C', str(repos[1]), 'rev-parse', 'HEAD:' + relative])
digest = hashlib.sha256(new).hexdigest()
guard.write_bytes(guard_raw.replace(previous['handoff_sha256'].encode(), digest.encode()))
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), section='20.376.43',
    heads=heads, github_main=heads[0], handoff_sha256=digest, handoff_bytes=len(new),
    four_local_and_remote_equal=True, payload_count=len(payloads), exact_committed_payloads=True,
    source_review='PASS/0blocking/same-family/provisional', actual_engineering_pass=True,
    native_model_updates=4, fit_started=True, formal_metrics_available=False,
    retained_best='original_g', goal_achieved=False)
(local / 'launch_publication.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
state_path = local / 'active_continuation_state.json'
state = json.loads(state_path.read_bytes())
state.update(latest_actual_publication=record, sole_local_observer_session=46724,
    formal_launcher_execution=dict(session_id=51531, status='CLOSED_EXIT0'),
    goal_status='ACTIVE_UNMET', formal_metrics_available=False)
state_path.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
with (workspace / 'memory/2026-10-04.md').open('a', encoding='utf-8') as stream:
    stream.write('\nPV-Ground ' + record['time_cst'] + ' actual §20.376.43 pushed ' + heads[0] +
        '; two frozen-G boundary probes passed; controller471894 launched ' + launch['time_cst'] +
        ', sole observer46724 scheduled13:36; no formal accuracy yet, originalG protected; goalACTIVE_UNMET.\n')
print(json.dumps(record), flush=True)
