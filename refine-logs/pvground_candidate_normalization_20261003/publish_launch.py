"""Append actual normalization launch evidence to the one project handoff."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import paramiko

workspace = Path(r'C:\Users\gb')
local = Path(__file__).parent
previous = json.loads((local.parent / 'pvground_candidate_consistency_20261003/complete_publication.json').read_bytes())
assert not (local / 'launch_publication.json').exists()
review = json.loads((local / 'EXPERIMENT_CODE_REVIEW.json').read_bytes())
assert review['verdict'] in ('PASS', 'WARN') and review['blocking_findings'] == []
cpu = json.loads((local / 'cpu_test.json').read_bytes())
preflight = json.loads((local / 'preflight.json').read_bytes())
launch = json.loads((local / 'launch.json').read_bytes())
cleanup = json.loads((local / 'local_nonbest_cleanup_receipt.json').read_bytes())
assert cleanup['deleted_count'] == 2 and cleanup['archive_sha256_reverified_before_deletion']
assert cpu['status'] == preflight['status'] == 'pass'
observations = sorted(local.glob('observation_*.json'))
assert observations
observed = json.loads(observations[-1].read_bytes())
assert observed['controller_alive'] and observed['status']['status'] == 'running'
assert observed['status']['stage'] == 'normalized/train'
assert preflight['capacity']['expanded_count_normalization'] and preflight['optimizer_steps'] == 2
repos = [workspace / '.codex_mcln_g0_20260905', workspace / '.codex_pvground_cs_20261002',
         workspace / '.codex_mcln_v99_internal_20260928']
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies = [repo / doc for repo in repos] + [workspace / 'Desktop/document' / Path(doc).name]
old = copies[0].read_bytes()
assert hashlib.sha256(old).hexdigest() == previous['handoff_sha256']
assert all(path.read_bytes() == old for path in copies)
assert b'### 20.376.29 ' not in old
for repo, head in zip(repos, previous['heads']):
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip() == head
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])

now = datetime.datetime.now().astimezone().isoformat()
snapshot = local / 'code_review_inputs'
snapshot.mkdir()
for name in ('EXPERIMENT_PLAN.md', 'EXPERIMENT_TRACKER.md'):
    (snapshot / name).write_bytes((local / name).read_bytes())
plan = (local / 'EXPERIMENT_PLAN.md').read_text(encoding='utf-8')
plan = plan.replace('状态：实现准备完成；尚未代码终审、真实CPU/GPU预检或正式训练。',
    '状态：代码审查及原生CPU、真实batch8两步预检完成；正式normalized/train已启动，无新精度结果。')
(local / 'EXPERIMENT_PLAN.md').write_bytes(plan.encode('utf-8'))
(local / 'EXPERIMENT_TRACKER.md').write_text(
    '# 归一化控制执行表\n\n| 阶段 | 状态 | 证据 |\n|---|---|---|\n'
    '| 代码实现 | REVIEWED | 复用runner；仅扩展最后层对比项分母改变；审查见EXPERIMENT_CODE_REVIEW |\n'
    '| CPU原生公式核验 | PASS | cpu_test.json；A=0损失/梯度；A>0直接调用；其他GT保护 |\n'
    '| GPU batch8预检 | PASS | preflight.json；2步、有限梯度、完整模型/优化器恢复与显存 |\n'
    '| normalized训练 | RUNNING | launch.json；原G/fresh AdamW；3723步；29778条一次 |\n'
    '| 正式9508验证 | NOT_STARTED | 由同一控制器在训练后执行，尚无新结果 |\n', encoding='utf-8')

capacity = preflight['capacity']
appendix = ('\n\n### 20.376.29 同对应集合的归一化控制：实际预检通过并启动单组训练\n\n'
    f'记录时间：{now}。上一轮原G控制5588／4423、一致性5596／4457的完整结果和限制维持§20.376.28；'
    '本轮不重跑这两组，只新增同一G资格集合的归一化组。原G5615／4495仍为内部最强权重，ScanRefer目标5615／4754未实现。\n\n'
    '实现保留上一轮run.py、G资格与CE模块的文件字节；官方父权重、原G起点、原生数据和评估接口、模型结构及全部256候选均不变。'
    '唯一损失变量：扩展后最后层对比项由L_expanded/N改为L_expanded/(N+A)，先移除旧最后层项再按0.5/7替换，未叠加两份完整loss。'
    '该变化缩放整个扩展对比项，包括原匹配、背景及新增对应，不是纯标签控制，也不声称只降低新增候选权重；无新增参数或独立评分。\n\n'
    f'代码审查：fresh gpt-6-astra/max，{review["verdict"]}，blocking_findings为空；same-family/provisional，具体范围和限制见审查报告。'
    '原生CPU方法核验已真实通过：A=0损失和梯度一致，A>0与直接扩展计数调用及梯度等价，其他GT匹配保留，资格不直接向几何反传。\n\n'
    f'实际A100 batch8预检完成2次优化更新，随后严格重载模型与Adam状态；预检从原G初始化，正式训练另从原G和fresh AdamW开始，预检步不计入正式预算。'
    f'预检原分母{capacity["contrastive_native_denominator"]}、扩展分母{capacity["contrastive_expanded_denominator"]}，'
    f'新增{capacity["contrastive_reassigned_queries"]}个对应，原最后层对比项{capacity["contrastive_native"]:.8f}，'
    f'扩展归一化后{capacity["contrastive_replaced"]:.8f}。有限梯度及无直接几何梯度见实际回执，负值不作为成功或错误依据。'
    f'峰值分配{preflight["peak_allocated_bytes"]}字节，实测模型与优化器序列化{preflight["serialization_bytes"]}字节。\n\n'
    f'启动时间：{launch["time_cst"]}；远端控制器{launch["process"].split()[0]}，screen {launch["screen"]}。'
    f'最新只读观察{observed["observed_cst"]}确认原控制器存活、normalized/train实际运行。'
    '单GPU串行train→formal9508，seed2027、batch8、29778条fit各一次／3723更新、LR核心与骨干1e-5、WD5e-4、clip0.1保持。'
    f'启动时输出目录配额余量{launch["resources"]["directory_free_bytes"]}字节，保存空间按实测序列化与原子替换检查；本轮未删除远端文件，原G及依赖继续保护。\n\n'
    f'用户再次要求及时清理自有无用权重、仅保留指标最好权重。本轮确认上一轮控制和一致性终点均低于原G后，'
    f'核对全SHA并仅清理两份本地归档terminal，共{int(cleanup["deleted_bytes"])}字节；C盘余量'
    f'{cleanup["local_c_free_before"]}→{cleanup["local_c_free_after"]}字节。'
    '它们的远端终点已在§20.376.28清理，本次保留全部日志、源码、逐行结果和SHA凭证；旧归档回执是历史记录，不代表两份负终点现在仍存在。'
    '原G、必要官方父权重、历史V99最佳及一个活动恢复点保护；失败终点不长期积累本地归档权重。\n\n'
    f'预计结束{launch["estimated_finish_cst"]}，按上一轮实测train与formal耗时估计，非实际完成时间；接近预计结束时按180～300秒只读轮询。'
    '只保留一个活动恢复点，终态如非最佳，在保存完整结果和核对权重凭证后按既有授权清理，不长期积累无用副本。当前没有新精度结果；CPU/GPU预检通过不等于Acc@0.50增益。'
    '比较仍须实际核验跨进程起点差异，报告相对控制、未归一化组及原G的增减，不将不同模型的固定框空间混作可部署收益。\n\n'
    '本轮不同时加入教师、P3、质量头或边界分布。后续结构仍研究完整预测Mask范围与实际超点成员坐标的全局—局部支撑，不重复固定7×16近邻的概率拼接。'
    '低排名／未匹配候选保留用于学习与比较，几何资格不等于身份真值。Nr3D／Sr3D新方法尚未训练，泛化与50%目标仍待验证。\n')
new = old + appendix.encode('utf-8')
prefix = 'refine-logs/pvground_candidate_normalization_20261003/'
names = ['run.py', 'controller.py', 'cpu_test.py', 'run_preflight_authorized.py', 'launch_authorized.py',
    'observe_authorized.py', 'wait_completion_authorized.py', 'prepare.py', 'publish_launch.py',
    'implementation_diff.patch', 'pvground_candidate_consistency.py',
    'pvground_semantic_assignment.py', 'pvground_task_observation_query.py', 'pvground_observation_query.py',
    'pvground_source_query.py', 'preflight_spec.json', 'normalized_spec.json', 'source_preparation.json',
    'EXPERIMENT_PLAN.md', 'EXPERIMENT_TRACKER.md', 'research_contract.md', 'EXPERIMENT_CODE_REVIEW.md',
    'EXPERIMENT_CODE_REVIEW.json', 'cpu_test.json', 'cpu.log', 'cpu.exit', 'preflight.json', 'preflight.log',
    'preflight.exit', 'preflight_intake.json', 'preflight_resource_check.json', 'load.json', 'imports.json', 'launch.json',
    'local_nonbest_cleanup_receipt.json']
payloads = {prefix + name: local / name for name in names}
payloads.update({prefix + path.name: path for path in observations})
payloads.update({prefix + 'code_review_inputs/' + path.name: path for path in snapshot.iterdir()})
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
remote = '/home/gb/new butd/butd_detr-main/MCLN-main'
sftp = client.open_sftp()

def remote_bytes(path):
    with sftp.open(path, 'rb') as stream:
        return stream.read()

assert remote_bytes(remote + '/' + doc) == old
folder = remote + '/' + prefix.rstrip('/')
_, stdout, stderr = client.exec_command(shlex.join(['mkdir', '-p', '--', folder, folder + '/code_review_inputs']), timeout=60)
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
for relative, source in payloads.items():
    raw = source.read_bytes()
    for repo in repos[:2]:
        target = repo / relative
        assert not target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    with sftp.open(remote + '/' + relative, 'wx') as stream:
        stream.write(raw)
    assert remote_bytes(remote + '/' + relative) == raw
for path in copies:
    path.write_bytes(new)
with sftp.open(remote + '/' + doc, 'wb') as stream:
    stream.write(new)
assert all(path.read_bytes() == new for path in copies) and remote_bytes(remote + '/' + doc) == new
sftp.close()
client.close()
heads = []
for index, repo in enumerate(repos):
    if index < 2:
        with (repo / 'MANIFEST.md').open('a', encoding='utf-8') as stream:
            stream.write('\n- ' + now + ' /experiment-bridge: ' + prefix + ' expanded-count normalization; native sanity passed; one-arm training launched.\n')
    stage = [doc] + (['MANIFEST.md', *payloads] if index < 2 else [])
    status = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all']).decode()
    assert {line[3:] for line in status.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage])
    subprocess.check_call(['git', '-C', str(repo), 'diff', '--cached', '--check', '--', *stage])
    for relative, source in (payloads.items() if index < 2 else []):
        assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + relative]) == source.read_bytes()
    subprocess.check_call(['git', '-C', str(repo), 'commit', '-m', 'Launch same-correspondence normalization control after native sanity'])
    heads.append(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip())
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
subprocess.check_call(['git', '-C', str(repos[0]), 'push', 'origin', 'HEAD:main'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == heads[0]
for relative in (doc, *payloads):
    assert subprocess.check_output(['git', '-C', str(repos[0]), 'rev-parse', 'HEAD:' + relative]) == subprocess.check_output(['git', '-C', str(repos[1]), 'rev-parse', 'HEAD:' + relative])
digest = hashlib.sha256(new).hexdigest()
guard = workspace / '.codex/tmp/sync_cs_handoff_remote_20260923.py'
code = guard.read_bytes()
assert code.count(previous['handoff_sha256'].encode()) == 1
guard.write_bytes(code.replace(previous['handoff_sha256'].encode(), digest.encode()))
receipt = dict(section='20.376.29', time_cst=datetime.datetime.now().astimezone().isoformat(),
    github_main=heads[0], heads=heads, handoff_sha256=digest, handoff_bytes=len(new),
    prior_prefix_unchanged=True, four_local_and_remote_equal=True, exact_committed_evidence=True,
    payload_count=len(payloads), GPU_preflight_pass=True, normalized_training_started=True,
    normalized_formal_results_available=False, scanrefer_target_pass=False,
    retained_best='original_g', review_independence='same-family', acceptance_status='provisional')
(local / 'launch_publication.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps(receipt))
