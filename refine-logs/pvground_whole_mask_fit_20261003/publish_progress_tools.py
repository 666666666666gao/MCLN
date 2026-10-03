"""Sync the actual first progress and verified read-only terminal tools."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import paramiko


local = Path(__file__).parent
workspace = Path('C:/Users/gb')
assert not (local/'progress_tools_publication.json').exists()
previous = json.loads((local/'launch_publication.json').read_bytes())
observation = json.loads((local/'observation_01.json').read_bytes())
check = json.loads((local/'terminal_tools_source_check.json').read_bytes())
state = json.loads((local/'active_formal_continuation_state.json').read_bytes())
assert observation['status']['stage'] == 'local_range/train' and observation['controller_alive']
assert state['remote_observation_available'] and not state['formal_result_available']
assert check['status'] == 'SOURCE_READY_NOT_CURRENT_TERMINAL_RESULT'
for name, digest in check['tool_sha256'].items():
    assert hashlib.sha256((local/name).read_bytes()).hexdigest() == digest
repos = [workspace/'.codex_mcln_g0_20260905', workspace/'.codex_pvground_cs_20261002',
    workspace/'.codex_mcln_v99_internal_20260928']
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies = [repo/doc for repo in repos] + [workspace/'Desktop/document'/Path(doc).name]
old = copies[0].read_bytes()
assert hashlib.sha256(old).hexdigest() == previous['handoff_sha256']
assert all(path.read_bytes() == old for path in copies)
for repo, head in zip(repos, previous['heads']):
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip() == head
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
guard = workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py'
guard_raw = guard.read_bytes()
assert guard_raw.count(previous['handoff_sha256'].encode()) == 1
text = old.decode('utf-8')
before = ('两组暂估6～8小时，不是新吞吐实测；唯一只读观察器首查40分钟，随后按已完成更新／完整eval估计结束前2～3分钟查看，'
    '未完成再240秒查，不启动第二份训练或观察器。')
after = ('启动时两组按历史暂估6～8小时，不是新吞吐实测。唯一只读观察器5205于'
    +observation['time_cst']+'首次实际读取：controller403326／child403332仍运行，local_range/train已320／3723更新，'
    'cumulative907.897465秒，约2.837180秒／更新；数据盘2621943808字节，未到512步保存点、无本轮owned权重。'
    'completed[]指包含初始评估／训练／终态评估的整个train阶段还未闭环，不是零更新。'
    '仅当前local样本遍历推算约2026-10-04T00:35:53+08:00结束，其后仍有6887终态评估、9508正式评估以及整个whole组；'
    '不把这个估计当作整对照完成时间。原观察器已自行改为00:32:51左右检查，接近预计训练遍历结束前3分钟；'
    '未完成按240秒检查，不新开第二观察器或训练。这个进度是21:54已发布快照，不假称后来服务器实时状态。')
assert text.count(before) == 1
text = text.replace(before, after)
marker = '主目标仍须比较本轮来源控制和原G4495严格命中；'
assert text.count(marker) == 1
paragraph = ('终态只读收取／分析工具collect_terminal.py、analyze_terminal.py已准备；当前未执行本轮终态收取或分析。'
    '工具通过Py3.7语法与一个实际历史9508行来源校验：旧tail_fused同Query严格4412→4406、修复15／破坏21，'
    'bbs／bbf所选框CPU阈值重算差0。该记录只证明工具可读取已完成历史证据，不是本轮新成绩或独立新方法审查。'
    '真实controller完整结束后才收取文本／逐行记录／清理回执，不下载权重；核对实际同预算输入、fit顺序、原生REC／Mask、'
    '配对修复破坏、同Query精修和Full256候选缺失／未选中，再对实际路径执行新鲜experiment-audit。'
    '不独立重建全部原始Mask／候选框或声称重放优化器。活动runner／spec／模型／监督／保存规则未改。\n\n')
text = text.replace(marker, paragraph+marker)
new = text.encode('utf-8')
prefix = 'refine-logs/pvground_whole_mask_fit_20261003/'
names = ['collect_terminal.py', 'analyze_terminal.py', 'verify_terminal_tools.py',
    'terminal_tools_source_check.json', 'observation_01.json', 'record_analysis_preparation.py',
    'record_first_observation.py', 'record_launched_continuation.py',
    'active_formal_continuation_state.json', 'publish_progress_tools.py']
payloads = {prefix+name: (local/name).read_bytes() for name in names}
assert all(not (repo/relative).exists() for repo in repos[:2] for relative in payloads)
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
remote = '/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(remote+'/'+doc, 'rb') as stream:
    assert stream.read() == old
folders = sorted({str(Path(remote+'/'+relative).parent).replace('\\', '/') for relative in payloads})
_, stdout, stderr = client.exec_command(shlex.join(['mkdir', '-p', '--', *folders]), timeout=60)
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
for relative, raw in payloads.items():
    for repo in repos[:2]:
        target = repo/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    with sftp.open(remote+'/'+relative, 'wx') as stream:
        stream.write(raw)
    with sftp.open(remote+'/'+relative, 'rb') as stream:
        assert stream.read() == raw
for path in copies:
    path.write_bytes(new)
with sftp.open(remote+'/'+doc, 'wb') as stream:
    stream.write(new)
assert all(path.read_bytes() == new for path in copies)
with sftp.open(remote+'/'+doc, 'rb') as stream:
    assert stream.read() == new
sftp.close()
client.close()
stamp = datetime.datetime.now().astimezone().isoformat()
heads = []
for index, repo in enumerate(repos):
    if index < 2:
        with (repo/'MANIFEST.md').open('a', encoding='utf-8') as stream:
            stream.write('\n- '+stamp+' /analyze-results: first actual range-source progress and source-verified read-only terminal tools; no new formal result.\n')
    stage = [doc] + (['MANIFEST.md', *payloads] if index < 2 else [])
    status = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all']).decode()
    assert {line[3:] for line in status.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage])
    subprocess.check_call(['git', '-C', str(repo), '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol',
        'diff', '--cached', '--check', '--', *stage])
    for relative, raw in (payloads.items() if index < 2 else []):
        assert subprocess.check_output(['git', '-C', str(repo), 'show', ':'+relative]) == raw
    subprocess.check_call(['git', '-C', str(repo), 'commit', '-m',
        'Record measured range-source progress and prepare verified terminal analysis'])
    heads.append(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip())
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
subprocess.check_call(['git', '-C', str(repos[0]), 'push', 'origin', 'HEAD:main'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == heads[0]
for relative in (doc, *payloads):
    assert subprocess.check_output(['git', '-C', str(repos[0]), 'rev-parse', 'HEAD:'+relative]) == subprocess.check_output([
        'git', '-C', str(repos[1]), 'rev-parse', 'HEAD:'+relative])
digest = hashlib.sha256(new).hexdigest()
guard.write_bytes(guard_raw.replace(previous['handoff_sha256'].encode(), digest.encode()))
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), section='20.376.37 updated in place',
    heads=heads, github_main=heads[0], handoff_sha256=digest, handoff_bytes=len(new),
    four_local_and_remote_equal=True, exact_committed_payloads=True, payload_count=len(payloads),
    actual_observation_cst=observation['time_cst'], observed_updates=320, formal_result_available=False,
    active_source_changed=False, new_weights_downloaded=0, method_promoted=False, goal_achieved=False)
(local/'progress_tools_publication.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps(record), flush=True)
