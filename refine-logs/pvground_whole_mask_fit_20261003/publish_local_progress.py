"""Publish the actual closed local-arm receipt; leave the active pair untouched."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import paramiko

local = Path(__file__).parent
assert not (local/'local_progress_publication.json').exists()
previous = json.loads((local/'progress_tools_publication.json').read_bytes())
observation = json.loads((local/'observation_07.json').read_bytes())
status = observation['status']
retention = status['last_weight_retention']
assert observation['controller_alive'] and observation['controller_exit'] is None
assert status['stage'] == 'whole_range/train'
assert [(r['arm'], r['mode']) for r in status['completed']] == [
    ('local_range', 'train'), ('local_range', 'formal')]
assert retention['cpu_box_threshold_recount']['bbs'] == dict(
    rec_hits25=5603, rec_hits50=4428, cpu_threshold_changes=0)
assert status['retained_best']['name'] == 'original_g'
assert retention['parent_sha256_after'] == status['retained_best']['sha256']
assert not retention['local_weight_archive_created'] and not observation['owned_weights']
assert retention['deleted'] == [dict(
    path='/root/autodl-tmp/pvground_whole_mask_fit_20261003/local_range/terminal.pth',
    bytes=347116945, sha256='313c69e4dff9afa1dd32181e301a0a6012998ac59110c27a538eff2ea0e61b4e')]
workspace = Path('C:/Users/gb')
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
stamp = datetime.datetime.now().astimezone().isoformat()
section = '''

## 20.376.38 local来源控制正式完成，非最佳终点及时退役（%s）

原controller403326中的local_range/train于2026-10-04T00:44:47.960446+08:00结束，随后formal于01:08:54.419982结束；两阶段均正常完成。训练阶段12753.09秒包含加载、6887初始评估、29778条fit一次／3723更新及6887终态评估，不冒称纯训练或GPU成本；formal阶段1446.46秒。

原唯一观察器5205于01:11:23实际确认local已闭环并进入whole_range/train。local完整9508原生last/bbs为5603／4428，即58.9293%%／46.5713%%，对保护原G5615／4495为−12／−67；bbf为5637／4431，单列、不拼模式最好列。此处是controller实际回执与原观察器快照，整对照逐行收取、配对分析和新鲜终态审查尚未执行；不能据local结果先判断完整109范围来源有效或失败。

controller已核对local正式行SHA5a5703d80ef6b1b48e315a9a6374f91764c3790a2e33d698d4f8b83265897e4a、终点SHA313c69e4dff9afa1dd32181e301a0a6012998ac59110c27a538eff2ea0e61b4e，以及bbs／bbf所选框对root的CPU阈值重算均0变化。01:08:58按用户持续授权仅删除本轮local_range/terminal.pth347116945字节；原G全SHA保持0575dfae333dabdf288a470fc09d8967f9867d1853a5f61d9cfc3cbcf7964522，不生成失败本地权重归档。日志、源码、配置和逐行证据保留，未声称重放模型、全部Mask或优化器。观察时数据盘2600157184字节，本轮owned权重为空；该磁盘值是01:11快照。

whole_range/train于01:08:58.583203实际启动，child421367；源代码、109统计、400614参数头、native＋G损失、优化器／batch／学习率／更新预算／保存协议均未改。两组有效batch8、各29778条一次及3723更新同时固定，不能只按batch变化调整LR而隐去更新次数。当前whole正式9508结果尚无，原G仍为指标最佳、未晋级新方法，ScanRefer5615／4754及独立Nr／Sr目标未完成。继续同一观察器按实际吞吐估计接近阶段结束时检查，不重启训练或另开观察器。
''' % stamp
new = old + section.encode('utf-8')
record = dict(time_cst=stamp, status='LOCAL_FORMAL_CLOSED_WHOLE_ACTIVE',
    evidence='observation_07.json', observation_cst=observation['time_cst'],
    local_bbs_hits=[5603,4428], local_bbf_hits=[5637,4431], rows=9508,
    versus_original_g=[-12,-67], retained_best='original_g_5615_4495',
    local_deleted_bytes=347116945, local_weight_archive_created=False,
    active_source_changed=False, pair_formal_result_available=False,
    pair_terminal_analysis_executed=False, method_promoted=False, goal_achieved=False)
prefix = 'refine-logs/pvground_whole_mask_fit_20261003/'
payloads = {
    prefix+'observation_07.json': (local/'observation_07.json').read_bytes(),
    prefix+'local_formal_observed.json': (json.dumps(record, indent=2)+'\n').encode(),
    prefix+'publish_local_progress.py': Path(__file__).read_bytes(),
}
assert all(not (repo/relative).exists() for repo in repos[:2] for relative in payloads)
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
    password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
remote = '/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(remote+'/'+doc, 'rb') as stream:
    assert stream.read() == old
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
heads = []
for index, repo in enumerate(repos):
    if index < 2:
        with (repo/'MANIFEST.md').open('a', encoding='utf-8') as stream:
            stream.write('\n- '+stamp+' actual local formal5603/4428 closed; verified nonbest endpoint retired, whole remains active.\n')
    stage = [doc] + (['MANIFEST.md', *payloads] if index < 2 else [])
    changed = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage])
    subprocess.check_call(['git', '-C', str(repo), '-c',
        'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol',
        'diff', '--cached', '--check', '--', *stage])
    for relative, raw in (payloads.items() if index < 2 else []):
        assert subprocess.check_output(['git', '-C', str(repo), 'show', ':'+relative]) == raw
    subprocess.check_call(['git', '-C', str(repo), 'commit', '-m',
        'Record local range formal result and verified weight retention'])
    heads.append(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip())
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
subprocess.check_call(['git', '-C', str(repos[0]), 'push', 'origin', 'HEAD:main'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == heads[0]
for relative in (doc, *payloads):
    assert subprocess.check_output(['git', '-C', str(repos[0]), 'rev-parse', 'HEAD:'+relative]) == subprocess.check_output([
        'git', '-C', str(repos[1]), 'rev-parse', 'HEAD:'+relative])
digest = hashlib.sha256(new).hexdigest()
guard.write_bytes(guard_raw.replace(previous['handoff_sha256'].encode(), digest.encode()))
record.update(time_cst=datetime.datetime.now().astimezone().isoformat(), section='20.376.38',
    heads=heads, github_main=heads[0], handoff_sha256=digest, handoff_bytes=len(new),
    four_local_and_remote_equal=True, exact_committed_payloads=True,
    payload_count=len(payloads), weights_downloaded=0)
(local/'local_progress_publication.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps(record), flush=True)
