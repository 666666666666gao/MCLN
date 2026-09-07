import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
archive_name = 'refine-logs/scanrefer_mask_geometry_pair_20260907_v1'
archive = repo / archive_name
queue_name = 'refine-logs/scanrefer_mask_geometry_posttraining_20260907_v1'
master_name = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
tracker_name = 'refine-logs/EXPERIMENT_TRACKER.md'
report_name = 'docs/SCANREFER_MASK_GEOMETRY_TERMINAL_RESULT_2026-09-07.md'
master = repo / master_name
desktop = Path('C:/Users/gb/Desktop/document') / master.name
old = master.read_bytes()
assert hashlib.sha256(old).hexdigest() == '687b00a90500088a25c4072425a4165a21aa8380f65b7121b7fd9f2692f9febd'
assert desktop.read_bytes() == old
receipt = json.loads((archive / 'receipt.json').read_bytes())
audit = json.loads((archive / 'independent_audit.json').read_bytes())
analysis = json.loads((archive / 'surrogate_transfer_analysis.json').read_bytes())
assert receipt['status'] == 'complete' and receipt['steps_per_arm'] == 2482
assert audit['status'] == 'pass' and audit['decision'] == 'seal_fixed_configuration'
assert not audit['eligible_for_fixed_terminal_formal_evaluation']
assert analysis['comparisons']['candidate_vs_control']['all']['soft_repairs_at_050']['repair']['rows'] == 328
for name in ['baseline_rows.json', 'fit_point_batches.json', 'terminal_rows.json']:
    assert gzip.decompress((archive / (name + '.gz')).read_bytes()) == (archive / name).read_bytes()


def git(*args):
    return subprocess.check_output(['git'] + list(args), cwd=str(repo))


assert git('rev-parse', 'HEAD').decode().strip() == '2e03888662917384fc4b8bce83c5e36ffef50596'
assert not git('diff', '--cached', '--name-only').strip()
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
addition = '''

### 20.140 Mask几何GT固定配对终态与软硬输出分解（TIME）

两臂各2482次更新、29778条fit及唯一6887条模块留出完整结束。22:03:55训练receipt落盘，22:04:11独立审计执行完整性PASS、方法质量FAIL；22:06:04接续队列controller退出0并封存。22:15:51实际SSH确认队列进程不在、正式目录不存在。当前不是等待状态，没有9508正式评估、Nr/Sr启动或控制替代晋级。

同一起点完整V99 REC6684/6426、原生6572/5955；GT控制终点完整6679/6442、原生6560/5994；几何候选完整6677/6448、原生6559/5990。候选系统相对起点−7/+22、相对控制−2/+6，@.25未过固定不退化门。候选原生相对控制−1/−4，不能把相对起点严格阈值+35全部记作辅助贡献。Mask控制6502/6084、mIoU77.82785734%；候选6508/6097、77.78733124%，起点6511/6097、77.81086079%。这些6887条均来自主干已见训练场景，不是正式泛化指标，保护V99正式成绩未变。

新匹配root几何诊断：控制soft6269/5680、hard6566/6048；候选soft6502/5973、hard6575/6053。软@.50相对控制净+293，硬仅+5；平均软IoU+0.02962778、硬+0.00044833，辅助loss1.58474046→1.09376843。两臂6708条匹配Query索引相同的子集也呈现soft+285/hard+8，不能仅由179条匹配变化解释；索引相同不证明语义身份完全相同。

逐行CPU分解发现328条soft@.50修复中，306条（93.2927%）的控制硬框已经过线；整个328条硬框修复1/破坏1，净0，最终REC修复0/破坏1。说明大部分替代几何改善没有创造部署收益。所有行的身份、点SHA和root GT严格相同。没有额外模型前向或正式数据；没有保存完整7变体/逐点logits，不能声称背景概率质量是唯一原因或精确归因某个变体。

独立真实权重核验：84允许参数中82改变，各82份optimizer状态全部2482步；两项norm1无状态/0步。与E71重建1144项实际重载，冻结state/读出/元数据及1201份mesh核验通过。每份delta含optimizer约42.18MB，未下载完整权重。训练receipt SHA563804011c1439e751a6fd65dd93c33ce79565f0ae3229a59dbe42de9460b4ca；audit SHAed72ccbf012c1c6d906dbfdf898e93babc2a03194a7c4eda6a107b2e0a586a9c。fit/terminal大JSON以无损gzip归档Git，原JSON仍保留；分析SHA0921fed57fc39751a3c08130d9251088377d9e46886cd1023cdcaa3dd429afc8，详见SCANREFER_MASK_GEOMETRY_TERMINAL_RESULT_2026-09-07.md。

该固定版本不重跑或扫权重/温度/分位数/LR/epoch，20.136—20.139的原生接口准备不构成晋级，失败终点不导出Nr/Sr。代码核对同时确认已失败joint/frozen_readout已经用root IoU监督合法Parent/Geometry及V99分层排序，不能将相同GT读出loss重新命名为新方案。下一步审计原生最终token聚合分数与匹配/分类/对比监督的实际对应，对照历史残差与读出，寻找新的REC信息或训练责任；暂不继续把Mask替代几何当主要REC路线。Scan正式过保护REC与原Mask底线即尽快Nr/Sr REC，完整三数据集目标不变且未完成。
'''.replace('TIME', now)
new_master = old + addition.encode('utf-8')
tracker = (repo / tracker_name).read_text(encoding='utf-8')
lines = tracker.splitlines()
lines[2] = 'Updated: ' + now + '. Section20.140: Mask geometry pair complete, integrity PASS / fixed REC FAIL; no live training or formal/Nr/Sr continuation.'
for i, line in enumerate(lines):
    if line.startswith('| Mask geometry GT supervision |'):
        lines[i] = '| Mask geometry GT supervision | Complete2482/arm and6887 terminal;integrity PASS, fixed REC FAIL | System6677/6448 vs6684/6426 and6679/6442;sealed,noformal/Nr/Sr |'
    elif line.startswith('| Native Mask geometry complete source |'):
        lines[i] = '| Native Mask geometry complete source | 622-file preparation and6CPU checks retained | Candidate failed;no real endpoint export or Nr/Sr launch;source readiness is not promotion |'
    elif line.startswith('| Native Mask geometry loss integration |'):
        lines[i] = '| Native Mask geometry loss integration | 11CPU checks retained;default off;detection rows excluded | Fixed auxiliary training failed Scan screen;no Nr/Sr activation |'
lines.insert(6, '| Mask geometry surrogate transfer | Saved6887 rows, no new inference;328soft50 repairs,306hard50 already correct | Same328hard net0/deployed−1;full soft+293 versus hard+5;no surrogate gain claim as REC |')
new_tracker = ('\n'.join(lines) + '\n').encode('utf-8')

client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
canonical = '/home/gb/new butd/butd_detr-main/MCLN-main/'
remote_run = '/root/autodl-tmp/mcln_scanrefer_mask_geometry_pair_20260907_v1'
for name in [master_name, tracker_name]:
    with sftp.open(canonical + name, 'rb') as stream:
        actual = stream.read()
    assert actual == (old if name == master_name else git('show', 'HEAD:' + name)), name
for name in ['receipt.json', 'independent_audit.json', 'controller.exit', 'training.exit']:
    with sftp.open(remote_run + '/' + name, 'rb') as stream:
        assert stream.read() == (archive / name).read_bytes(), name
sftp.put(str(archive / 'analyze_surrogate_transfer.py'), remote_run + '/analyze_surrogate_transfer.py')
_, out, err = client.exec_command('/root/miniconda3/envs/bdetr/bin/python ' + shlex.quote(remote_run + '/analyze_surrogate_transfer.py'), timeout=30)
remote_output = out.read()
assert out.channel.recv_exit_status() == 0, err.read().decode()
with sftp.open(remote_run + '/surrogate_transfer_analysis.json', 'rb') as stream:
    remote_analysis = stream.read()
assert json.loads(remote_analysis) == analysis
assert remote_analysis == (archive / 'surrogate_transfer_analysis.json').read_bytes().replace(b'\r\n', b'\n')
(archive / 'surrogate_transfer_remote_analysis.json').write_bytes(remote_analysis)
(archive / 'surrogate_transfer_remote_stdout.json').write_bytes(remote_output)
_, out, err = client.exec_command("/root/miniconda3/envs/bdetr/bin/python -c " + shlex.quote("import os,json,socket,shutil,subprocess;print(json.dumps({'uid':os.getuid(),'hostname':socket.gethostname(),'free_bytes':shutil.disk_usage('/root/autodl-tmp').free,'processes':subprocess.run(['ps','-p','62966,62969,63082,63084','-o','pid,comm,stat,etime'],stdout=subprocess.PIPE).stdout.decode(),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader']).decode()}))"), timeout=30)
observation = json.loads(out.read())
assert out.channel.recv_exit_status() == 0, err.read().decode()
assert len(observation['processes'].splitlines()) == 1
observation['time_cst'] = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
(archive / 'terminal_state_observation.json').write_text(json.dumps(observation, indent=2) + '\n', encoding='utf-8')

changed_docs = [report_name, 'docs/SCANREFER_MASK_GEOMETRY_GT_PLAN_2026-09-07.md',
                'docs/MASK_GEOMETRY_NATIVE_INITIALIZATION_2026-09-07.md']
for name in changed_docs[1:]:
    with sftp.open(canonical + name, 'rb') as stream:
        assert stream.read() == git('show', 'HEAD:' + name), name
for name, raw in [(master_name, new_master), (tracker_name, new_tracker)] + [(n, (repo / n).read_bytes()) for n in changed_docs]:
    with sftp.open(canonical + name, 'wb') as stream:
        stream.set_pipelined(True)
        stream.write(raw)
    with sftp.open(canonical + name, 'rb') as stream:
        stream.prefetch(file_size=len(raw))
        assert stream.read() == raw
master.write_bytes(new_master)
desktop.write_bytes(new_master)
(repo / tracker_name).write_bytes(new_tracker)
proof = {'time_cst': now, 'master_sha256': hashlib.sha256(new_master).hexdigest(),
         'three_master_copies_equal': True, 'integrity_pass': True, 'fixed_quality_pass': False,
         'remote_analysis_json_exact': True, 'raw_difference_only_windows_crlf': True,
         'formal_count': 0, 'nr_sr_started': False,
         'goal_complete': False, 'remote_observation': observation}
(archive / 'terminal_handoff_sync.json').write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8')
for name, source in [('publish_terminal_from_local.py', Path(__file__)),
                     ('collect_terminal_from_local.py', Path('C:/Users/gb/.codex/tmp/collect_mask_geometry_terminal_20260907.py')),
                     ('collect_fit_from_local.py', Path('C:/Users/gb/.codex/tmp/collect_mask_geometry_fit_20260907.py'))]:
    (archive / name).write_bytes(source.read_bytes())
for name in ['terminal_handoff_sync.json', 'terminal_state_observation.json', 'publish_terminal_from_local.py',
             'collect_terminal_from_local.py', 'collect_fit_from_local.py', 'terminal_collection.json', 'fit_collection.json']:
    sftp.put(str(archive / name), remote_run + '/' + name)
sftp.close()
client.close()
paths = [master_name, tracker_name, '.gitignore', archive_name, queue_name] + changed_docs
git('add', '--', *paths)
for name in git('diff', '--cached', '--name-only', '-z').decode().strip('\0').split('\0'):
    assert name in paths or name.startswith(archive_name + '/') or name.startswith(queue_name + '/'), name
    raw = git('show', ':' + name)
    assert os.environ['MCLN_SSH_PASSWORD'].encode() not in raw, name
    if name.endswith('.gz'):
        assert os.environ['MCLN_SSH_PASSWORD'].encode() not in gzip.decompress(raw), name
    assert len(raw) < 10 * 1024**2 and Path(name).suffix not in ['.pth', '.pt', '.pkl', '.npz'], name
    if name.startswith(archive_name + '/') or name.startswith(queue_name + '/'):
        assert raw == (repo / name).read_bytes(), name
git('-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--cached', '--check')
print(git('commit', '-m', 'Seal ScanRefer Mask geometry trial and trace surrogate output mismatch').decode(), flush=True)
print(git('push', 'origin', 'HEAD:main').decode(), flush=True)
head = git('rev-parse', 'HEAD').decode().strip()
assert git('ls-remote', 'origin', 'refs/heads/main').decode().split()[0] == head
assert not git('status', '--porcelain').strip()
proof['commit'] = head
Path('C:/Users/gb/.codex/tmp/mask_geometry_terminal_publication.json').write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8')
print(json.dumps(proof), flush=True)
