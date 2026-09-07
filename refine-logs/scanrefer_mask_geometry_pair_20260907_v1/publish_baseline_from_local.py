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
archive = 'refine-logs/scanrefer_mask_geometry_pair_20260907_v1'
local = repo / archive
record = json.loads((local / 'baseline_cross_run_audit.json').read_bytes())
assert record['status'] == 'pass' and record['current_arms_exact_row_parity']
assert record['identity_difference_row_ids'] == []
assert all(not differences for differences in record['cross_run_difference_row_ids'].values())
for name, digest in record['baseline_file_sha256'].items():
    assert hashlib.sha256((local / name).read_bytes()).hexdigest() == digest
raw_rows = (local / 'baseline_rows.json').read_bytes()
assert len(raw_rows) > 10 * 1024**2
compressed = gzip.compress(raw_rows, mtime=0)
assert len(compressed) < 10 * 1024**2 and gzip.decompress(compressed) == raw_rows
(local / 'baseline_rows.json.gz').write_bytes(compressed)
ignore = repo / '.gitignore'
entry = '/' + archive + '/baseline_rows.json'
assert entry not in ignore.read_text(encoding='utf-8').splitlines()
with ignore.open('a', encoding='utf-8', newline='\n') as stream:
    stream.write('\n# Full Mask geometry baseline is archived losslessly as baseline_rows.json.gz.\n' + entry + '\n')
observation = json.loads((local / 'latest_observation.json').read_bytes())
assert '62969' in observation['processes'] and not observation['exit_files']
latest_train = json.loads([line.split('TRAIN ', 1)[1] for line in observation['progress']
                           if line.startswith('SCANREFER MASK GEOMETRY GT TRAIN')][-1])
master = repo / 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop = Path('C:/Users/gb/Desktop/document') / master.name
old = master.read_bytes()
assert desktop.read_bytes() == old
assert hashlib.sha256(old).hexdigest() == '9ba058adb9fef9e926347e6958a9e1a7942dac07fe1b6e4980f8799a80335501'
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
system, native = record['system_metrics'], record['native_rec_metrics']
geometry = record['matched_root_geometry_metrics']
addition = '''

### 20.137 Mask几何配对完整起点重算与跨运行一致性（TIME）

起点6887条模块留出已实际完成，两臂全部逐行记录完全一致。CPU独立重算完整V99为SYSTEM025/SYSTEM050，原生REC为NATIVE025/NATIVE050；Mask为MASK025/MASK050，mIoU MASKMIOU%。与此前教师框转移实验的同E71/mesh起点相比，6887条行身份、场景、物理空间和输入点SHA差异0；原生Query索引、原生IoU、系统IoU、Mask IoU及实际几何变体位置差异也全部0。不是新模型增益，正式9508尚未执行。

本次新增GT匹配root Query几何记录：有效硬框HARDVALID/6887；软范围过线SOFTHITS，实际硬范围过线HARDHITS；平均IoU软SOFTMEAN、硬HARDMEAN（硬无效按0）。从保存的中心/尺寸独立重算软/硬IoU、中心尺寸L1、GIoU以及平均辅助loss，全部在预先沿用的数值容差内通过。这里的候选由训练GT匹配取得，不是实际部署REC，也不等同完整候选oracle。

核验未运行模型forward、未更新参数、未写新权重。完整baseline_rows.json保存在服务器及本地归档，SHA ROWSHA；Git归档使用已逐字节解压核对的baseline_rows.json.gz，避免提交12.8MB原始JSON。指标和baseline_cross_run_audit.py/json已归档，审计SHA AUDITSHA。审计仅读输出，不修改正在运行的训练文件、终态CPU审计、正式晋级门或接续队列。

最新真实运行观察时间OBSERVED：PROGRESS。当前训练仍为原Python62969，未重复启动；固定2482更新/臂及唯一终点评估继续。完整训练、正式Scan、Nr/Sr训练均不能以这个起点通过替代；正式Scan过现行REC与Mask底线后按原要求接Nr/Sr REC，不等待59/51争取线。完整目标未完成。
'''
values = {'TIME': now, 'SYSTEM025': system['rec_hits025'], 'SYSTEM050': system['rec_hits050'],
          'NATIVE025': native['rec_hits025'], 'NATIVE050': native['rec_hits050'],
          'MASK025': system['mask_hits025'], 'MASK050': system['mask_hits050'],
          'MASKMIOU': format(system['mask_miou'], '.8f'), 'HARDVALID': geometry['hard_valid'],
          'SOFTHITS': geometry['soft_hits'], 'HARDHITS': geometry['hard_hits'],
          'SOFTMEAN': format(geometry['soft_mean_iou'], '.8f'),
          'HARDMEAN': format(geometry['hard_mean_iou_invalid_as_zero'], '.8f'),
          'ROWSHA': record['baseline_file_sha256']['baseline_rows.json'],
          'AUDITSHA': hashlib.sha256((local / 'baseline_cross_run_audit.json').read_bytes()).hexdigest(),
          'OBSERVED': observation['time_cst'],
          'PROGRESS': '两臂均已完成{}/2482次实际更新；累计训练{:.2f}秒，日志估计训练剩余{:.2f}秒。该批每臂{}/84项参数有梯度，两个loss及梯度范数有限；损失数值不是质量晋级证据'.format(
              latest_train['step'], latest_train['elapsed_seconds'],
              latest_train['estimated_training_remaining_seconds'],
              latest_train['arms']['native_gt_mask_geometry']['gradient_presence'].count('1'))}
for key, value in values.items():
    addition = addition.replace(key, str(value))
new = old + addition.encode()
names = ['baseline_cross_run_audit.py', 'baseline_cross_run_audit.json', 'baseline_rows.json.gz',
         'baseline_metrics.json', 'baseline_native_metrics.json', 'baseline_geometry_metrics.json',
         'run_baseline_audit_from_local.py', 'latest_observation.json']
paths = [archive + '/' + name for name in names]
queue_observation = 'refine-logs/scanrefer_mask_geometry_posttraining_20260907_v1/latest_observation.json'
paths += [queue_observation, '.gitignore']
client = paramiko.SSHClient(); client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
runtime = '/home/gb/new butd/butd_detr-main/MCLN-main/'
training = '/root/autodl-tmp/mcln_scanrefer_mask_geometry_pair_20260907_v1'
queue = '/root/autodl-tmp/mcln_scanrefer_mask_geometry_posttraining_20260907_v1'
queue_probe = '''import datetime,json,subprocess
from pathlib import Path
root=Path(QUEUE)
print(json.dumps({'time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
'processes':subprocess.check_output(['ps','-p','63082,63084','-o','pid,ppid,comm,stat,etime,args'],universal_newlines=True),
'terminal_files':{p.name:p.read_text() for p in root.iterdir() if p.name in ['controller.exit','queue.exit','decision.json']},
'formal_directory_exists':Path('/root/autodl-tmp/mcln_scanrefer_mask_geometry_official_20260907_v1').exists(),
'observations_tail':(root/'training_observations.jsonl').read_text()[-6000:]}))
'''.replace('QUEUE', repr(queue))
_, output, error = client.exec_command('/root/miniconda3/envs/bdetr/bin/python -c ' + shlex.quote(queue_probe), timeout=30)
body = output.read(); errors = error.read().decode()
assert output.channel.recv_exit_status() == 0, errors
queue_record = json.loads(body)
assert '63084' in queue_record['processes'] and not queue_record['terminal_files']
(repo / queue_observation).write_bytes((json.dumps(queue_record, indent=2) + '\n').encode())
with sftp.open(runtime + 'docs/' + master.name, 'rb') as stream:
    stream.prefetch(file_size=len(old)); assert stream.read() == old
with sftp.open(training + '/input_manifest.json', 'rb') as stream:
    raw = stream.read()
assert hashlib.sha256(raw).hexdigest() == record['training_manifest_sha256']
for name, digest in json.loads(raw)['files'].items():
    with sftp.open(training + '/' + name, 'rb') as stream:
        assert hashlib.sha256(stream.read()).hexdigest() == digest, name
with sftp.open(runtime + 'docs/' + master.name, 'wb') as stream:
    stream.set_pipelined(True); stream.write(new)
with sftp.open(runtime + 'docs/' + master.name, 'rb') as stream:
    stream.prefetch(file_size=len(new)); assert stream.read() == new
master.write_bytes(new); desktop.write_bytes(new)
proof = {'time_cst': now, 'master_sha256': hashlib.sha256(new).hexdigest(),
         'three_master_copies_equal': True, 'running_source_unchanged': True,
         'baseline_audit_sha256': values['AUDITSHA'], 'formal_rows': 0, 'goal_complete': False}
(local / 'baseline_handoff_sync.json').write_bytes((json.dumps(proof, indent=2) + '\n').encode())
(local / 'publish_baseline_from_local.py').write_bytes(Path(__file__).read_bytes())
for name in ['baseline_handoff_sync.json', 'publish_baseline_from_local.py', 'latest_observation.json', 'baseline_rows.json.gz']:
    sftp.put(str(local / name), training + '/' + name)
sftp.put(str(repo / queue_observation), queue + '/latest_observation.json')
sftp.close(); client.close()
paths += [archive + '/baseline_handoff_sync.json', archive + '/publish_baseline_from_local.py', 'docs/' + master.name]


def git(*args):
    return subprocess.check_output(['git'] + list(args), cwd=repo)


assert not git('diff', '--cached', '--name-only').strip()
git('add', '--', *paths)
staged = git('diff', '--cached', '--name-only', '-z').decode().strip('\0').split('\0')
for name in staged:
    assert name in paths, name
    raw = git('show', ':' + name)
    assert os.environ['MCLN_SSH_PASSWORD'].encode() not in raw
    assert Path(name).suffix.lower() not in ['.pth', '.pt', '.pkl', '.npz'] and len(raw) < 10 * 1024**2
    if name.startswith('refine-logs/'):
        assert raw == (repo / name).read_bytes()
git('-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--cached', '--check')
print(git('commit', '-m', 'Verify complete Mask geometry baseline against protected outputs').decode(), flush=True)
print(git('push', 'origin', 'HEAD:main').decode(), flush=True)
head = git('rev-parse', 'HEAD').decode().strip()
assert git('ls-remote', 'origin', 'refs/heads/main').decode().split()[0] == head
assert not git('status', '--porcelain').strip()
proof['commit'] = head
Path('C:/Users/gb/.codex/tmp/mask_geometry_baseline_publication.json').write_bytes((json.dumps(proof, indent=2) + '\n').encode())
print(json.dumps(proof), flush=True)
