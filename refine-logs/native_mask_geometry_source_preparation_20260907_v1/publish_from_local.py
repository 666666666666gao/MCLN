import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
archive_name = 'refine-logs/native_mask_geometry_source_preparation_20260907_v1'
archive = repo / archive_name
receipt = json.loads((archive / 'receipt.json').read_text(encoding='utf-8'))
assert receipt['status'] == 'pass' and receipt['source_files'] == 622
assert '6 passed' in (archive / 'check_2.txt').read_text(encoding='utf-8')
master_name = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
master = repo / master_name
desktop = Path('C:/Users/gb/Desktop/document') / master.name
old = master.read_bytes()
assert hashlib.sha256(old).hexdigest() == '8a5ee37f048f211ff7f3f310b06f1c906811d899adbfb53f7004b3c3d25d6c91'
assert desktop.read_bytes() == old
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
receipt_sha = hashlib.sha256((archive / 'receipt.json').read_bytes()).hexdigest()
addition = '''

### 20.139 原生Mask几何完整源码接续准备（TIME）

为解决20.138已发现的旧canonical源码差异，将已核验618文件原生依赖与Git9c7c9f4中的当前main_utils/losses及辅助文件组成622文件完整快照，位于`/root/autodl-tmp/mcln_native_mask_geometry_source_preparation_20260907_v1/model_source`。全部组成和SHA明确记录，不声称所有依赖来自同一次Git提交。旧canonical核心和当前Scan训练源码保持不变，无测试包的跨目录加载补丁。

20:34:41 CST检查完成：实际`train_dist_mod.py --help`成功，Nr3D/Sr3D均能用新开关构造原生criterion，旧local/extent开关关闭，7个关键模块普通导入来源均在完整快照内。该快照内6项现有原生辅助CPU集成检查通过（0.26秒），使用合成点云/Mask；没有重跑84项初始化fixture或新增网络模块。首次检查仅因命名空间重复列出相同scripts路径三次而失败，修正检查器为解析目录集合比较后通过，模型源码未改，原日志保留。

source manifest SHA `dbdc1da768fb0689f9b7ce1b140bfafc5cc2646d99fc3ed69b1ada5a27693030`；receipt SHA `RECEIPT_SHA`。输入、初次检查、路径诊断、最终6项测试和普通导入记录归档`refine-logs/native_mask_geometry_source_preparation_20260907_v1/`。GPU前向0、优化器更新0、新权重0、正式行0、真实Scan终点绑定尚未执行。该快照是后续原生接续基础，不是Nr/Sr训练或精度结果。

核对旧range预检仍绑定已失败extent，不能直接拿来启动当前机制；nr_contract中的历史--eval、E57路径和max_epoch240也不是新训练授权。正式Scan晋级后需在上述完整源目录绑定真实84项初始化，核验实际数据批次与明确训练预算，再尽快Nr/Sr REC。现有Scan配对/唯一终态/条件正式队列设置不变。

最近实际进程观察为20:22:40：Python62969存活，两臂1216/2482更新（约49%），loss/梯度有限、磁盘10485620736 bytes；20:23:51接续Python63084存活，正式目录尚未创建。预计训练21:28左右、终态核验22:05—22:15左右，下一主要观察21:20；这些是当时进度与估计，不是本节发布时的新训练记录。三数据集总目标尚未完成，正式最好未变。
'''.replace('TIME', now).replace('RECEIPT_SHA', receipt_sha)
new_master = old + addition.encode('utf-8')
tracker_name = 'refine-logs/EXPERIMENT_TRACKER.md'
tracker = (repo / tracker_name).read_text(encoding='utf-8')
start = tracker.index('Updated:')
end = tracker.index('\n', start)
tracker = tracker[:start] + 'Updated: ' + now + '. Section20.139: complete native source packaging PASS;Scan fixed training ongoing, no terminal quality yet.' + tracker[end:]
tracker = tracker.replace('|---|---|---|\n', '|---|---|---|\n| Native Mask geometry complete source | 622 files bound;normal CLI and7 module imports verified;6existing CPU criterion tests PASS | No package-path overlay or canonical overwrite;real endpoint/GPU training still conditional on Scan promotion |\n', 1)
tracker = tracker.replace('19:32 actual256/2482 perarm,Python62969 live;queue63084 confirmed19:34',
    '20:22 actual1216/2482 perarm,Python62969 live;queue63084 confirmed20:23')


def git(*args):
    return subprocess.check_output(['git'] + list(args), cwd=str(repo))


assert git('rev-parse', 'HEAD').decode().strip() == '9c7c9f44ecfdada79cddfcfedfe2cfd47957b935'
c = paramiko.SSHClient()
c.load_system_host_keys()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
s = c.open_sftp()
canonical = '/home/gb/new butd/butd_detr-main/MCLN-main/'
for name in [master_name, 'docs/MASK_GEOMETRY_NATIVE_INITIALIZATION_2026-09-07.md', tracker_name]:
    with s.open(canonical + name, 'rb') as stream:
        actual = stream.read()
    assert actual == (old if name == master_name else git('show', 'HEAD:' + name)), name
for name, raw in [(master_name, new_master), (tracker_name, tracker.encode('utf-8')),
                  ('docs/MASK_GEOMETRY_NATIVE_INITIALIZATION_2026-09-07.md', (repo / 'docs/MASK_GEOMETRY_NATIVE_INITIALIZATION_2026-09-07.md').read_bytes())]:
    with s.open(canonical + name, 'wb') as stream:
        stream.set_pipelined(True)
        stream.write(raw)
    with s.open(canonical + name, 'rb') as stream:
        stream.prefetch(file_size=len(raw))
        assert stream.read() == raw
master.write_bytes(new_master)
desktop.write_bytes(new_master)
(repo / tracker_name).write_bytes(tracker.encode('utf-8'))
proof = {'time_cst': now, 'master_sha256': hashlib.sha256(new_master).hexdigest(),
         'three_master_copies_equal': True, 'source_manifest_sha256': receipt['source_manifest_sha256'],
         'receipt_sha256': receipt_sha, 'cpu_tests': 6, 'source_files': 622,
         'gpu_training_started': False, 'goal_complete': False}
(archive / 'handoff_sync.json').write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8')
for filename, source in [('publish_from_local.py', Path(__file__)),
                         ('stage_from_local.py', Path('C:/Users/gb/.codex/tmp/stage_native_mask_geometry_source_20260907.py')),
                         ('finish_from_local.py', Path('C:/Users/gb/.codex/tmp/finish_native_mask_source_20260907.py'))]:
    (archive / filename).write_bytes(source.read_bytes())
remote = '/root/autodl-tmp/mcln_native_mask_geometry_source_preparation_20260907_v1'
for name in ['handoff_sync.json', 'publish_from_local.py', 'stage_from_local.py', 'finish_from_local.py', 'namespace_diagnostic.txt']:
    s.put(str(archive / name), remote + '/' + name)
s.close()
c.close()
with (repo / '.gitattributes').open('a', encoding='utf-8', newline='\n') as stream:
    stream.write(archive_name + '/** -text\n')
    stream.write(archive_name + '/*.txt -whitespace\n')
paths = [master_name, tracker_name, 'docs/MASK_GEOMETRY_NATIVE_INITIALIZATION_2026-09-07.md',
         archive_name, '.gitattributes', 'refine-logs/scanrefer_mask_geometry_pair_20260907_v1/latest_observation.json']
assert not git('diff', '--cached', '--name-only').strip()
git('add', '--', *paths)
for name in git('diff', '--cached', '--name-only', '-z').decode().strip('\0').split('\0'):
    assert name in paths or name.startswith(archive_name + '/'), name
    raw = git('show', ':' + name)
    assert os.environ['MCLN_SSH_PASSWORD'].encode() not in raw
    assert len(raw) < 10 * 1024**2 and Path(name).suffix not in ['.pth', '.pt', '.pkl', '.npz']
    if name.startswith(archive_name + '/'):
        assert raw == (repo / name).read_bytes()
git('-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--cached', '--check')
print(git('commit', '-m', 'Prepare complete native Mask geometry source for cross-dataset continuation').decode(), flush=True)
print(git('push', 'origin', 'HEAD:main').decode(), flush=True)
head = git('rev-parse', 'HEAD').decode().strip()
assert git('ls-remote', 'origin', 'refs/heads/main').decode().split()[0] == head
assert not git('status', '--porcelain').strip()
proof['commit'] = head
Path('C:/Users/gb/.codex/tmp/native_mask_source_publication.json').write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8')
print(json.dumps(proof), flush=True)
