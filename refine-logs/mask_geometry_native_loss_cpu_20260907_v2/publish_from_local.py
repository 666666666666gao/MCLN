import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
archives = ['refine-logs/mask_geometry_native_loss_cpu_20260907_v1',
            'refine-logs/mask_geometry_native_loss_cpu_20260907_v2']
record = json.loads((repo / archives[1] / 'receipt.json').read_bytes())
assert record['status'] == 'pass' and record['pytest_exit'] == 0
assert '11 passed' in (repo / archives[1] / 'cpu_tests.txt').read_text(encoding='utf-8')
master = repo / 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop = Path('C:/Users/gb/Desktop/document') / master.name
old = master.read_bytes()
assert desktop.read_bytes() == old
assert hashlib.sha256(old).hexdigest() == '7956279ea356e5d0b355f118ea11350382a6691b7ed1d78f10e8d699a5e38aaf'
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
addition = '''

### 20.138 Mask几何辅助接入原生Nr/Sr训练入口，CPU集成通过（TIME）

核验发现原生joint_det把scannet纯检测样本加入训练（dataset_dict['scannet']=10），不能把检测行第一个框当成句子指代目标。新增默认关闭的`--native_mask_geometry_supervision`，经BaseTrainTester进入原生compute_hungarian_loss；直接读取已有last_match_indices，只对sample_dataset非scannet的指代表达行加入当前固定辅助，系数1、分位数0.005及L1/GIoU公式不变。纯检测批次不调用辅助，保留原生损失和梯度；现有专用输出头提前返回模式不能与此开关组合。未增加推理模块或新的可调辅助权重。

原环境11项CPU检查通过（0.89秒测试，1.57秒整体），实际调用原生loss入口、HungarianMatcher和SetCriterion，使用合成点云/Mask：Nr/Sr混合批次选择last而非proposal匹配，只给指代表达的当前root Query附加Mask梯度；检测行梯度不变；全检测批次原loss及梯度不变；默认关闭不调用辅助或新增字段；批次子集与单行目标一致。现有辅助/CLI/损失转发和density默认路径回归也通过。第一次仅因CPU测试包缺少旧测试按绝对相对路径读取的density模块而收集失败；补齐原文件后通过，未改网络逻辑来绕过测试。

GPU前向0、优化器更新0、权重写入0、正式评估0。原生快照618文件及当前Scan训练manifest/source在检查前后逐项SHA一致；运行中配对源码不改。CPU结果归档`refine-logs/mask_geometry_native_loss_cpu_20260907_v2`，receipt SHA `980acb3b1a07e9179894399770595a1bfa27add7f5ebcf26b84cd9eb70fc0511`。用法与边界已续写`docs/MASK_GEOMETRY_NATIVE_INITIALIZATION_2026-09-07.md`。

这只完成跨数据集原生训练的loss接续代码与CPU集成，不证明真实数据训练或REC收益。当前Scan仍按20.135/20.137既定配对、终态和正式门接续；不在专用Scan循环同时开启此原生flag，以免重复加辅助。正式Scan通过后才导出实际84项终点，并用新原生入口核验Nr/Sr真实GPU批次和启动训练。未提前启动Nr/Sr；运行观察仍以20.137的实际时间为准，下一次主要观察接近训练末尾。完整三数据集目标未完成。

发布预检发现旧远端canonical main_utils只有CRLF差异，但models/losses尚缺Git已有的density和部分counterfactual接口（标准化后3969行对4170行），最近50次该文件历史中未找到完全相同版本。本轮未覆盖旧canonical main_utils/models.losses；新增入口完整源码已在上述v2隔离目录并按明确overlay路径通过CPU测试，同时发布GitHub。该差异不影响当前冻结Scan快照，后续正式训练必须从明确的新源码快照启动，不能直接混用旧canonical目录。差异证据保存remote_canonical_source_differences.json；不据此改写受保护指标。
'''.replace('TIME', now)
new = old + addition.encode()
tracker = repo / 'refine-logs/EXPERIMENT_TRACKER.md'
text = tracker.read_text(encoding='utf-8')
start = text.index('Updated:'); end = text.index('\n', start)
text = text[:start] + 'Updated: ' + now + '. Section20.138: native Mask geometry loss CPU integration PASS;Scan pair continues, no new formal result.' + text[end:]
row = '| Native Mask geometry loss integration | 11CPU checks PASS using actual native loss/Hungarian/SetCriterion on synthetic tensors;default off;ScanNet detection rows excluded | No GPU/dataset training or new weights;actual Nr/Sr entry only after formalScan promotion |\n'
text = text.replace('|---|---|---|\n', '|---|---|---|\n' + row, 1)
text = text.replace('Fixed pair LIVE Python62969 since18:35;18:40 data parsing,no updates verified;gated formal queue63084 live;4+8 CPU checks PASS;not a REC result',
                    'Baseline6887 exact/recount PASS;19:32 actual256/2482 perarm,Python62969 live;queue63084 confirmed19:34;no new formal result')
tracker.write_bytes(text.encode())
paths = ['main_utils.py', 'models/losses.py', 'scripts/native_mask_geometry_supervision.py',
         'tests/test_native_mask_geometry_training.py', 'docs/MASK_GEOMETRY_NATIVE_INITIALIZATION_2026-09-07.md',
         'refine-logs/EXPERIMENT_TRACKER.md']
remote_paths = [name for name in paths if name not in ['main_utils.py', 'models/losses.py']]
differences = Path('C:/Users/gb/.codex/tmp/native_loss_remote_source_differences.json').read_bytes()
(repo / archives[1] / 'remote_canonical_source_differences.json').write_bytes(differences)


def git(*args):
    return subprocess.check_output(['git'] + list(args), cwd=repo)


client = paramiko.SSHClient(); client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp(); runtime = '/home/gb/new butd/butd_detr-main/MCLN-main/'
with sftp.open(runtime + 'docs/' + master.name, 'rb') as stream:
    stream.prefetch(file_size=len(old)); assert stream.read() == old
training = '/root/autodl-tmp/mcln_scanrefer_mask_geometry_pair_20260907_v1'
with sftp.open(training + '/input_manifest.json', 'rb') as stream:
    raw = stream.read()
assert hashlib.sha256(raw).hexdigest() == '15f46411069a7172a55373c5c13075b22bcb4146d39251e9fca2c37ed5867eb3'
for name, digest in json.loads(raw)['files'].items():
    with sftp.open(training + '/' + name, 'rb') as stream:
        assert hashlib.sha256(stream.read()).hexdigest() == digest, name
for name in remote_paths:
    if name != 'tests/test_native_mask_geometry_training.py':
        expected = git('show', 'HEAD:' + name)
        with sftp.open(runtime + name, 'rb') as stream:
            assert stream.read() == expected, name
for name in remote_paths:
    raw = (repo / name).read_bytes()
    with sftp.open(runtime + name, 'wb') as stream:
        stream.set_pipelined(True); stream.write(raw)
    with sftp.open(runtime + name, 'rb') as stream:
        stream.prefetch(file_size=len(raw)); assert stream.read() == raw
with sftp.open(runtime + 'docs/' + master.name, 'wb') as stream:
    stream.set_pipelined(True); stream.write(new)
with sftp.open(runtime + 'docs/' + master.name, 'rb') as stream:
    stream.prefetch(file_size=len(new)); assert stream.read() == new
master.write_bytes(new); desktop.write_bytes(new)
proof = {'time_cst': now, 'master_sha256': hashlib.sha256(new).hexdigest(),
         'three_master_copies_equal': True, 'running_source_unchanged': True,
         'cpu_tests': 11, 'actual_dataset_gpu_training': False, 'goal_complete': False,
         'legacy_canonical_main_and_loss_preserved': True,
         'new_native_main_and_loss_remote_directory': '/root/autodl-tmp/mcln_mask_geometry_native_loss_cpu_20260907_v2'}
final = repo / archives[1]
(final / 'handoff_sync.json').write_bytes((json.dumps(proof, indent=2) + '\n').encode())
(final / 'publish_from_local.py').write_bytes(Path(__file__).read_bytes())
for name in ['handoff_sync.json', 'publish_from_local.py', 'stage_from_local.py', 'remote_canonical_source_differences.json']:
    sftp.put(str(final / name), '/root/autodl-tmp/mcln_mask_geometry_native_loss_cpu_20260907_v2/' + name)
sftp.close(); client.close()
with (repo / '.gitattributes').open('a', encoding='utf-8', newline='\n') as stream:
    for archive in archives:
        stream.write(archive + '/** -text -whitespace\n')
paths += ['docs/' + master.name, '.gitattributes'] + archives
assert not git('diff', '--cached', '--name-only').strip()
git('add', '--', *paths)
for archive in archives:
    git('add', '--renormalize', '--', archive)
for name in git('diff', '--cached', '--name-only', '-z').decode().strip('\0').split('\0'):
    assert name in paths or any(name.startswith(archive + '/') for archive in archives), name
    raw = git('show', ':' + name)
    assert os.environ['MCLN_SSH_PASSWORD'].encode() not in raw
    assert Path(name).suffix.lower() not in ['.pt', '.pth', '.npz', '.pkl'] and len(raw) < 10 * 1024**2
    if any(name.startswith(archive + '/') for archive in archives):
        assert raw == (repo / name).read_bytes()
git('-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--cached', '--check')
print(git('commit', '-m', 'Integrate root Mask geometry supervision into native grounding training').decode(), flush=True)
print(git('push', 'origin', 'HEAD:main').decode(), flush=True)
head = git('rev-parse', 'HEAD').decode().strip()
assert git('ls-remote', 'origin', 'refs/heads/main').decode().split()[0] == head
assert not git('status', '--porcelain').strip()
proof['commit'] = head
Path('C:/Users/gb/.codex/tmp/native_mask_geometry_loss_publication.json').write_bytes((json.dumps(proof, indent=2) + '\n').encode())
print(json.dumps(proof), flush=True)
