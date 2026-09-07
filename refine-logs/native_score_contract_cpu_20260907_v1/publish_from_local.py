import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
archive_name = 'refine-logs/native_score_contract_cpu_20260907_v1'
archive = repo / archive_name
master_name = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
tracker_name = 'refine-logs/EXPERIMENT_TRACKER.md'
report_name = 'docs/NATIVE_REC_SCORE_CONTRACT_AUDIT_2026-09-07.md'
script_name = 'scripts/audit_native_rec_score_semantics.py'
desktop = Path('C:/Users/gb/Desktop/document') / Path(master_name).name
old = (repo / master_name).read_bytes()
assert hashlib.sha256(old).hexdigest() == 'b6b52d6640d8b027d6da7ebf727fe994d28b602fe2a060c6855af5edf897b942'
assert desktop.read_bytes() == old
raw_receipt = (archive / 'receipt.json').read_bytes()
assert hashlib.sha256(raw_receipt).hexdigest() == '8fb6e1bcc7d5a44c5545faba0ca7691f4af79584a5cd21c886c517ca9c23aeef'
receipt = json.loads(raw_receipt)
assert receipt['status'] == 'pass' and receipt['dataset_rows'] == 0


def git(*args):
    return subprocess.check_output(['git'] + list(args), cwd=str(repo))


assert git('rev-parse', 'HEAD').decode().strip() == '9612b4bec28a1bb2ec38cc37fe8732a768f7a317'
assert not git('diff', '--cached', '--name-only').strip()
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
addition = '''

### 20.141 原生与候选适配器的default口径核对（TIME）

沿20.140转回REC监督/读出检查。原生evaluator与SourceChoice default均将main正值置1；SourceChoice历史V4修复仍存在，不是未修复旧问题。Parent候选特征适配器保留输入main权重，dataset存在按token数归一化的构造，原生_get_inputs透传。故同名default并不保证同一数值口径。Hungarian分类成本使用main，position alignment使用原有分项权重，另外还有框/对比监督；不能写成原生完全没有关系或质量训练。

原Python3.7/Torch1.10 CPU隔离执行现有源码表达式的1/2/4个main token合成例：原生两候选分数均约0.400/0.350、选A，SourceChoice一致；适配器依次0.400/0.350、0.200/0.300、0.100/0.275，后两例改选B。输入保持不变。仅证明该定义差异能改变排名；没有加载模型、真实数据、完整evaluator过滤或旧学习读出，GPU/更新/正式行均0，不证明任何指标收益。

保护Parent/Geometry/V99已按旧特征分布训练，不能直接改map破坏其契约。已有SourceMoE和joint/frozen_readout还包括quality/listwise目标，普通质量loss也不是新方向。下一步在固定fit实际输入检查map分布、两种排名及原生Top1被旧Top16保留情况，明确数值差异是否影响真实候选；不在正式集调map，也不把合成反例当新长训结果或授权。

源码6文件与Git/本地/622快照核对：首次仅因losses及dataset的CRLF/LF原始SHA不同停在源检查，标准化内容完全一致；改用实际远端字节SHA后检查PASS，模型源码未改，初次日志保留。receipt SHA8fb6e1bcc7d5a44c5545faba0ca7691f4af79584a5cd21c886c517ca9c23aeef；报告NATIVE_REC_SCORE_CONTRACT_AUDIT_2026-09-07.md，归档refine-logs/native_score_contract_cpu_20260907_v1。Scan Mask几何固定版本仍封存，当前无训练等待，三数据集正式最好与目标未变。
'''.replace('TIME', now)
new = old + addition.encode('utf-8')
tracker = (repo / tracker_name).read_text(encoding='utf-8').splitlines()
tracker[2] = 'Updated: ' + now + '. Section20.141: CPU score-definition counterexample complete;Mask geometry fixed trial sealed;no live training/formal/Nr/Sr.'
tracker.insert(6, '| Native default score definitions | Existing evaluator/SourceChoice agree;adapter changesTop1 in2/3synthetic token-count cases | Contract difference only;no rule change or metric gain;real fit prevalence remains to measure |')
new_tracker = ('\n'.join(tracker) + '\n').encode('utf-8')
c = paramiko.SSHClient()
c.load_system_host_keys()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
s = c.open_sftp()
canonical = '/home/gb/new butd/butd_detr-main/MCLN-main/'
remote = '/root/autodl-tmp/mcln_native_score_contract_cpu_20260907_v1'
for name in [master_name, tracker_name]:
    with s.open(canonical + name, 'rb') as stream:
        assert stream.read() == git('show', 'HEAD:' + name), name
with s.open(remote + '/receipt.json', 'rb') as stream:
    assert stream.read() == raw_receipt
for name, raw in [(master_name, new), (tracker_name, new_tracker),
                  (report_name, (repo / report_name).read_bytes()), (script_name, (repo / script_name).read_bytes())]:
    with s.open(canonical + name, 'wb') as stream:
        stream.set_pipelined(True)
        stream.write(raw)
    with s.open(canonical + name, 'rb') as stream:
        stream.prefetch(file_size=len(raw))
        assert stream.read() == raw
(repo / master_name).write_bytes(new)
desktop.write_bytes(new)
(repo / tracker_name).write_bytes(new_tracker)
proof = {'time_cst': now, 'three_master_copies_equal': True,
         'master_sha256': hashlib.sha256(new).hexdigest(),
         'receipt_sha256': hashlib.sha256(raw_receipt).hexdigest(),
         'synthetic_cases': 3, 'changed_model_rules': False, 'gpu_forwards': 0,
         'dataset_rows': 0, 'formal_rows': 0, 'goal_complete': False}
(archive / 'handoff_sync.json').write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8')
(archive / 'publish_from_local.py').write_bytes(Path(__file__).read_bytes())
for name in ['source_binding_line_endings.json', 'handoff_sync.json', 'run_from_local.py', 'finish_from_local.py', 'publish_from_local.py']:
    s.put(str(archive / name), remote + '/' + name)
s.close()
c.close()
with (repo / '.gitattributes').open('a', encoding='utf-8', newline='\n') as stream:
    stream.write(archive_name + '/** -text\n')
paths = [master_name, tracker_name, report_name, script_name, archive_name, '.gitattributes']
git('add', '--', *paths)
for name in git('diff', '--cached', '--name-only', '-z').decode().strip('\0').split('\0'):
    assert name in paths or name.startswith(archive_name + '/'), name
    raw = git('show', ':' + name)
    assert os.environ['MCLN_SSH_PASSWORD'].encode() not in raw
    assert len(raw) < 10 * 1024**2 and Path(name).suffix not in ['.pt', '.pth', '.pkl', '.npz']
    if name.startswith(archive_name + '/'):
        assert raw == (repo / name).read_bytes()
git('-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--cached', '--check')
print(git('commit', '-m', 'Audit native and candidate REC score definitions without changing deployment').decode(), flush=True)
print(git('push', 'origin', 'HEAD:main').decode(), flush=True)
head = git('rev-parse', 'HEAD').decode().strip()
assert git('ls-remote', 'origin', 'refs/heads/main').decode().split()[0] == head
assert not git('status', '--porcelain').strip()
proof['commit'] = head
Path('C:/Users/gb/.codex/tmp/native_score_contract_publication.json').write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8')
print(json.dumps(proof), flush=True)
