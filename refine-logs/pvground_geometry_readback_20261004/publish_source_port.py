"""Publish the isolated source port draft; never mutate the active run."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import paramiko

local = Path(__file__).resolve().parent
face = local.parent / 'pvground_face_conditioned_20261004'
assert not (local / 'port_publication.json').exists()
previous = json.loads((local / 'source_publication.json').read_bytes())
proof = json.loads((local / 'NATIVE_SOURCE_PORT_CHECK.json').read_bytes())
assert proof['status'] == 'SOURCE_PORT_DRAFT_AST_ONLY'
assert not proof['native_factory_constructed'] and not proof['active_face_source_changed']
for name, item in proof['files'].items():
    raw = (local / name).read_bytes()
    assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
for item in json.loads((face / 'EXPERIMENT_CODE_REVIEW.json').read_bytes())['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
workspace = Path('C:/Users/gb')
repos = [workspace / '.codex_mcln_g0_20260905', workspace / '.codex_pvground_cs_20261002',
         workspace / '.codex_mcln_v99_internal_20260928']
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies = [repo / doc for repo in repos] + [workspace / 'Desktop/document' / Path(doc).name]
old = copies[0].read_bytes()
assert hashlib.sha256(old).hexdigest() == previous['handoff_sha256']
assert len(old) == previous['handoff_bytes'] and all(path.read_bytes() == old for path in copies)
for repo, head in zip(repos, previous['heads']):
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip() == head
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
guard = workspace / '.codex/tmp/sync_cs_handoff_remote_20260923.py'
guard_raw = guard.read_bytes()
assert guard_raw.count(previous['handoff_sha256'].encode()) == 1
prefix = 'refine-logs/pvground_geometry_readback_20261004/'
names = list(proof['files']) + ['NATIVE_SOURCE_PORT_CHECK.json', 'NATIVE_SOURCE_PORT_SCOPE.md', 'publish_source_port.py']
payloads = {prefix + name: (local / name).read_bytes() for name in names}
assert len(payloads) == 9
assert all(not (repo / name).exists() for repo in repos[:2] for name in payloads)
stamp = datetime.datetime.now().astimezone().isoformat()
section = f'''

## 20.376.49 几何回写的原生调用接线草稿（{stamp}）

在独立`source_preview`中生成PV模型与预测头的最小修改：显式安装回写单元时，仅延后最后层语义子头；原框、分割与对比路径仍按原顺序计算，尾部精修后由原语义子头评分一次。原参数名称保留，模型工厂尚未真实构建；当前六面训练源码与进程未修改。输入源码摘要、生成差异和Python3.7 AST结果已保存，没有新源码评审PASS、GPU执行、优化或指标。

新增未调用的`native_root_bbs`辅助函数按实际评估公式汇总token-softmax：positive使用非零指示，增加modify/pron/rel证据并减去other_entity证据。它与原CE的0.6/0.2/0.2/0.1目标不同，不能当作普通概率直接套BCE；真实评估等价性及梯度响应仍待预检。几何来源须由本轮终态决定，结构回写与质量监督仍分别控制。

源码快照保留父PV文件29处既有行尾空白，仅对该独立快照配置字节/空白规则，没有顺手格式化活动源码。下一部署门槛为真实工厂与参数集合、零残差等价、语义头单次调用、BN/Dropout状态、固定框/Mask、实际分数响应和保存恢复；不得将本次AST通过写成这些检查已通过。
'''
new = old + section.encode('utf-8')
rule = b'\n# Preserve29 inherited whitespace lines in the exact isolated PV source snapshot.\n' + prefix.encode() + b'source_preview/** -text whitespace=cr-at-eol,-blank-at-eol,-blank-at-eof\n'
rule += prefix.encode() + b'NATIVE_SOURCE_PORT.diff -text whitespace=cr-at-eol,-blank-at-eol\n'
for repo in repos[:2]:
    assert prefix.encode() + b'source_preview/' not in (repo / '.gitattributes').read_bytes()

client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
remote = '/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(remote + '/' + doc, 'rb') as stream:
    assert stream.read() == old
for directory in ('source_preview', 'source_preview/PV-Ground', 'source_preview/PV-Ground/models'):
    sftp.mkdir(remote + '/' + prefix + directory)
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
assert all(path.read_bytes() == new for path in copies)
sftp.close()
client.close()

heads = []
for index, repo in enumerate(repos):
    if index < 2:
        path = repo / '.gitattributes'
        path.write_bytes(path.read_bytes() + rule)
        with (repo / 'MANIFEST.md').open('a', encoding='utf-8') as stream:
            stream.write(f'\n- {stamp} isolated delayed-semantic source port and exact bbs helper prepared; AST-only, no factory/GPU/accuracy, active fit unchanged.\n')
    stage = [doc] + (['MANIFEST.md', '.gitattributes', *payloads] if index < 2 else [])
    changed = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage])
    if index < 2:
        subprocess.check_call(['git', '-C', str(repo), 'add', '--renormalize', '--', prefix+'source_preview/'])
    subprocess.check_call(['git', '-C', str(repo), '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol',
                           'diff', '--cached', '--check', '--', *stage])
    for relative, raw in (payloads.items() if index < 2 else []):
        assert subprocess.check_output(['git', '-C', str(repo), 'show', ':'+relative]) == raw
    subprocess.check_call(['git', '-C', str(repo), 'commit', '-m', 'Prepare isolated native semantic deferral for boundary evidence readback'])
    heads.append(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip())
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
subprocess.check_call(['git', '-C', str(repos[0]), '-c', 'http.version=HTTP/1.1', 'push', 'origin', 'HEAD:main'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == heads[0]
digest = hashlib.sha256(new).hexdigest()
guard.write_bytes(guard_raw.replace(previous['handoff_sha256'].encode(), digest.encode()))
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), section='20.376.49',
    predecessor=str(local/'source_publication.json'), heads=heads, github_main=heads[0],
    handoff_sha256=digest, handoff_bytes=len(new), four_local_and_remote_equal=True,
    exact_committed_payloads=True, payload_count=len(payloads), execution_scope='SOURCE_PORT_DRAFT_AST_ONLY',
    native_factory_constructed=False, gpu_forwards=0, optimizer_updates=0, accuracy_result=False,
    active_face_fit_changed=False, retained_metric_best='plain_distribution_5616_4506', goal_achieved=False)
(local/'port_publication.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
with (workspace/'memory/2026-10-04.md').open('a',encoding='utf-8') as stream:
    stream.write(f'\nPVGround {record["time_cst"]} isolated delayed-semantic source port published {heads[0]}, doc49 fourlocal+remote equal; AST-only, no native factory/GPU/accuracy, active face fit unchanged. GoalACTIVE_UNMET.\n')
print(json.dumps(record),flush=True)
