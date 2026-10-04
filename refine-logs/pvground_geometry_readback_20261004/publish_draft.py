"""Publish the isolated AST-only readback draft, without changing the active fit."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import paramiko

local = Path(__file__).resolve().parent
face = local.parent / 'pvground_face_conditioned_20261004'
assert not (local / 'source_publication.json').exists()
previous = json.loads((face / 'launch_publication.json').read_bytes())
check = json.loads((local / 'LOCAL_SOURCE_CHECK.json').read_bytes())
assert check['status'] == 'LOCAL_AST_ONLY_UNINTEGRATED'
assert check['gpu_forwards'] == check['optimizer_updates'] == 0
assert not check['native_factory_integrated'] and not check['active_face_fit_changed']
for name, item in check['files'].items():
    raw = (local / name).read_bytes()
    assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
review = json.loads((face / 'EXPERIMENT_CODE_REVIEW.json').read_bytes())
for item in review['reviewed_files']:
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
names = ('pvground_boundary_evidence_readback.py', 'pvground_boundary_box_refiner.py',
         'SOURCE_SCOPE.md', 'check_source.py', 'LOCAL_SOURCE_CHECK.json', 'publish_draft.py')
payloads = {prefix + name: (local / name).read_bytes() for name in names}
face_prefix = 'refine-logs/pvground_face_conditioned_20261004/'
for name in ('prepare_readback_interface.py', 'READBACK_INTERFACE_WITNESS.json', 'READBACK_INTERFACE.md'):
    payloads[face_prefix + name] = (face / name).read_bytes()
assert len(payloads) == 9
assert all(not (repo / name).exists() for repo in repos[:2] for name in payloads)
stamp = datetime.datetime.now().astimezone().isoformat()
section = f'''

## 20.376.48 最终几何证据回写：依赖核对与未集成草稿（{stamp}）

当前六面条件化训练仍按§20.376.47的隔离源码运行，本次没有部署新模块、修改活动训练、查询新的精度或创建权重。单一观察器仍按预计阶段结束时间检查；当前保留指标最好为5616/4506，目标尚未达成。

对真实预检导入的PV模型、原生预测头和评估器做SHA绑定的只读依赖核对：最后层语义分数在Mask与尾部精修之前产生，Mask不依赖该分数；语义子头含两层BN和两层Dropout。后续独立集成应只延后最后层语义子头，并在几何回写后调用一次，不能重复整个框头，也不能将一个质量标量广播到全部token后期待改变token-softmax。

独立目录`refine-logs/pvground_geometry_readback_20261004`准备了`BoundaryEvidenceReadback`草稿。六面各读取44维最终分布、粗/精修边界、对应轴支撑统计与实际尺寸下限状态；结合原生语义Query和完整文本，再由同候选Query读取六面证据，零初始化残差返回原生288维表示。没有第二套质量排名、GT推理门控、候选裁剪、教师、新监督或候选集合自注意力。96672参数/23张量为手工计算，尚未真实构建测量。

仅执行Python3.7 AST及文件摘要检查；未集成模型工厂、未GPU前向、未优化、无精度结果，也没有新的源码评审PASS。几何来源须依据当前训练终态选择；结构可见性与最终IoU监督应分别控制。原G的实际文本目标、父权重依赖及两阶段对象协议保留。未来部署前仍需独立源码门禁与真实零残差等价、单次评分、梯度及保存恢复检查；分布熵不能直接称为校准质量或实例身份真值。
'''
new = old + section.encode('utf-8')
rule = b'\n# Preserve isolated final geometry readback draft bytes.\n' + prefix.encode() + b'** -text whitespace=cr-at-eol\n'
for repo in repos[:2]:
    assert prefix.encode() not in (repo / '.gitattributes').read_bytes()

client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
               password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
remote = '/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(remote + '/' + doc, 'rb') as stream:
    assert stream.read() == old
sftp.mkdir(remote + '/' + prefix.rstrip('/'))
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
assert all(path.read_bytes() == new for path in copies)
with sftp.open(remote + '/' + doc, 'rb') as stream:
    assert stream.read() == new
sftp.close()
client.close()

heads = []
for index, repo in enumerate(repos):
    if index < 2:
        attributes = repo / '.gitattributes'
        attributes.write_bytes(attributes.read_bytes() + rule)
        with (repo / 'MANIFEST.md').open('a', encoding='utf-8') as stream:
            stream.write(f'\n- {stamp} isolated final boundary evidence readback draft; AST-only, no factory/GPU/accuracy, active face fit unchanged.\n')
    stage = [doc] + (['MANIFEST.md', '.gitattributes', *payloads] if index < 2 else [])
    changed = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage])
    if index < 2:
        subprocess.check_call(['git', '-C', str(repo), 'add', '--renormalize', '--', prefix])
    subprocess.check_call(['git', '-C', str(repo), '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol',
                           'diff', '--cached', '--check', '--', *stage])
    for relative, raw in (payloads.items() if index < 2 else []):
        assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + relative]) == raw
    subprocess.check_call(['git', '-C', str(repo), 'commit', '-m', 'Prepare isolated final boundary evidence readback source draft'])
    heads.append(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip())
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
subprocess.check_call(['git', '-C', str(repos[0]), '-c', 'http.version=HTTP/1.1', 'push', 'origin', 'HEAD:main'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == heads[0]
digest = hashlib.sha256(new).hexdigest()
guard.write_bytes(guard_raw.replace(previous['handoff_sha256'].encode(), digest.encode()))
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), section='20.376.48',
    predecessor=str(face / 'launch_publication.json'), heads=heads, github_main=heads[0],
    handoff_sha256=digest, handoff_bytes=len(new), four_local_and_remote_equal=True,
    exact_committed_payloads=True, payload_count=len(payloads), execution_scope='LOCAL_AST_ONLY_UNINTEGRATED',
    native_factory_integrated=False, gpu_forwards=0, optimizer_updates=0, accuracy_result=False,
    active_face_fit_changed=False, retained_metric_best='plain_distribution_5616_4506', goal_achieved=False)
(local / 'source_publication.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
with (workspace / 'memory/2026-10-04.md').open('a', encoding='utf-8') as stream:
    stream.write(f'\nPVGround {record["time_cst"]} isolated final boundary readback AST-only source published {heads[0]}, doc48 fourlocal+remote equal; no factory/GPU/accuracy claim, active face fit unchanged, only observer84531. GoalACTIVE_UNMET.\n')
print(json.dumps(record), flush=True)
