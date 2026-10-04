"""Publish the implemented post-terminal analysis without querying active training."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import paramiko

local = Path(__file__).resolve().parent
formal = local / 'formal_draft'
face = local.parent / 'pvground_face_conditioned_20261004'
receipt_path = local / 'terminal_analysis_preparation_publication.json'
assert not receipt_path.exists()
previous = json.loads((local / 'revision2_formal_launch_publication.json').read_bytes())
prepared = json.loads((formal / 'ANALYSIS_PREPARATION.json').read_bytes())
source = (formal / 'analyze_closed_formal.py').read_bytes()
assert prepared['status'] == 'IMPLEMENTED_AST37_ONLY_NOT_EXECUTED'
assert hashlib.sha256(source).hexdigest() == prepared['source_sha256']
workspace = Path('C:/Users/gb')
repos = [workspace / '.codex_mcln_g0_20260905', workspace / '.codex_pvground_cs_20261002',
    workspace / '.codex_mcln_v99_internal_20260928']
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies = [repo / doc for repo in repos] + [workspace / 'Desktop/document' / Path(doc).name]
old = copies[0].read_bytes()
assert hashlib.sha256(old).hexdigest() == previous['handoff_sha256']
assert all(path.read_bytes() == old for path in copies)
for repo, head in zip(repos, previous['heads']):
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip() == head
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
prefix = 'refine-logs/pvground_geometry_readback_20261004/'
payloads = {prefix+'formal_draft/analyze_closed_formal.py':source,
    prefix+'formal_draft/ANALYSIS_PREPARATION.json':(formal/'ANALYSIS_PREPARATION.json').read_bytes(),
    prefix+'publish_terminal_analysis_preparation.py':Path(__file__).read_bytes()}
stamp = datetime.datetime.now().astimezone().isoformat()
section = f'''

## 20.376.55 回读对照的终态分析工具已实现，尚未执行结果计算（{stamp}）

`formal_draft/analyze_closed_formal.py`已实现并通过Python3.7 AST检查，尚未读取新正式结果或执行推理。工具只在原控制器和观察器真实关闭、日志收集完成后运行：检查两组训练实际行顺序、29778条各一次/3723次更新、原生9508行及保存恢复记录；分别汇总可见对隐藏、相对保护4506父模型、同帧R前后语义回放的修复与破坏。记录全256候选资格及仅离线使用的GT体积四分位，不把GT诊断作为推理输入，也不把同框回放当作独立训练消融。只生成JSON/CSV/分析文本，不加载权重、重跑推理或优化。

本轮实际检查原生本地观察器66222仍在运行，其首次远端检查仍为2026-10-05 01:02:12 CST，随后240秒；没有提前SSH训练查询、重启控制器或调整活动训练。此时暂无R正式精度，§54的正式启动与预算继续有效；新的完整性审查只在真实终态结果产生后进行。该来源工具不构成运行通过或指标提升。
'''
new = old + section.encode('utf-8')
assert new.startswith(old) and new.count(b'## 20.376.55 ') == 1
guard = workspace / '.codex/tmp/sync_cs_handoff_remote_20260923.py'
guard_raw = guard.read_bytes()
assert guard_raw.count(previous['handoff_sha256'].encode()) == 1
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp = client.open_sftp()
remote = '/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(remote+'/'+doc,'rb') as stream:
    assert stream.read() == old
for relative, raw in payloads.items():
    for repo in repos[:2]:
        (repo/relative).write_bytes(raw)
    with sftp.open(remote+'/'+relative,'wb') as stream:
        stream.write(raw)
    with sftp.open(remote+'/'+relative,'rb') as stream:
        assert stream.read() == raw
for path in copies:
    path.write_bytes(new)
with sftp.open(remote+'/'+doc,'wb') as stream:
    stream.write(new)
with sftp.open(remote+'/'+doc,'rb') as stream:
    assert stream.read() == new
sftp.close()
client.close()
heads = []
for index, repo in enumerate(repos):
    stage = [doc]
    if index < 2:
        with (repo/'MANIFEST.md').open('a',encoding='utf-8') as stream:
            stream.write(f'\n- {stamp} closed-formal native-bbs analysis implemented/AST37 only; no new result execution, active readback controller unchanged.\n')
        stage += ['MANIFEST.md',*payloads]
    changed = subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git','-C',str(repo),'add','-f','--',*stage])
    subprocess.check_call(['git','-C',str(repo),'diff','--cached','--check','--',*stage])
    for relative, raw in (payloads.items() if index < 2 else []):
        assert subprocess.check_output(['git','-C',str(repo),'show',':'+relative]) == raw
    subprocess.check_call(['git','-C',str(repo),'commit','--quiet','-m','Prepare actual closed-formal readback analysis without replaying training'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'-c','http.version=HTTP/1.1','push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0] == heads[0]
digest = hashlib.sha256(new).hexdigest()
guard.write_bytes(guard_raw.replace(previous['handoff_sha256'].encode(),digest.encode()))
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.55',
    predecessor=str(local/'revision2_formal_launch_publication.json'),heads=heads,github_main=heads[0],
    handoff_sha256=digest,handoff_bytes=len(new),four_local_and_remote_equal=True,
    exact_committed_payloads=True,payload_count=3,analysis_implemented=True,analysis_executed=False,
    formal_accuracy_result=False,observer_session=66222,goal_achieved=False)
receipt_path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
state_path = face/'active_continuation_state.json'
state = json.loads(state_path.read_bytes())
state.update(latest_publication=str(receipt_path),github_publication_predecessor=str(receipt_path),published_heads=heads,
    handoff_sha256=digest,handoff_section='20.376.55',terminal_analysis_source=str(formal/'analyze_closed_formal.py'),
    terminal_analysis_executed=False,readback_formal_result_available=False)
state_path.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
cursor = f'\nLatestpublication {receipt_path}/MAIN{heads[0]}, doc55/{len(new)}B/SHA{digest}. Fourlocal+remoteexact/3sourcepayloads. Analysis implemented ASTonly, notexecuted; soleobserver66222 first01:02:12Oct5. Neverreplaycompletedpublishers. GoalACTIVE_UNMET.\n'
for path in (face/'NEXT_CONTINUATION.md',local/'NEXT_CONTINUATION.md'):
    with path.open('a',encoding='utf-8') as stream:
        stream.write(cursor)
with (workspace/'memory/2026-10-04.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+record['time_cst']+cursor)
print(json.dumps(record),flush=True)
