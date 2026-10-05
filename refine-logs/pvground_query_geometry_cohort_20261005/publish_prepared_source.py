import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

import paramiko

root=Path(__file__).resolve().parent
study=root.parent/'pvground_query_supported_geometry_20261005'
state_path=study/'active_continuation_state.json'
state=json.loads(state_path.read_bytes())
previous=json.loads(Path(state['latest_publication']).read_bytes())
assert previous['section']=='20.376.65'
assert not (root/'source_publication.json').exists()
assert not (root/'spec.json').exists() and not (root/'launch.json').exists()
repos=[Path(r'C:\Users\gb\.codex_mcln_g0_20260905'),
    Path(r'C:\Users\gb\.codex_pvground_cs_20261002'),Path(r'C:\Users\gb\.codex_mcln_v99_internal_20260928')]
relative='refine-logs/pvground_query_geometry_cohort_20261005'
doc='docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'

def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()

for repo,head in zip(repos,previous['heads']):
    assert git(repo,'rev-parse','HEAD')==head
    assert not git(repo,'status','--porcelain')
    assert hashlib.sha256((repo/doc).read_bytes()).hexdigest()==previous['handoff_sha256']
desktop=Path(r'C:\Users\gb\Desktop\document\MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md')
assert hashlib.sha256(desktop.read_bytes()).hexdigest()==previous['handoff_sha256']
payload=[p for p in root.iterdir() if p.suffix=='.py']
payload += [root/name for name in ('GENERATION.json','PREPARATION.json','METRIC_PRIMITIVE_CHECK.json','EXPERIMENT_PLAN.md')]
assert all(p.is_file() for p in payload)
for repo in repos[:2]:
    destination=repo/relative
    assert not destination.exists()
    destination.mkdir()
    for path in payload:
        (destination/path.name).write_bytes(path.read_bytes())

client=paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',
    password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp()
project='/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(project+'/'+doc,'rb') as stream:
    assert hashlib.sha256(stream.read()).hexdigest()==previous['handoff_sha256']
remote=project+'/'+relative
sftp.mkdir(remote)
for path in payload:
    raw=path.read_bytes()
    with sftp.open(remote+'/'+path.name,'wx') as stream:
        stream.write(raw)
    with sftp.open(remote+'/'+path.name,'rb') as stream:
        assert stream.read()==raw
sftp.close()
client.close()
for repo in repos[:2]:
    git(repo,'add','--',relative)
    git(repo,'commit','-m','Prepare fixed-cohort geometry comparison after paired fit')
git(repos[0],'push','origin','HEAD:main')
heads=[git(repo,'rev-parse','HEAD') for repo in repos]
github=git(repos[0],'ls-remote','origin','refs/heads/main').split()[0]
assert github==heads[0]
for repo in repos:
    assert not git(repo,'status','--porcelain')
    assert hashlib.sha256((repo/doc).read_bytes()).hexdigest()==previous['handoff_sha256']
record=dict(previous)
record.update(time_cst=datetime.datetime.now().astimezone().isoformat(),heads=heads,github_main=github,
    status='PREPARED_DIAGNOSTIC_SOURCE_PUBLISHED',payload_count=len(payload),
    source='Local prepared fixed-cohort scripts and prior-data CPU metric check only',
    new_accuracy_result=False,diagnostic_GPU_executed=False,diagnostic_source_review_pending=True,
    closed_pair_spec_binding_pending=True,cleanup_executed=False,full_goal_status='ACTIVE_UNMET')
(root/'source_publication.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
state.update(latest_publication=str(root/'source_publication.json'),published_heads=heads,
    cohort_prepared_source_published=True,cohort_source_review_pending=True,
    current_goal_turn_classification='PROGRESS_COHORT_PREPARED_SOURCE_PUBLISHED')
state_path.write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
memory=Path(r'C:\Users\gb\memory\2026-10-05.md')
with memory.open('a',encoding='utf-8') as stream:
    stream.write('\n\n### '+record['time_cst']+' 固定候选终点对照准备\n'
        '准备复用64条增强fit输入，对父4506/control/query_supported几何头作同上游只读对照，补足本轮对应候选改善证据；'
        '每批一次完整父前向、三个缓存几何头调用，无优化/新增权重。源码已同步MAIN/PV及远端项目并推送，交接仍§65原字节。'
        '尚无两组闭组spec/新鲜源码审查/GPU诊断，必须在两组终态收集后审查、GPU空闲后执行，并在当前非最佳权重清理前完成。'
        'CPU工具仅以既有B面板16384候选核对，不能算新策略结果。MAIN '+heads[0]+'。\n')
print(json.dumps({key:record[key] for key in ('time_cst','section','heads','github_main','handoff_sha256',
    'status','payload_count','new_accuracy_result','diagnostic_GPU_executed','diagnostic_source_review_pending')}))
