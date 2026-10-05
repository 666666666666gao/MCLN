"""Append only the reviewed new architecture and actual sanity launch."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import paramiko

local=Path(__file__).resolve().parents[1]
workspace=Path('C:/Users/gb')
old_publication=json.loads((local/'fit_launch_publication.json').read_bytes())
review=json.loads((local/'RETENTION_REVIEW.json').read_bytes())
call=json.loads((local/'RETENTION_REVIEW_CALL.json').read_bytes())
launch=json.loads((local/'fit_launch.json').read_bytes())
assert review['verdict'] in ('PASS','WARN') and not review['blocking_findings'] and call['result_received']
assert review['execution_scope']=='SOURCE_ONLY' and launch['status']=='TWO_ARM_FIT_LAUNCHED_NOT_COMPLETED'
assert not (local/'retention_tools_publication.json').exists()
repos=[workspace/'.codex_mcln_g0_20260905',workspace/'.codex_pvground_cs_20261002',workspace/'.codex_mcln_v99_internal_20260928']
doc='docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies=[repo/doc for repo in repos]+[workspace/'Desktop/document/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md']
old=copies[0].read_bytes()
assert hashlib.sha256(old).hexdigest()==old_publication['handoff_sha256']
assert all(path.read_bytes()==old for path in copies)
for index,repo in enumerate(repos):
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==old_publication['heads'][index]
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
stamp=datetime.datetime.now().astimezone().isoformat()
prefix='refine-logs/pvground_support_reference_20261005/'
for entry in review['reviewed_files']:
    assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()==entry['sha256'], entry['path']
assert not (local/'weight_retention.json').exists()
section=f"""

## 20.376.78 支撑参考对照的终态核算与最佳权重清理工具审阅（{stamp}）

当前正式对照仍使用§77发布的配置与原控制器653283；唯一观察者40310首次远端检查安排在2026-10-06 01:26:31，其后240秒轮询。本节没有新增正式精度，也没有执行终态收集或删除权重。当前受保护最佳仍为5616／4511，Acc@0.50约47.4443%，距离50%还差243条命中。

终态工具已准备并完成SOURCE_ONLY代码审阅，结论为{review['verdict']}，无阻断项。实际审阅记录见 {prefix}RETENTION_REVIEW.json／md 及调用回执；这是同系列审阅、暂定接受，不是独立后端身份认证，也不是训练结果验收。分析工具将核对两组完整9508行身份与独立CPU几何阈值、原粗框→支撑参考→最终框的内部修复／破坏、同预算两组实际选择及训练顺序。当前尚未产生这些终态分析结果。

清理入口仅在实际控制器闭合、完整结果核算与新终态完整性审阅通过后运行。它核验旧10项／456102参数头或新增12项／459180参数头及优化器终态，按Acc@0.50选优；严格指标与父4511持平时保留父权重。删除范围限于本轮两份终点与受保护父几何头三条明确路径中的非最佳文件；原PV/G、V99历史链、活动恢复状态与全部源码／日志／指标保留，不为负结果另建权重归档。工具已准备不代表清理已经发生，也不代表新参考结构已有效。
"""
assert chr(65533) not in section
new=old+section.encode('utf-8')
assert new.startswith(old) and new.count(b'## 20.376.78 ')==1
names=['RETENTION_REVIEW.json','RETENTION_REVIEW.md','RETENTION_REVIEW_CALL.json',
    'postrun/analyze_reference_formal.py','postrun/collect_formal_authorized.py',
    'postrun/retain_metric_best.py','postrun/retain_metric_best_authorized.py',
    'postrun/prepare_closed_tools.py','postrun/prepare_retention_publication.py',
    'postrun/publish_retention_tools.py']
payloads={prefix+name:(local/name).read_bytes() for name in names}
payloads[prefix+'.gitattributes']=b'** -text\n'
assert all('.aris' not in name and not name.endswith(('.pt','.pth')) for name in payloads)
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp();project='/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(project+'/'+doc,'rb') as stream:
    assert stream.read()==old
remote_directory=project+'/'+prefix.rstrip('/')
assert prefix.rstrip('/').split('/')[-1] in sftp.listdir(project+'/refine-logs')
directories={str(Path(name).parent).replace('\\','/') for name in names if '/' in name}
for relative in sorted(directories,key=lambda value:(value.count('/'),value)):
    parts=relative.split('/')
    for count in range(1,len(parts)+1):
        directory='/'.join(parts[:count]);parent,_,basename=directory.rpartition('/')
        if basename not in sftp.listdir(remote_directory+('/'+parent if parent else '')):
            sftp.mkdir(remote_directory+'/'+directory)
for name,raw in payloads.items():
    for repo in repos[:2]:
        destination=repo/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(raw)
    with sftp.open(project+'/'+name,'wb') as stream:
        stream.write(raw)
    with sftp.open(project+'/'+name,'rb') as stream:
        assert stream.read()==raw
for path in copies:
    path.write_bytes(new)
with sftp.open(project+'/'+doc,'wb') as stream:
    stream.write(new)
with sftp.open(project+'/'+doc,'rb') as stream:
    assert stream.read()==new
sftp.close();client.close()
heads=[]
for index,repo in enumerate(repos):
    stage=[doc]
    if index<2:
        with (repo/'MANIFEST.md').open('a',encoding='utf-8') as stream:
            stream.write('\n- '+stamp+' Support-reference closed-result and strict-best retention tools SOURCE reviewed; formal fit unchanged, tools not executed.\n')
        stage += ['MANIFEST.md',*payloads]
    changed=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git','-C',str(repo),'add','-f','--',*stage],stderr=subprocess.DEVNULL)
    subprocess.check_call(['git','-C',str(repo),'-c','core.whitespace=cr-at-eol,-blank-at-eof','diff','--cached','--check','--',*stage])
    if index<2:
        assert all(subprocess.check_output(['git','-C',str(repo),'show',':'+name])==raw for name,raw in payloads.items())
    assert subprocess.check_output(['git','-C',str(repo),'show',':'+doc]).startswith(
        subprocess.check_output(['git','-C',str(repo),'show',old_publication['heads'][index]+':'+doc]))
    subprocess.check_call(['git','-C',str(repo),'commit','--quiet','-m','Record reviewed reference closeout and strict-best retention tools'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'-c','http.version=HTTP/1.1','push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0]==heads[0]
digest=hashlib.sha256(new).hexdigest()
guard=workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py';raw=guard.read_bytes()
assert raw.count(old_publication['handoff_sha256'].encode())==1
guard.write_bytes(raw.replace(old_publication['handoff_sha256'].encode(),digest.encode()))
record={'time_cst':datetime.datetime.now().astimezone().isoformat(),'section':'20.376.78','heads':heads,
    'handoff_sha256':digest,'four_local_and_remote_equal':True,'github_main':heads[0],
    'fit_launch':launch,'new_accuracy_result':False,'source_verdict':review['verdict']}
(local/'retention_tools_publication.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
state_path=local.parent/'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state=json.loads(state_path.read_bytes())
state.update(time_cst=record['time_cst'],latest_publication=str(local/'retention_tools_publication.json'),
    status='SUPPORT_REFERENCE_FORMAL_FIT_ACTIVE_PUBLISHED',owned_gpu_job_active=True,active_reviewer=None,
    support_reference_retention_source_review_pending=False,support_reference_retention_source_review_complete=True,
    support_reference_postrun_tools_prepared=True,support_reference_postrun_tools_executed=False,
    support_reference_deployed=True,support_reference_preflight_complete=True,
    support_reference_preflight_observer_closed=True,support_reference_preflight_observer_session_id=None,
    support_reference_fit_observer_session_id=40310,support_reference_fit_observer_closed=False,
    protected_best_hits=[5616,4511],handoff_section=record['section'],handoff_sha256=digest,published_heads=heads,
    next_action='Keep sole fit observer40310 until estimated closure; first check2026-10-06 01:26:31 CST, later240s. Do not restart controller653283 or create another observer. Collect both actual9508 formal rows only after closure, fresh terminal audit, then strict-best retention and publication.')
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (workspace/'memory/2026-10-06.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+record['time_cst']+': Doc78 closed-result/strict-best retention tools SOURCE reviewed and published, not executed; active formal fit unchanged and no new accuracy result. Fourlocal/remote equal, main '+heads[0]+'. Protect4511; source3078newparams; no teacher or new deployed ranking.\n')
print(json.dumps(record),flush=True)
