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
old_publication=json.loads((local/'preflight_launch_publication.json').read_bytes())
review=json.loads((local/'SOURCE_REVIEW.json').read_bytes())
call=json.loads((local/'SOURCE_REVIEW_CALL.json').read_bytes())
launch=json.loads((local/'fit_launch.json').read_bytes())
assert review['verdict'] in ('PASS','WARN') and not review['blocking_findings'] and call['result_received']
assert review['execution_scope']=='SOURCE_ONLY' and launch['status']=='TWO_ARM_FIT_LAUNCHED_NOT_COMPLETED'
assert not (local/'fit_launch_publication.json').exists()
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
proofs={arm:json.loads((local/'preflight_complete'/arm/'preflight.json').read_bytes()) for arm in ('control','support_reference')}
wait=json.loads((local/'preflight_wait.json').read_bytes())
assert wait['observer_closed'] and not wait['controller_alive'] and wait['exitcode']==0
for arm,proof in proofs.items():
    assert proof['status']=='pass' and proof['optimizer_steps']==2 and proof['weight_files_created']==0
    assert proof['all_parent_and_R_states_exact'] and proof['optimizer_exact_check']['all_keys_moments_steps_and_groups_exact']
assert proofs['support_reference']['witnesses'][0]['zero_reference_cached_old_head_exact']
assert all(value['reference_direct_geometry_gradient']>0 for value in proofs['support_reference']['witnesses'])
section=f'''

## 20.376.77 固定／自身支撑参考：两组真实预检通过，正式对照已启动（{stamp}）

§76所述参考结构保持。两组预检实际于 {wait['status']['finished_cst']} 闭合，控制器退出0，耗时 {wait['status']['elapsed_seconds']:.2f} 秒；各2次真实更新，不写权重。缓存同一上游时，学习参考零初始化与旧头输出一致；参考直接定位梯度两步均非零，原额外定位目标仍只对其资格输出产生直接梯度。头更新后的缓存原生bbs和Mask保持，PV/G及全零R状态保持，CPU保存／严格重载和优化器全部键、矩、步及组检查通过。这些是工程预检，不是新精度结果，也不证明跨进程CUDA前向逐位一致。

实际allocator分配峰值：control {proofs['control']['peak_allocated_bytes']} 字节，support_reference {proofs['support_reference']['peak_allocated_bytes']} 字节；仅代表冻结父模型的预检batch，不外推到全模型联合训练或任意场景。当前batch及学习率不变。

正式控制器实际于 {launch['time_cst']} 启动，进程 {launch['process']}，screen {launch['screen']}；采用同一受保护4511权重、fresh optimizer，预检状态不承接。每组29778条fit各一次／3723更新，B8、累积1、seed2027、LR1e-5、WD5e-4、clip0.1。父PV/G、Mask、语义及全零R冻结；控制456102参数／10状态，学习参考459180参数／12状态。旧十项几何状态累计11169→14892更新；新增3078参考参数只训练本轮3723步，原G历史另算。

保留各自初始／终点6887 seen-scene模块留出，之后分别9508条原生开发验证。新增参考定位目标的预算及梯度范数单独记录；结构、参数量与监督预算同时变化，不把它称作纯输入或单项loss消融。唯一bbs、全256、同Query Box／Mask、原GT评价保持，没有教师或第二套排名。

正式预计约 {launch['estimated_seconds']} 秒；首次按 {launch['first_check_seconds']} 秒、控制组终点附近安排，随后240秒轮询。只有一个fit观察者，不因未到估计时刻或观察超时重启GPU训练。启动前数据盘余量 {launch['resources']['data_free_bytes']} 字节、系统盘 {launch['resources']['system_free_bytes']} 字节，所需保存及日志储备 {launch['resources']['required_reserve_bytes']} 字节，GPU空闲及预检闭合已实际检查。

实际预检原始记录见 {prefix}preflight_complete/INTAKE.json及两组preflight.json；正式启动见fit_launch.json／fit_resource_check.json。结果仍待正式9508评估；当前best仍5616／4511、Acc@0.50约47.4443%，到50%还差243净命中。终态经独立核验后才晋级与删除闭合非best，不归档负权重；原PV/G/V99保留。未有Nr3D／Sr3D新结果，总目标ACTIVE_UNMET。
'''
assert chr(65533) not in section
new=old+section.encode('utf-8')
assert new.startswith(old) and new.count(b'## 20.376.77 ')==1
names=['fit_launch.json','fit_resource_check.json','preflight_wait.json']
names += [str(path.relative_to(local)).replace('\\','/') for path in sorted((local/'preflight_complete').rglob('*')) if path.is_file()]
names += ['postrun/publish_fit_launch.py','postrun/prepare_fit_publication.py','postrun/analyze_reference_formal.py']
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
            stream.write('\n- '+stamp+' Own/fused support reference two-step GPU sanity passed; same-start formal pair launched, no new accuracy result.\n')
        stage += ['MANIFEST.md',*payloads]
    changed=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git','-C',str(repo),'add','-f','--',*stage],stderr=subprocess.DEVNULL)
    subprocess.check_call(['git','-C',str(repo),'-c','core.whitespace=cr-at-eol,-blank-at-eof','diff','--cached','--check','--',*stage])
    if index<2:
        assert all(subprocess.check_output(['git','-C',str(repo),'show',':'+name])==raw for name,raw in payloads.items())
    assert subprocess.check_output(['git','-C',str(repo),'show',':'+doc]).startswith(
        subprocess.check_output(['git','-C',str(repo),'show',old_publication['heads'][index]+':'+doc]))
    subprocess.check_call(['git','-C',str(repo),'commit','--quiet','-m','Record passed reference sanity and launch paired formal comparison'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'-c','http.version=HTTP/1.1','push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0]==heads[0]
digest=hashlib.sha256(new).hexdigest()
guard=workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py';raw=guard.read_bytes()
assert raw.count(old_publication['handoff_sha256'].encode())==1
guard.write_bytes(raw.replace(old_publication['handoff_sha256'].encode(),digest.encode()))
record={'time_cst':datetime.datetime.now().astimezone().isoformat(),'section':'20.376.77','heads':heads,
    'handoff_sha256':digest,'four_local_and_remote_equal':True,'github_main':heads[0],
    'fit_launch':launch,'new_accuracy_result':False,'source_verdict':review['verdict']}
(local/'fit_launch_publication.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
state_path=local.parent/'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state=json.loads(state_path.read_bytes())
state.update(time_cst=record['time_cst'],latest_publication=str(local/'fit_launch_publication.json'),
    status='SUPPORT_REFERENCE_FORMAL_FIT_ACTIVE_PUBLISHED',owned_gpu_job_active=True,active_reviewer=None,
    support_reference_deployed=True,support_reference_preflight_complete=True,
    support_reference_preflight_observer_closed=True,support_reference_preflight_observer_session_id=None,
    support_reference_fit_observer_session_id=40310,support_reference_fit_observer_closed=False,
    protected_best_hits=[5616,4511],handoff_section=record['section'],handoff_sha256=digest,published_heads=heads,
    next_action='Keep sole fit observer40310 until estimated closure; first check2026-10-06 01:26:31 CST, later240s. Do not restart controller653283 or create another observer. Collect both actual9508 formal rows only after closure, fresh terminal audit, then strict-best retention and publication.')
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (workspace/'memory/2026-10-05.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+record['time_cst']+': Doc77 own/fused support reference actual two-step sanity passed and formal pair launched; no new accuracy result. Fourlocal/remote equal, main '+heads[0]+'. Protect4511; source3078newparams; no teacher or new deployed ranking.\n')
print(json.dumps(record),flush=True)
