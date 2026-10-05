"""Append only the reviewed new architecture and actual sanity launch."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import paramiko

local=Path(__file__).resolve().parent
workspace=Path('C:/Users/gb')
old_publication=json.loads((local.parent/'pvground_auxiliary_target_20261005/terminal_publication.json').read_bytes())
review=json.loads((local/'SOURCE_REVIEW.json').read_bytes())
call=json.loads((local/'SOURCE_REVIEW_CALL.json').read_bytes())
launch=json.loads((local/'preflight_launch.json').read_bytes())
assert review['verdict'] in ('PASS','WARN') and not review['blocking_findings'] and call['result_received']
assert review['execution_scope']=='SOURCE_ONLY' and launch['status']=='PREFLIGHT_LAUNCHED_NOT_COMPLETED'
assert not (local/'preflight_launch_publication.json').exists()
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
section=f'''

## 20.376.76 候选自身支撑参考：代码审阅完成，真实两步预检已启动（{stamp}）

上一轮原GT控制5616／4511已通过§75终态核对并保留；成员范围目标组5616／4502未晋级。新实验继续使用原GT，不再重复成员GT对照。ScanRefer原生Acc@0.50优先；全256候选及同Query Box／Mask保持。

新代码见 {prefix}support_reference.py、run_geometry_fit.py、两组spec及EXPERIMENT_PLAN.md。固定参考控制保留456102参数／10项几何状态；学习参考组只新增3078参数的512→6线性层，共459180参数／12项状态。输入分别保留几何Query288、自身Query Mask成员范围109、原生融合Mask范围109及场景归一化粗框6；两种Mask范围角色不混淆，不以公共Text支撑替代候选自身支撑。参考中心按原粗框尺度残差更新，尺寸使用expm1变化的加性形式保持零输出时原始粗尺寸。原先真实出现的尺寸下限仍按既有边界解码处理，没有新回退框。

学习参考改变七位置局部读取与原33节点六面分布的参考中心／尺度；最终仍输出一个框，原生bbs不变。旧几何十项状态逐项加载；新增参考层零初始化。固定PV/G、Mask、语义及全零R；不加入教师、质量排序、普通attention或第二套推理答案。

原native＋G＋匹配DFL、Query自身／融合支撑确认的额外几何监督保持。学习参考组额外监督其参考框L1/GIoU：原匹配仍负责真实匹配GT，额外资格集合仍负责root；原额外集合排除全部原匹配，资格停止梯度且每表达内平均后按batch平均。新增参考项独立记录。因此本实验同时增加结构与参考定位目标、参数量也不同，是整项改动对照，不能把结果单独归因于某一个输入或损失。

每组计划29778 fit行各一次、3723更新、物理／有效batch8、累积1、seed2027、LR1e-5、WD5e-4、clip0.1；均从同一4511权重重新初始化优化器。几何头历史11169次，下一正式终点累计14892次，原G适配历史另算。6887模块留出被作者预训练见过；正式9508开发验证仍为原生GT及augment=False。保存原生粗框、参考框、最终框，分别统计同Query作用及实际原生选择，内部修复不称baseline增益。

新上下文SOURCE审阅为 {review['verdict']}、阻断发现0；实际调用已返回，按same-family／provisional记录，后端模型身份未独立证实。证据见SOURCE_REVIEW.json／md及SOURCE_REVIEW_CALL.json。两步GPU预检实际于 {launch['time_cst']} 启动，进程 {launch['process']}，screen {launch['screen']}，目录 {launch['root']}。仅预检启动，不是预检通过，也没有新精度结果；不写权重、不承接预检优化器启动正式训练。预计约 {launch['estimated_seconds']} 秒，首次检查按 {launch['first_check_seconds']} 秒估计，之后240秒复查，使用已有GPU锁顺序运行。

预检将实际核对缓存同上游的零参考／旧头结果、参考及最终定位梯度、冻结父状态与Mask／bbs保持、真实保存恢复和显存。新增原生evaluator的每次导入路径／SHA收据，补足§75已记录的执行器绑定限制，不追溯修改旧结果。预检闭合通过后才正式启动；保持4511权重，完整正式终态后按Acc@0.50优先保留best和清理闭合非best。当前没有Nr3D／Sr3D新成绩，目标仍ACTIVE_UNMET。
'''
assert chr(65533) not in section
new=old+section.encode('utf-8')
assert new.startswith(old) and new.count(b'## 20.376.76 ')==1
names=[path.name for path in local.glob('*.py')]+[
    'EXPERIMENT_PLAN.md','control_spec.json','support_reference_spec.json','SOURCE_PACKET.json',
    'SOURCE_REVIEW.json','SOURCE_REVIEW.md','SOURCE_REVIEW_CALL.json','preflight_launch.json','resource_check.json']
payloads={prefix+name:(local/name).read_bytes() for name in names}
payloads[prefix+'.gitattributes']=b'** -text\n'
assert all('.aris' not in name and not name.endswith(('.pt','.pth')) for name in payloads)
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp();project='/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(project+'/'+doc,'rb') as stream:
    assert stream.read()==old
remote_directory=project+'/'+prefix.rstrip('/')
assert prefix.rstrip('/').split('/')[-1] not in sftp.listdir(project+'/refine-logs')
sftp.mkdir(remote_directory)
for name,raw in payloads.items():
    for repo in repos[:2]:
        destination=repo/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(raw)
    with sftp.open(project+'/'+name,'wx') as stream:
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
            stream.write('\n- '+stamp+' Own/fused support reference SOURCE review complete; actual two-step GPU sanity launched, no accuracy result.\n')
        stage += ['MANIFEST.md',*payloads]
    changed=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git','-C',str(repo),'add','-f','--',*stage],stderr=subprocess.DEVNULL)
    subprocess.check_call(['git','-C',str(repo),'-c','core.whitespace=cr-at-eol,-blank-at-eof','diff','--cached','--check','--',*stage])
    if index<2:
        assert all(subprocess.check_output(['git','-C',str(repo),'show',':'+name])==raw for name,raw in payloads.items())
    assert subprocess.check_output(['git','-C',str(repo),'show',':'+doc]).startswith(
        subprocess.check_output(['git','-C',str(repo),'show',old_publication['heads'][index]+':'+doc]))
    subprocess.check_call(['git','-C',str(repo),'commit','--quiet','-m','Prepare own-support reference and launch real GPU sanity'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'-c','http.version=HTTP/1.1','push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0]==heads[0]
digest=hashlib.sha256(new).hexdigest()
guard=workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py';raw=guard.read_bytes()
assert raw.count(old_publication['handoff_sha256'].encode())==1
guard.write_bytes(raw.replace(old_publication['handoff_sha256'].encode(),digest.encode()))
record={'time_cst':datetime.datetime.now().astimezone().isoformat(),'section':'20.376.76','heads':heads,
    'handoff_sha256':digest,'four_local_and_remote_equal':True,'github_main':heads[0],
    'preflight_launch':launch,'new_accuracy_result':False,'source_verdict':review['verdict']}
(local/'preflight_launch_publication.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
state_path=local.parent/'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state=json.loads(state_path.read_bytes())
state.update(time_cst=record['time_cst'],latest_publication=str(local/'preflight_launch_publication.json'),
    status='SUPPORT_REFERENCE_PREFLIGHT_LAUNCHED_PUBLISHED',owned_gpu_job_active=True,active_reviewer=None,
    support_reference_deployed=True,support_reference_preflight_complete=False,
    protected_best_hits=[5616,4511],handoff_section=record['section'],handoff_sha256=digest,published_heads=heads,
    next_action='Wait for the sole estimated preflight observer. Read actual primary logs if failed; collect successful closure, then launch the reviewed same-budget formal pair with fresh optimizer.')
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (workspace/'memory/2026-10-05.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+record['time_cst']+': Doc76 own/fused support reference SOURCE review and actual two-step preflight launch; no accuracy result. Fourlocal/remote equal, main '+heads[0]+'. Protect4511; source3078newparams; no teacher or new deployed ranking.\n')
print(json.dumps(record),flush=True)
