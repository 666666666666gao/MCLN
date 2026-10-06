"""Append only reference-preservation source and actual ScanRefer M0 launch to Doc88."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

import paramiko

root=Path(__file__).resolve().parent
assert not (root/'preflight_publication.json').exists()
review=json.loads((root/'PUBLISH_PREFLIGHT_SOURCE_REVIEW.json').read_bytes())
assert review['execution_scope']=='SOURCE_ONLY' and review['verdict'] in ('PASS','WARN') and not review['blocking_findings']
for item in review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
source_review=json.loads((root/'SOURCE_REVIEW.json').read_bytes())
assert source_review['execution_scope']=='SOURCE_ONLY' and source_review['verdict'] in ('PASS','WARN') and not source_review['blocking_findings']
for item in source_review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
launch=json.loads((root/'preflight_launch.json').read_bytes())
assert launch['accuracy_result'] is False and launch['optimizer_steps_planned_per_arm']==2
assert launch['root']=='/root/autodl-tmp/pvground_reference_keep_20261006'
previous_path=root.parent/'pvground_referit_mask_reference_20261006/comparison_publication.json'
previous=json.loads(previous_path.read_bytes());assert previous['section']=='20.376.88'
state_path=root.parent/'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state=json.loads(state_path.read_bytes());assert Path(state['latest_publication']).resolve()==previous_path.resolve()
workspace=Path('C:/Users/gb')
repos=[workspace/'.codex_mcln_g0_20260905',workspace/'.codex_pvground_cs_20261002',workspace/'.codex_mcln_v99_internal_20260928']
doc='docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies=[repo/doc for repo in repos]+[workspace/'Desktop/document'/Path(doc).name]
old=copies[0].read_bytes();assert hashlib.sha256(old).hexdigest()==previous['handoff_sha256']
assert all(path.read_bytes()==old for path in copies)
for repo,head in zip(repos,previous['heads']):
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==head
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
stamp=datetime.datetime.now().astimezone().isoformat()
prefix='refine-logs/pvground_reference_keep_20261006/preflight_registration/'
section=f'''

## 20.376.89 ScanRefer参考保持训练约束：源码通过与真实两组预检启动（{stamp}）

§88新用户目标不变：同一完整9508模型至少5620/4764，三个经直接消融支持的有效模块后再固定结构、从对应作者预训练权重独立训练Sr/Nr；single seed2027。当前保护5598/4848零更新Mask参考模型，不把预检或零输出分支算有效学习贡献。Nr/Sr正式训练未启动。

当前唯一新增方向是训练期参考相对退化约束，不增加推理参数、排序源或Mask/文本更新。两组严格从保留initial.pth完整10项几何状态＋官方PV＋原G重建，复用实际1304状态CPU重建证明，无需已清理的4511权重。相同456102几何参数、融合Mask空间参考、全局109维/局部7×16支撑、六面33节点分布，父模型/Mask/语言/原生语义/全零R冻结；全部256候选，一套last/bbs、同QueryBoxMask。原native/G/匹配DFL及既有额外几何项不改。

控制λ=0、策略λ=1；可信集合内计算squared ReLU(stopgrad(IoU(reference,GT))-IoU(final,GT))，先表达内平均再实际batch平均。原匹配查询对应其原生过滤后真实GT；额外查询须自身Query和融合Mask对root均IoU>.5，排除全部原匹配，不以Box是否刚过.5截断保持责任。资格只用于训练，不输入模型/推理。该项允许有益修正，惩罚相对参考的几何退化；目前只是待验证训练机制，不保证精度或参考一定不被破坏。

fresh SOURCE_ONLY审查已完成、0剩余阻断，same-family/provisional、backend与runtime未认证。发现并最小修正真实CPU/CUDA接口错误：原HungarianMatcher返回CPU int64索引，新增Mask资格索引在CUDA，直接cat会失败；只在新增函数内显式将matched_queries/targets移至boxes.device，原匹配/旧loss不变。原finding和修正保留。

真实M0两组已于{launch['time_cst']}由唯一控制器{launch['controller_pid']}启动，每组真实增强batch8两步更新、seed2027/lr1e-5/WD.0005/clip.1，更新不承接正式训练，不保存预检权重。待核对初始中性解码/保持损失严格0、实际新项梯度资格范围、nativebbs/Mask与父状态、几何梯度和Adam内存CPU重载。当前仅launch证据，不写预检通过或新精度。

启动资源：GPU实际空闲、保护4848权重SHA核对；数据盘{launch['resource']['data_free_bytes']}字节，系统盘{launch['resource']['system_free_bytes']}字节。复用既有A10040GB/Torch1.10.2环境，不安装新包。上一同构M0单组470秒，本轮两组约16～20分钟；首次启动720秒后查看、后续240秒或已完成阶段实测估计，不提前查询或另起观察器。后续完整pair约7小时，正式前另核对真实保存/评估空间。

通过M0才拟进行两组同起点/同预算29778条fit一次、3723更新（B8/累积1）与初始/终点6887模块留出及完整9508原生验证。模块留出场景被作者预训练见过；只超过历史4832终点不足以证明新增学习，要同时比本轮控制与4848起点看修复/破坏，最终核对5620/4764。当前0新正式结果、三模块证据未成立，不调seed、不同时加边界新采样/attention/教师/质量头。源码和方案见{prefix}；权重/原始NPZ/凭据/.aris不发布。
'''
new=old+section.encode('utf-8');assert new.startswith(old) and new.count(b'## 20.376.89 ')==1
names=['EXPERIMENT_PLAN.md','EXPERIMENT_TRACKER.md','research_contract.md','control_spec.json','keep_spec.json',
    'preflight_launch.json','resource_check.json','SOURCE_REVIEW.json','SOURCE_REVIEW.md',
    'reference_keep.py','run_reference_keep_fit.py','selected_mask_reference_factory.py',
    'query_supported_geometry.py','mask_reference.py','check_invalid_reference.py','invalid_reference_fixture.json',
    'controller.py','deploy_preflight_authorized.py','observe_preflight_authorized.py','prepare_reference_keep.py',
    'prepare_preflight_publication.py','publish_preflight_authorized.py',
    'PUBLISH_PREFLIGHT_SOURCE_REVIEW.json','PUBLISH_PREFLIGHT_SOURCE_REVIEW.md']
payloads={prefix+name:(root/name).read_bytes() for name in names};payloads[prefix+'.gitattributes']=b'** -text\n'
assert all(not name.endswith(('.pt','.pth','.npz')) and '.aris' not in name for name in payloads)
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
spec=json.loads((root/'control_spec.json').read_bytes())
code='''import base64,hashlib,json,sys
from pathlib import Path
project=Path(sys.argv[1]);b=json.load(sys.stdin);doc=project/b['doc'];old=doc.read_bytes()
assert hashlib.sha256(old).hexdigest()==b['old_sha256']
evidence=(project/b['prefix']).resolve()
assert str(evidence)=='/root/autodl-tmp/mcln_published_evidence_20261005/pvground_reference_keep_20261006/preflight_registration'
assert not evidence.exists()
for name,encoded in b['files'].items():
    assert name.startswith(b['prefix']) and not name.endswith(('.pt','.pth','.npz')) and '.aris' not in name
    p=project/name;assert evidence in p.resolve().parents;p.parent.mkdir(parents=True,exist_ok=True)
    raw=base64.b64decode(encoded)
    with p.open('xb') as f:f.write(raw)
    assert p.read_bytes()==raw
new=base64.b64decode(b['new_doc']);assert new.startswith(old);doc.write_bytes(new)
assert doc.read_bytes()==new
print(json.dumps(dict(files=len(b['files']),handoff_sha256=hashlib.sha256(new).hexdigest())))
'''
bundle=dict(doc=doc,prefix=prefix,old_sha256=previous['handoff_sha256'],new_doc=base64.b64encode(new).decode(),
    files={name:base64.b64encode(raw).decode() for name,raw in payloads.items()})
stdin,stdout,stderr=client.exec_command(shlex.join([spec['runtime']+'/venv/bin/python','-B','-c',code,
    '/home/gb/new butd/butd_detr-main/MCLN-main']),timeout=180)
stdin.write(json.dumps(bundle).encode());stdin.flush();stdin.channel.shutdown_write()
raw=stdout.read();assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
remote=json.loads(raw);client.close()
for name,raw in payloads.items():
    for repo in repos[:2]:
        path=repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
for path in copies:path.write_bytes(new)
heads=[]
for index,repo in enumerate(repos):
    stage=[doc]
    if index<2:
        with (repo/'MANIFEST.md').open('a',encoding='utf-8') as stream:
            stream.write('\n- '+stamp+' Reference-preservation-only ScanRefer loss contrast actual M0 launched; no new formal metrics.\n')
        stage+=['MANIFEST.md',*payloads]
    changed=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git','-C',str(repo),'add','-f','--',*stage],stderr=subprocess.DEVNULL)
    subprocess.check_call(['git','-C',str(repo),'-c','core.whitespace=cr-at-eol,-blank-at-eof','diff','--cached','--check','--',*stage])
    if index<2:
        assert all(subprocess.check_output(['git','-C',str(repo),'show',':'+name])==raw for name,raw in payloads.items())
    assert subprocess.check_output(['git','-C',str(repo),'show',':'+doc]).startswith(
        subprocess.check_output(['git','-C',str(repo),'show',previous['heads'][index]+':'+doc]))
    subprocess.check_call(['git','-C',str(repo),'commit','--quiet','-m','Register ScanRefer reference-preservation loss and real M0 launch'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'-c','http.version=HTTP/1.1','push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0]==heads[0]
digest=hashlib.sha256(new).hexdigest();assert digest==remote['handoff_sha256'] and all(p.read_bytes()==new for p in copies)
guard=workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py';raw=guard.read_bytes()
assert raw.count(previous['handoff_sha256'].encode())==1
guard.write_bytes(raw.replace(previous['handoff_sha256'].encode(),digest.encode()))
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.89',heads=heads,
    handoff_sha256=digest,four_local_and_remote_equal=True,github_main=heads[0],payload_count=len(payloads),
    execution_scope='REFERENCE_KEEP_M0_LAUNCH_NOT_PASSED',new_formal_result=False,
    controller_pid=launch['controller_pid'],raw_npz_published=False,weights_published=False)
(root/'preflight_publication.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
state.update(time_cst=record['time_cst'],latest_publication=str(root/'preflight_publication.json'),
    handoff_section=record['section'],handoff_sha256=digest,published_heads=heads,
    reference_keep_preflight_launch=str(root/'preflight_launch.json'),reference_keep_preflight_controller_pid=launch['controller_pid'],
    owned_gpu_job_active=True,overall_goal_complete=False,next_action='Wait until00:16:49.814 CST original32309 first M0 observation; Scan first, Nr/Sr deferred')
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (workspace/'memory/2026-10-06.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPV-Ground '+record['time_cst']+': Doc89 reference-preservation λ0/1 SOURCEPASS and actualM0 controller'+str(launch['controller_pid'])+'; current5598/4848 protected, no newformal result. Main '+heads[0]+'. Single2027; Nr/Srdeferred untilScan5620/4764+threeeffective modules. First sole32309 check00:16:49.814, noearlyNNquery.\n')
print(json.dumps(record),flush=True)
