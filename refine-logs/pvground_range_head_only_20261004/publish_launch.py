"""Publish actual frozen-protocol sanity and verified bounded-pair launch."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import paramiko

local=Path(__file__).parent
assert not (local/'launch_publication.json').exists()
previous=json.loads((local.parent/'pvground_whole_mask_fit_20261003/terminal_publication.json').read_bytes())
engineering=json.loads((local/'preflight_analysis.json').read_bytes())
launch=json.loads((local/'launch.json').read_bytes())
review=json.loads((local/'EXPERIMENT_CODE_REVIEW.json').read_bytes())
assert engineering['engineering_status']=='PASS' and engineering['original_g_state_unchanged']
assert engineering['optimizer_steps']==2 and engineering['weights_created']==0
assert review['verdict']=='PASS' and not review['blocking_findings']
assert launch['head_only'] and launch['formal_training_started'] and not launch['formal_result_available']
workspace=Path('C:/Users/gb')
repos=[workspace/'.codex_mcln_g0_20260905',workspace/'.codex_pvground_cs_20261002',workspace/'.codex_mcln_v99_internal_20260928']
doc='docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies=[repo/doc for repo in repos]+[workspace/'Desktop/document'/Path(doc).name]
old=copies[0].read_bytes()
assert hashlib.sha256(old).hexdigest()==previous['handoff_sha256']
assert all(path.read_bytes()==old for path in copies)
for repo,head in zip(repos,previous['heads']):
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==head
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
guard=workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py';guard_raw=guard.read_bytes()
assert guard_raw.count(previous['handoff_sha256'].encode())==1
stamp=datetime.datetime.now().astimezone().isoformat()
section='''

## 20.376.40 固定原G、仅训练既有范围头：真实检查闭环及有限对照启动（%s）

上一来源对照已完整闭环：local9508 bbs5603／4428、whole5594／4461；whole比local严格+33但仍比原G−34，同已选Query最后精修严格4471→4461。遵循已接受的条件分支，先分离新头学习与共同适配，再发展六面边界，不直接堆第三个质量模块。

新协议不重复以前只冻结视觉骨干的E：严格重建官方PV＋原G，所有原G参数和持久运行buffer固定，原模块保持eval；只训练现有candidate_box_refiner十项400614参数。原G语义／Mask输出和109范围都是冻结观察，故本控制有意不保留“框损失→Mask”的更新通路。模型结构、109范围定义、local成员读取、原生BBox／GIoU、256候选、原生bbs评分未改，没有新loss／教师／P2／多正例／分布头／双源／GT推理Gate。

新鲜gpt-6-astra／max／forknone源码审查实际PASS，0阻断、0非阻断；含新两步controller、正式启动门槛、保留逻辑及观察／收取代码，55份已读文件身份核对。same-family／provisional，未独立验证后端SKU；源码审查不被冒称GPU验收。

真实预检controller449570于%s启动，仅whole_range一个真实batch8重复两次。已正常闭环：4次原生完整forward，另有缓存head来源开／关重放；零头同缓存local／whole与重置相同RNG的有头／无头输出均精确保持。实际两步都有原生定位输出层梯度，第二步十项头参数任务梯度均>0；所有原G状态与初始值完全相同，梯度均None。语义／Mask／范围观察均无requires_grad，不把冻结路径“零梯度”解释成故障。内存BytesIO严格恢复十项delta、AdamW三组（10／0／0）、moments／steps，零磁盘权重。

预检allocator分配峰值%s字节、保留峰值%s字节，范围为原生工厂之前到恢复结束；序列化%s字节，runner wall %.2f秒。仅单一真实batch测量，不代表所有场景峰值，不据此翻倍batch。没有正式9508指标；预检PASS不等于精度涨点，原G仍为指标最佳5615／4495。

取得实际闭环receipt后，于%s启动原生有限head-only local_range／whole_range串行对照，controller%s。每组从原G和新零头重启、freshAdamW、seed2027、有效batch8、LR1e-5、WD5e-4、clip0.1，29778fit各一次／3723更新（尾batch2），6887初始／终态和9508开发验证分别记录；whole需逐批核对local数据行次序。全冻结eval也改变共同适配、Dropout与梯度预算，故结果将解释为这一完整冻结学习策略，而非单一算子的独立因果作用。该运行已验证启动，未假定训练计数或终态精度。

正式只保存小的十项head delta和optimizer，不再为每个终点重复保存全部原G状态；原G和官方PV是严格重建所需保护父权重。保留原生bbs Acc@0.5指标最佳和一份活动恢复，原G以下的自有终点在完整恢复评估、逐行／CPU阈值重算后自动退役，不创建失败本地权重归档。额外存储预算根据实际预检序列化大小检查，不能将理论文件大小写成已保存实测大小。

唯一预检观察器实际按首查360秒／后续240秒闭环，不额外开GPU任务；正式总时长暂估4—7小时，冻结fit吞吐未测，后续根据实际记录修正并接近阶段结束时观察。源码、配置、真实预检、请求／响应、启动与逐行证据保留。原G不因新协议启动而晋级；ScanRefer5615／4754及独立Nr3D／Sr3D目标仍未完成。

后续判断：若稳定G上已有头仍学不出有效边界，则推进方向保留的六面支撑解码与直接边界监督；若有净增，再单独检验有限联合适配。最终几何质量回写和V99训练期支撑教师仍为后续独立变量，不恢复部署双源，不按Query编号硬对齐，不在开发验证上蒸馏。
''' % (stamp,json.loads((local/'preflight_launch.json').read_bytes())['time_cst'],
    engineering['peak_allocated_bytes'],engineering['peak_reserved_bytes'],engineering['serialization_bytes'],
    engineering['runner_wall_seconds'],launch['time_cst'],launch['process'].split()[0])
new=old+section.encode('utf-8')
prefix='refine-logs/pvground_range_head_only_20261004/'
payloads={}
for path in sorted(local.iterdir()):
    if path.is_file() and path.suffix in ('.py','.md','.json'):
        payloads[prefix+path.name]=path.read_bytes()
payloads[prefix+'.gitattributes']=b'* -text\n'
for path in sorted((local/'complete_preflight').rglob('*')):
    if path.is_file():payloads[prefix+path.relative_to(local).as_posix()]=path.read_bytes()
for path in sorted((local/'.aris/traces/experiment-bridge/2026-10-04_head_only_run01').iterdir()):
    if path.is_file():payloads[prefix+'source_review_trace/'+path.name]=path.read_bytes()
assert all(not (repo/relative).exists() for repo in repos[:2] for relative in payloads)
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp();remote='/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(remote+'/'+doc,'rb') as stream:assert stream.read()==old
directories=sorted({str(Path(relative).parent).replace('\\','/') for relative in payloads},key=lambda x:(x.count('/'),x))
created={'refine-logs'}
for directory in directories:
    parts=directory.split('/')
    for count in range(1,len(parts)+1):
        current='/'.join(parts[:count])
        if current not in created:sftp.mkdir(remote+'/'+current);created.add(current)
for relative,raw in payloads.items():
    for repo in repos[:2]:
        target=repo/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    with sftp.open(remote+'/'+relative,'wx') as stream:stream.write(raw)
    with sftp.open(remote+'/'+relative,'rb') as stream:assert stream.read()==raw
for path in copies:path.write_bytes(new)
with sftp.open(remote+'/'+doc,'wb') as stream:stream.write(new)
assert all(path.read_bytes()==new for path in copies)
with sftp.open(remote+'/'+doc,'rb') as stream:assert stream.read()==new
sftp.close();client.close()
heads=[]
for index,repo in enumerate(repos):
    if index<2:
        with (repo/'MANIFEST.md').open('a',encoding='utf-8') as stream:
            stream.write('\n- '+stamp+' stable originalG/10-head-tensor real two-step PASS; bounded same-budget local/whole launched, no new accuracy result.\n')
    stage=[doc]+(['MANIFEST.md',*payloads] if index<2 else [])
    changed=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git','-C',str(repo),'add','-f','--',*stage])
    subprocess.check_call(['git','-C',str(repo),'-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check','--',*stage])
    for relative,raw in (payloads.items() if index<2 else []):
        assert subprocess.check_output(['git','-C',str(repo),'show',':'+relative])==raw
    subprocess.check_call(['git','-C',str(repo),'commit','-m','Run stable G range head learning control after real two step sanity'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0]==heads[0]
for relative in (doc,*payloads):
    assert subprocess.check_output(['git','-C',str(repos[0]),'rev-parse','HEAD:'+relative])==subprocess.check_output(['git','-C',str(repos[1]),'rev-parse','HEAD:'+relative])
digest=hashlib.sha256(new).hexdigest();guard.write_bytes(guard_raw.replace(previous['handoff_sha256'].encode(),digest.encode()))
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.40',heads=heads,github_main=heads[0],
    handoff_sha256=digest,handoff_bytes=len(new),four_local_and_remote_equal=True,exact_committed_payloads=True,payload_count=len(payloads),
    actual_preflight='PASS',active_controller=launch['process'],formal_result_available=False,retained_metric_best='original_g_5615_4495',
    weights_downloaded=0,goal_achieved=False)
(local/'launch_publication.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
with (workspace/'memory/2026-10-04.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+record['time_cst']+' §20.376.40 published '+heads[0]+' fourlocal+remote SHA'+digest+'; actualfrozenG/only10head tensors2step PASS/0diskweights, originalG unchanged. SameB8/29778once/3723update localwhole pair controller'+launch['process'].split()[0]+' actuallylaunched/'+launch['time_cst']+'; no9508result, G4495best/GoalACTIVE_UNMET. No unchangedsanity/publisher rerun.\n')
print(json.dumps(record),flush=True)
