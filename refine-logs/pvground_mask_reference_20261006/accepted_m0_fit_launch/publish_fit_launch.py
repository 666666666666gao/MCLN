"""Publish accepted actual two-update runtime and submitted full-budget pair."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import base64
import paramiko

local=Path(__file__).resolve().parent
workspace=Path('C:/Users/gb')
assert not (local/'fit_launch_publication.json').exists()
source=json.loads((local/'SOURCE_REVIEW.json').read_bytes())
publication_review=json.loads((local/'PUBLISH_FIT_REVIEW.json').read_bytes())
launch=json.loads((local/'fit_launch.json').read_bytes())
live=json.loads((local/'M0_ACCEPTANCE_AND_FIT_LAUNCH.json').read_bytes())
assert source['execution_scope']=='SOURCE_ONLY' and source['verdict'] in ('PASS','WARN') and not source['blocking_findings']
for item in source['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
assert publication_review['execution_scope']=='SOURCE_ONLY' and publication_review['verdict'] in ('PASS','WARN') and not publication_review['blocking_findings']
for item in publication_review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
assert launch['status']=='TWO_ARM_FIT_LAUNCHED_NOT_COMPLETED' and launch['process'].startswith('711378 ')
assert live['fit_controller_pid']==711378 and live['sole_fit_observer_native_session_id']==27853
assert live['preflight_controller_closed'] and live['preflight_observer_closed'] and not live['observer_closed']
state_path=local.parent/'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state=json.loads(state_path.read_bytes())
previous=json.loads(Path(state['latest_publication']).read_bytes())
assert previous['section']=='20.376.82'
repos=[workspace/'.codex_mcln_g0_20260905',workspace/'.codex_pvground_cs_20261002',workspace/'.codex_mcln_v99_internal_20260928']
doc='docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies=[repo/doc for repo in repos]+[workspace/'Desktop/document'/Path(doc).name]
old=copies[0].read_bytes()
assert hashlib.sha256(old).hexdigest()==previous['handoff_sha256'] and all(path.read_bytes()==old for path in copies)
for repo,head in zip(repos,previous['heads']):
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==head
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
stamp=datetime.datetime.now().astimezone().isoformat()
prefix='refine-logs/pvground_mask_reference_20261006/accepted_m0_fit_launch/'
section=f'''

## 20.376.83 Mask空间参考两组真实预检闭合，固定预算完整对照已提交（{stamp}）

承接§82的启动快照。M0实际于2026-10-06 11:08:20.348532闭合，controller703128、最后child707951、exit0；唯一observer44835于11:08:29第二次240秒检查记录闭合，native会话也已exit0，不重启。每组同8条真实fit输入重复2次更新，父模型/R全部状态不变、每组456102参数/10状态，原语义头每普通forward一次；更新后缓存上游评分/Mask重放精确。此不宣称两次fresh forward或跨进程输出逐位一致。

中性33节点分布初始解码精确等于其所选参考中心及既有SIZE_FLOOR尺寸；实际raw点独立核验全部256参考、误差0；39条既存空支撑在CUDA执行确认保留提供的原粗框先验、无GT字段。第二步所有10参数梯度非零、冻结参数无梯度；CPU保存/严格重载模型delta与Adam组/步数/所有moment精确。两组序列化5481547B，0权重文件、0精度结果，全部24实际预检文件154131B已收集/校验。native/fused各461.21/470.05秒，合计controller946.20秒；各峰值allocated3911791616B、reserved5672796160B，仅这次冻结骨干预检，不据此机械扩batch。

小批次也显示范围限制未消失：第一步native匹配DFL目标48面中outside0，Mask参考outside2；额外native143候选/858面中outside120，Mask参考128候选/768面中outside196。第二步相应126/200面。参考不可形成的候选计数1033→1045（每步全部8×256），它们保留原粗参考而没有删除；不把这些小批次计数外推为全验证发生率或失败唯一原因。不修改本轮节点/损失/采样规则。

M1实际于{launch['time_cst']}提交、pgrep核对controller711378，screen pvg_mask_reference_fit_20261006，唯一本地observer27853已启动。各组3723更新、29778输入各一次、B8/effective8/accum1、LR1e−5/WD5e−4/clip0.1；相同官方PV/G/4511、共同重置output2/保留hidden8，从全新Adam开始、不承接M0两步状态。每组初始9508、新fit的6887初始与终点、终点9508；初始架构收益与训练收益分开。只训练已有几何头，原生last/bbs、同QueryBox/Mask、全256保留，不加入新质量排名、teacher或V99双源。

实际启动前GPU空闲，数据盘可用1206886400B、所需reserve960163041B，系统盘463011840B，warm环境/算子缓存复用、无安装。估计总22000秒约17:20:54；sole observer首17:14:14、之后240秒，到闭合后流式收集实际证据、排除权重，不在远端另建大tar。此发布尚未取得M1闭合或新准确率；SourceOnly审查及M0通过不算涨点。结束后fresh terminal审计、所选初始/终点实际strict恢复，并仅保留真实指标最佳已闭合权重，原PV/G与V99必需依赖保护。当前trainedbest仍5616/4511、47.4443%、距4754差243；offline4848仍是§81诊断，目标ACTIVE_UNMET。

已审主源保持原版本；acceptedM0、actualfitlaunch/observer、源码与收集清单见{prefix}。原始point/NPZ与权重不上传Git；三仓、桌面与远端文档同步仍为append-only原字节前缀。旧§82是当时启动快照，当前状态以本节和M0_ACCEPTANCE_AND_FIT_LAUNCH为准。
'''
assert chr(65533) not in section
new=old+section.encode('utf-8')
assert new.startswith(old) and new.count(b'## 20.376.83 ')==1
names=['prepare_fit_publication.py','publish_fit_launch.py','record_m0_and_fit.py',
    'M0_ACCEPTANCE_AND_FIT_LAUNCH.json','preflight_wait.json','fit_launch.json','fit_resource_check.json',
    'fit_observer_started.json','PUBLISH_FIT_REVIEW.json','PUBLISH_FIT_REVIEW.md']
names += [str(path.relative_to(local)).replace('\\','/') for path in sorted((local/'preflight_complete').rglob('*')) if path.is_file()]
assert len(names)==len(set(names))
assert all('.aris' not in name and not name.endswith(('.pth','.pt','.npz')) for name in names)
payloads={prefix+name:(local/name).read_bytes() for name in names}
payloads[prefix+'.gitattributes']=b'** -text\n'
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
project='/home/gb/new butd/butd_detr-main/MCLN-main'
code='''import base64,hashlib,json,sys
from pathlib import Path
project=Path(sys.argv[1]);bundle=json.load(sys.stdin)
doc=project/bundle['doc'];old=doc.read_bytes()
assert hashlib.sha256(old).hexdigest()==bundle['old_sha256']
evidence=(project/bundle['prefix']).resolve()
assert str(evidence)=='/root/autodl-tmp/mcln_published_evidence_20261005/pvground_mask_reference_20261006/accepted_m0_fit_launch'
assert not evidence.exists()
for name,encoded in bundle['files'].items():
    assert name.startswith(bundle['prefix']) and '.aris' not in name and not name.endswith(('.pt','.pth','.npz'))
    path=project/name;assert evidence in path.resolve().parents
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=base64.b64decode(encoded)
    with path.open('xb') as stream:stream.write(raw)
    assert path.read_bytes()==raw
new=base64.b64decode(bundle['new_doc']);assert new.startswith(old)
doc.write_bytes(new);assert doc.read_bytes()==new
print(json.dumps(dict(files=len(bundle['files']),handoff_sha256=hashlib.sha256(new).hexdigest())))
'''
bundle=dict(doc=doc,prefix=prefix,old_sha256=previous['handoff_sha256'],
            files={name:base64.b64encode(raw).decode() for name,raw in payloads.items()},new_doc=base64.b64encode(new).decode())
runtime=json.loads((local/'native_reference_spec.json').read_bytes())['runtime']
stdin,stdout,stderr=client.exec_command(shlex.join([runtime+'/venv/bin/python','-B','-c',code,project]),timeout=180)
stdin.write(json.dumps(bundle).encode());stdin.flush();stdin.channel.shutdown_write()
remote_raw=stdout.read();remote_exit=stdout.channel.recv_exit_status();remote_error=stderr.read().decode()
assert remote_exit==0,remote_error
remote=json.loads(remote_raw)
client.close()
for name,raw in payloads.items():
    for repo in repos[:2]:
        destination=repo/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(raw)
for path in copies:path.write_bytes(new)
heads=[]
for index,repo in enumerate(repos):
    stage=[doc]
    if index<2:
        with (repo/'MANIFEST.md').open('a',encoding='utf-8') as stream:
            stream.write('\n- '+stamp+' Protected4511 Mask spatial-reference pair: actualM0 accepted,full3723fitperarm submitted; no newaccuracyyet.\n')
        stage += ['MANIFEST.md',*payloads]
    changed=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git','-C',str(repo),'add','-f','--',*stage],stderr=subprocess.DEVNULL)
    reports=[name for name in payloads if Path(name).name in ('SOURCE_REVIEW.json','SOURCE_REVIEW.md','FORMAL_LAUNCH_REVIEW.json','FORMAL_LAUNCH_REVIEW.md')] if index<2 else []
    subprocess.check_call(['git','-C',str(repo),'-c','core.whitespace=cr-at-eol','diff','--cached','--check','--',*stage,*[':(exclude)'+name for name in reports]])
    subprocess.check_call(['git','-C',str(repo),'-c','core.whitespace=cr-at-eol,-blank-at-eof','diff','--cached','--check','--',*stage])
    if index<2:
        assert all(subprocess.check_output(['git','-C',str(repo),'show',':'+name])==raw for name,raw in payloads.items())
    assert subprocess.check_output(['git','-C',str(repo),'show',':'+doc]).startswith(subprocess.check_output(['git','-C',str(repo),'show',previous['heads'][index]+':'+doc]))
    subprocess.check_call(['git','-C',str(repo),'commit','--quiet','-m','Record accepted Mask-reference two-step runtime and actual fixed-budget pair launch'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'-c','http.version=HTTP/1.1','push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0]==heads[0]
digest=hashlib.sha256(new).hexdigest();assert digest==remote['handoff_sha256']
guard=workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py';raw=guard.read_bytes()
assert raw.count(previous['handoff_sha256'].encode())==1
guard.write_bytes(raw.replace(previous['handoff_sha256'].encode(),digest.encode()))
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.83',heads=heads,
    handoff_sha256=digest,four_local_and_remote_equal=True,github_main=heads[0],
    new_accuracy_result=False,fit_controller=launch['process'],observer_native_session_id=27853,
    source_verdict=source['verdict'],payload_count=len(payloads),raw_npz_published=False,
    protected_trained_model_hits=[5616,4511])
(local/'fit_launch_publication.json').write_text(json.dumps(record,indent=2)+'\n')
state.update(time_cst=record['time_cst'],latest_publication=str(local/'fit_launch_publication.json'),
    status='MASK_REFERENCE_FULL_PAIR_ACTIVE_PUBLISHED',owned_gpu_job_active=True,active_reviewer=None,
    reference_fit_launch=str(local/'fit_launch.json'),reference_observer_native_session_id=27853,
    handoff_section=record['section'],handoff_sha256=digest,published_heads=heads,
    next_action='Do not restart711378/27853;sole firstcheck17:14:14,240safter;fullclosure auto collects. No newaccuracy or best yet.')
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (workspace/'memory/2026-10-06.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+record['time_cst']+': Doc83 actualMask-reference M0 accepted and fullpair launched;controller711378/soleobserver27853;no newaccuracyyet. Fourlocal/remote exact, main '+heads[0]+'. Full fit launched; no retrieved new accuracy, protected5616/4511 unchanged.\n')
print(json.dumps(record),flush=True)
