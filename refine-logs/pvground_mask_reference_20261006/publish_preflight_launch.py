"""Publish reviewed reference sources and actual two-update sanity launch."""
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
assert not (local/'preflight_launch_publication.json').exists()
source=json.loads((local/'SOURCE_REVIEW.json').read_bytes())
publication_review=json.loads((local/'PUBLISH_PREFLIGHT_REVIEW.json').read_bytes())
launch=json.loads((local/'preflight_launch.json').read_bytes())
live=json.loads((local/'PREFLIGHT_EXECUTION_SESSIONS.json').read_bytes())
assert source['execution_scope']=='SOURCE_ONLY' and source['verdict'] in ('PASS','WARN') and not source['blocking_findings']
for item in source['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
assert publication_review['execution_scope']=='SOURCE_ONLY' and publication_review['verdict'] in ('PASS','WARN') and not publication_review['blocking_findings']
for item in publication_review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
assert launch['status']=='PREFLIGHT_LAUNCHED_NOT_COMPLETED' and launch['process'].startswith('703128 ')
assert live['controller_pid']==703128 and live['sole_preflight_observer_native_session_id']==44835
assert live['deployment_session_closed'] and live['deployment_exit_code']==0 and not live['observer_closed']
state_path=local.parent/'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state=json.loads(state_path.read_bytes())
previous=json.loads(Path(state['latest_publication']).read_bytes())
assert previous['section']=='20.376.81'
repos=[workspace/'.codex_mcln_g0_20260905',workspace/'.codex_pvground_cs_20261002',workspace/'.codex_mcln_v99_internal_20260928']
doc='docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies=[repo/doc for repo in repos]+[workspace/'Desktop/document'/Path(doc).name]
old=copies[0].read_bytes()
assert hashlib.sha256(old).hexdigest()==previous['handoff_sha256'] and all(path.read_bytes()==old for path in copies)
for repo,head in zip(repos,previous['heads']):
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==head
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
stamp=datetime.datetime.now().astimezone().isoformat()
prefix='refine-logs/pvground_mask_reference_20261006/'
section=f'''

## 20.376.82 全融合Mask空间参考进入六面训练：共同输出初始化的native/fused对照已启动真实两步预检（{stamp}）

承接§81。实际只读9508证明当前同Query完整Mask范围有严格信息，但713修复/376破坏不支持无条件硬换框；训练模型最好仍5616/4511。新实验只研究空间参考，不加attention/质量排名/教师/多seed，不恢复V99双源或七版本选择。

两组完整重建官方PV＋原G＋4511几何delta，保留geometry hidden8，共同将output.weight/output.bias2置零。native_reference以原回归c/s为参考，fused_mask_reference以原生融合logit sigmoid>0.5所支持的真实超点成员极值为参考，覆盖全部256候选。原粗框始终保留6维priorfeatures；新参考控制局部7×16采样和33节点六面分布坐标/DFL目标。109维全实例统计仍保留，1302输入、456102参数/10状态两组相同。范围无法形成时保留原粗框参考：基于§81实测39空支撑，已有有效体积规则；不读取GT或IoU作部署门控、不按当前分数删除候选、不产生第二套答案。

原PV/G/Mask/语言/语义/零输出R冻结，只更新已有geometryhead。native＋G＋匹配DFL及自身Query/fused训练GT确认的未匹配、Box仍不足候选额外几何loss保持，native真实GT主目标、原Hungarian算法和其他已匹配实例排除规则不变。语义头每次forward只执行一次，原生last/bbs、同QueryBox/Mask。参考改变其定位输出会改变匹配结果/额外资格，这是被比较的空间干预，不宣称匹配索引恒定。

seed2027、物理/有效batch8、accum1、LR1e−5、WD5e−4、clip0.1；每组29778输入各一次、3723更新、最后batch2。freshAdam，不承接预检状态。各组初始/终点6887是预训练见过场景的模块留出；另记录新架构零更新initial_formal9508及终点formal9508。声明过的初始/终点与4511比较，若零更新即改善必须标成空间参考初始化效果，而非训练学习增量。保留hidden8累计11169＋3723，重置output2只计本轮3723；官方/G历史另外记录。initial.pth的step0和空Adam须单独实际strict重建，不能用terminal的3723恢复见证代替。新权重按实际原生Acc@0.50保留最佳，闭合非best不归档；官方PV/G、V99必需链和活动恢复文件保护。

fresh-context SOURCE_ONLY初审PASS后，真实readonly前置probe发现旧controller的字段为completed/exit_code、脚本误读status。失败发生在mkdir/upload/GPU之前、0优化步；保留真实错误和初审版本，最小修正两字段并同上下文补审PASS、0未解决阻断。模型/runner/规格没有改动；归属same-family/provisional，backend未attest。sourcePASS不是GPU或精度证据。

实际M0于{launch['time_cst']}提交，已pgrep核对controller703128；两组各2真实更新，检查中性分布等于所选参考、实际raw输入全部256范围、真实39空支撑保留先验、第二步内部梯度、父/R/评分/Mask冻结、CPU模型/Adam严格恢复。现有warm环境与算子缓存复用，不重装。sole observer44835计划11:04:28首查、后续每240s；估时900s。此发布不声称已完成预检、已正式训练或获得准确率。M1未开始；预计完整pair约22000s，首次观察21600s约结束前7min，再240s；依据实际吞吐校正，不重复controller/observer。代码/规格/失败/实际launch/review见{prefix}，fixture二进制和权重不上传Git。目标ACTIVE_UNMET，当前best4511距4754仍243条。
'''
assert chr(65533) not in section
new=old+section.encode('utf-8')
assert new.startswith(old) and new.count(b'## 20.376.82 ')==1
names=[path.name for path in sorted(local.glob('*.py'))]
names+=['native_reference_spec.json','fused_mask_reference_spec.json','EXPERIMENT_PLAN.md',
    'research_contract.md','EXPERIMENT_TRACKER.md','ACTUAL_EXECUTION_TRACKER.md','MANIFEST.md',
    'invalid_reference_fixture.json','DEPLOY_PROBE_FAILURE.json','resource_check.json',
    'preflight_launch.json','PREFLIGHT_EXECUTION_SESSIONS.json','SOURCE_REVIEW_CALL.json',
    'SOURCE_REVIEW_FOLLOWUP_CALL.json','SOURCE_REVIEW.json','SOURCE_REVIEW.md',
    'PUBLISH_PREFLIGHT_REVIEW.json','PUBLISH_PREFLIGHT_REVIEW.md']
names += [path.name for path in sorted(local.glob('SOURCE_REVIEW_2026*'))]
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
assert str(evidence)=='/root/autodl-tmp/mcln_published_evidence_20261005/pvground_mask_reference_20261006'
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
            stream.write('\n- '+stamp+' Protected4511 native/fused Mask spatial-reference pair: SOURCE reviewed,actual2-step sanity launched; no accuracyyet.\n')
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
    subprocess.check_call(['git','-C',str(repo),'commit','--quiet','-m','Record reviewed Mask spatial-reference source and actual two-step sanity launch'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'-c','http.version=HTTP/1.1','push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0]==heads[0]
digest=hashlib.sha256(new).hexdigest();assert digest==remote['handoff_sha256']
guard=workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py';raw=guard.read_bytes()
assert raw.count(previous['handoff_sha256'].encode())==1
guard.write_bytes(raw.replace(previous['handoff_sha256'].encode(),digest.encode()))
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.82',heads=heads,
    handoff_sha256=digest,four_local_and_remote_equal=True,github_main=heads[0],
    new_accuracy_result=False,preflight_controller=launch['process'],observer_native_session_id=44835,
    source_verdict=source['verdict'],payload_count=len(payloads),raw_npz_published=False,
    protected_trained_model_hits=[5616,4511])
(local/'preflight_launch_publication.json').write_text(json.dumps(record,indent=2)+'\n')
state.update(time_cst=record['time_cst'],latest_publication=str(local/'preflight_launch_publication.json'),
    status='MASK_REFERENCE_PREFLIGHT_ACTUAL_LAUNCH_PUBLISHED',owned_gpu_job_active=True,active_reviewer=None,
    reference_preflight_launch=str(local/'preflight_launch.json'),reference_observer_native_session_id=44835,
    handoff_section=record['section'],handoff_sha256=digest,published_heads=heads,
    next_action='Wait sole44835; collect actual M0 closure before fresh full fit. No new accuracy or best yet.')
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (workspace/'memory/2026-10-06.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+record['time_cst']+': Doc82 actualMask-reference M0 launched;controller703128/soleobserver44835;fullfitnotstarted. Fourlocal/remote exact, main '+heads[0]+'. No full fit/new saved weights, protected5616/4511 unchanged.\n')
print(json.dumps(record),flush=True)
