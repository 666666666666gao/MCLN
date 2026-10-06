"""Publish reviewed sources, accepted M0 and actual read-only formal launch."""
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
assert not (local/'formal_launch_publication.json').exists()
source=json.loads((local/'SOURCE_REVIEW.json').read_bytes())
review=json.loads((local/'FORMAL_LAUNCH_REVIEW.json').read_bytes())
publication_review=json.loads((local/'PUBLISH_LAUNCH_V2_REVIEW.json').read_bytes())
wait=json.loads((local/'preflight_wait.json').read_bytes())
live=json.loads((local/'formal_observer_live.json').read_bytes())
launch=json.loads((local/'formal_launch.json').read_bytes())
receipt=json.loads((local/'complete/preflight/receipt.json').read_bytes())
assert source['verdict'] in ('PASS','WARN') and not source['blocking_findings']
assert review['verdict'] in ('PASS','WARN') and not review['blocking_findings']
assert publication_review['execution_scope']=='SOURCE_ONLY' and publication_review['verdict'] in ('PASS','WARN') and not publication_review['blocking_findings']
for item in publication_review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
assert wait['observer_closed'] and wait['controller']['exit_code']==0
assert live['status']=='ACTUAL_FORMAL_CONTROLLER_AND_CHILD_LIVE' and not live['observer_closed']
assert receipt['model_states_unchanged'] and receipt['optimizer_updates']==0
state_path=local.parent/'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state=json.loads(state_path.read_bytes())
previous=json.loads(Path(state['latest_publication']).read_bytes())
assert previous['section']=='20.376.79'
repos=[workspace/'.codex_mcln_g0_20260905',workspace/'.codex_pvground_cs_20261002',workspace/'.codex_mcln_v99_internal_20260928']
doc='docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies=[repo/doc for repo in repos]+[workspace/'Desktop/document'/Path(doc).name]
old=copies[0].read_bytes()
assert hashlib.sha256(old).hexdigest()==previous['handoff_sha256'] and all(path.read_bytes()==old for path in copies)
for repo,head in zip(repos,previous['heads']):
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==head
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
stamp=datetime.datetime.now().astimezone().isoformat()
prefix='refine-logs/pvground_mask_extent_diagnostic_20261006/'
section=f'''

## 20.376.80 当前最好4511的完整Mask范围只读诊断：真实首8条通过，9508条已启动（{stamp}）

承接§79，固定PV-Ground主线，当前保留5616/4511，ScanRefer原生Acc@0.50距离4754仍差243条。上一轮支撑参考实验已经闭合，control4505、support4495未超过4511；两份闭合非best权重已删除，释放11210680字节。此次不重新训练，不加入注意力、质量排序或教师，不恢复V99双源及几何版本选择。

唯一问题：当前同一前向、同一原生bbs选中Query中，完整预测融合Mask的范围是否提供学习框尚未利用的严格定位信息。固定比较原学习框、fused sigmoid>0.5真实输入点成员的精确极值框、同一前景点每轴0.5%/99.5%线性分位框。后两者仅作离线诊断，不写回模型，不改变选择、Mask、评分或部署输出；空支撑或非正范围按无效IoU0，不回填原框。GT仅在forward和选择之后用于IoU、Mask资格和分组，不用于范围构造或筛选候选。

冻结重建同一官方PV、原G与受保护4511几何delta，eval/no_grad，无优化器，无更新或权重保存。seed2027、B8、workers2、val9508、augment/augment_det均关闭，原生两阶段预测对象输入、全256候选、同Query Box/Mask保留；最终原生语义头每次只执行一次。spec继承的fit/update计数属于历史配置，不能当成本次实际训练。

SOURCE_ONLY新鲜审查{source['verdict']}，0阻断；正式launcher/observer同上下文后续审查{review['verdict']}，0阻断。原SOURCE审查和followup分开保留；same-family/provisional，模型后端身份未attest。真实M0于{wait['controller']['finished_cst']}闭合exit0，controller{wait['controller']['controller_pid']}、child{wait['controller']['child_pid']}；sole observer已闭合。实际8条50000点XYZRGB、超点映射和GT点Mask独立CPU复算，成员极值、np.quantile、Mask交集与远端摘要一致，阈值翻转0。模型状态不变，0优化器/更新/权重。8条原生严格4，exact2、q0052仅用于工程预检，不能称正式收益或预测全9508结果。allocator峰值{receipt['cuda_peak_allocated_bytes']}字节、reserved{receipt['cuda_peak_reserved_bytes']}字节，属于只读预检。

正式只读命令于{launch['time_cst']}提交，已核对实际controller{live['controller']['controller_pid']}、child{live['controller']['child_pid']}和/proc命令。实际开始{live['controller']['started_cst']}，预估1800秒，唯一observer首次于{live['first_check_cst']}检查，之后每240秒；不会重复启动controller/observer。此发布时尚无完整9508终态或新精度。启动时实际数据盘空闲{launch['resources']['data_free_bytes']}字节；首批压缩体积推算加128MiB的保留门槛{launch['forecast_bytes_from_first_batch']}字节仅为样本预测，不是全场景上界。

正式每批保留实际超点成员数、XYZ极值、GT成员数、selected Query/Text/fused logits及分位排序相邻节点和rank。CPU可以复核exact、Mask交集和分位插值解码；50000原始点独立排序回放仅M0首8条，不能声称正式全9508原始点重放。NPZ只保留一份本地证据，不上传GitHub；公开来源、SHA和小型状态记录见{prefix}。诊断结束后报告修复/破坏、无效范围及目标尺寸分组，再决定是否值得改变空间参考。V99历史条件收益不能代填本次结果；无Nr3D/Sr3D新成绩。目标ACTIVE_UNMET。
'''
assert chr(65533) not in section
new=old+section.encode('utf-8')
assert new.startswith(old) and new.count(b'## 20.376.80 ')==1
names=[path.name for path in sorted(local.glob('*.py')) if path.name!='record_source_completion.py']
names += ['spec.json','EXPERIMENT_PLAN.md','EXPERIMENT_TRACKER.md','ACTUAL_EXECUTION_TRACKER.md','research_contract.md','MANIFEST.md',
          'SOURCE_REVIEW.json','SOURCE_REVIEW.md','SOURCE_REVIEW_CALL.json',
          'FORMAL_LAUNCH_REVIEW.json','FORMAL_LAUNCH_REVIEW.md','FORMAL_LAUNCH_REVIEW_CALL.json',
          'PUBLISH_LAUNCH_REVIEW.json','PUBLISH_LAUNCH_REVIEW.md',
          'PUBLISH_LAUNCH_V2_REVIEW.json','PUBLISH_LAUNCH_V2_REVIEW.md','PUBLICATION_FAILURE_INSPECTION.json',
          'preflight_launch.json','preflight_wait.json','formal_launch.json','formal_observer_live.json','FORMAL_LIVE_SESSIONS.json']
names += [str(path.relative_to(local)).replace('\\','/') for path in sorted((local/'complete').rglob('*'))
          if path.is_file() and path.suffix in ('.json','.jsonl','.log','.exit') and 'formal' not in path.relative_to(local).parts]
names += ['local_cpu_m0/CPU_SUMMARY.json']
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
assert not (project/bundle['prefix']).exists()
for name,encoded in bundle['files'].items():
    assert name.startswith(bundle['prefix']) and '.aris' not in name and not name.endswith(('.pt','.pth','.npz'))
    path=project/name;assert project.resolve() in path.resolve().parents
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
runtime=json.loads((local/'spec.json').read_bytes())['runtime']
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
            stream.write('\n- '+stamp+' Protected4511 completeMask extent read-only diagnostic: actual raw8 M0 accepted, formal9508 running; no new accuracy or weights.\n')
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
    subprocess.check_call(['git','-C',str(repo),'commit','--quiet','-m','Record accepted raw Mask extent preflight and actual formal launch'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'-c','http.version=HTTP/1.1','push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0]==heads[0]
digest=hashlib.sha256(new).hexdigest();assert digest==remote['handoff_sha256']
guard=workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py';raw=guard.read_bytes()
assert raw.count(previous['handoff_sha256'].encode())==1
guard.write_bytes(raw.replace(previous['handoff_sha256'].encode(),digest.encode()))
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.80',heads=heads,
            handoff_sha256=digest,four_local_and_remote_equal=True,github_main=heads[0],
            new_accuracy_result=False,formal_controller=live['controller'],source_verdict=source['verdict'],
            formal_wrapper_verdict=review['verdict'],payload_count=len(payloads),raw_npz_published=False)
(local/'formal_launch_publication.json').write_text(json.dumps(record,indent=2)+'\n')
state.update(time_cst=record['time_cst'],latest_publication=str(local/'formal_launch_publication.json'),
             status='MASK_EXTENT_FORMAL_ACTIVE_PUBLISHED',owned_gpu_job_active=True,active_reviewer=None,
             mask_extent_formal_launched=True,mask_extent_formal_live=str(local/'formal_observer_live.json'),
             handoff_section=record['section'],handoff_sha256=digest,published_heads=heads,
             next_action='Keep single formal observer; collect actual9508 closure, fresh terminal audit; no new training until results.')
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (workspace/'memory/2026-10-06.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+record['time_cst']+': Doc80 actual8row rawCPU M0 accepted and sameQuery9508 read-only diagnostic launched; one formal observer near estimated finish. Fourlocal/remote exact, main '+heads[0]+'. No training/new weights, protected5616/4511 unchanged.\n')
print(json.dumps(record),flush=True)
