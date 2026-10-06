"""Reuse the reviewed append-only publication lifecycle for actual new M0 launch."""
import ast
from pathlib import Path

root=Path(__file__).resolve().parent
source=(root.parent/'pvground_mask_extent_diagnostic_20261006/publish_formal_launch_v3.py').read_text(encoding='utf-8')
source=source.replace("formal_launch_publication.json","preflight_launch_publication.json")
source=source.replace('"""Publish reviewed sources, accepted M0 and actual read-only formal launch."""',
    '"""Publish reviewed reference sources and actual two-update sanity launch."""')
source=source.replace('No training/new weights, protected5616/4511 unchanged.',
    'No full fit/new saved weights, protected5616/4511 unchanged.')
source=source.replace("runtime=json.loads((local/'spec.json').read_bytes())['runtime']",
    "runtime=json.loads((local/'native_reference_spec.json').read_bytes())['runtime']")
start=source.index("source=json.loads((local/'SOURCE_REVIEW.json').read_bytes())")
end=source.index("state_path=local.parent",start)
source=source[:start]+"""source=json.loads((local/'SOURCE_REVIEW.json').read_bytes())
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
"""+source[end:]
source=source.replace("=='20.376.79'","=='20.376.81'")
start=source.index("prefix='refine-logs/")
end=source.index("assert chr(65533) not in section",start)
source=source[:start]+"""prefix='refine-logs/pvground_mask_reference_20261006/'
section=f'''

## 20.376.82 全融合Mask空间参考进入六面训练：共同输出初始化的native/fused对照已启动真实两步预检（{stamp}）

承接§81。实际只读9508证明当前同Query完整Mask范围有严格信息，但713修复/376破坏不支持无条件硬换框；训练模型最好仍5616/4511。新实验只研究空间参考，不加attention/质量排名/教师/多seed，不恢复V99双源或七版本选择。

两组完整重建官方PV＋原G＋4511几何delta，保留geometry hidden8，共同将output.weight/output.bias2置零。native_reference以原回归c/s为参考，fused_mask_reference以原生融合logit sigmoid>0.5所支持的真实超点成员极值为参考，覆盖全部256候选。原粗框始终保留6维priorfeatures；新参考控制局部7×16采样和33节点六面分布坐标/DFL目标。109维全实例统计仍保留，1302输入、456102参数/10状态两组相同。范围无法形成时保留原粗框参考：基于§81实测39空支撑，已有有效体积规则；不读取GT或IoU作部署门控、不按当前分数删除候选、不产生第二套答案。

原PV/G/Mask/语言/语义/零输出R冻结，只更新已有geometryhead。native＋G＋匹配DFL及自身Query/fused训练GT确认的未匹配、Box仍不足候选额外几何loss保持，native真实GT主目标、原Hungarian算法和其他已匹配实例排除规则不变。语义头每次forward只执行一次，原生last/bbs、同QueryBox/Mask。参考改变其定位输出会改变匹配结果/额外资格，这是被比较的空间干预，不宣称匹配索引恒定。

seed2027、物理/有效batch8、accum1、LR1e−5、WD5e−4、clip0.1；每组29778输入各一次、3723更新、最后batch2。freshAdam，不承接预检状态。各组初始/终点6887是预训练见过场景的模块留出；另记录新架构零更新initial_formal9508及终点formal9508。声明过的初始/终点与4511比较，若零更新即改善必须标成空间参考初始化效果，而非训练学习增量。保留hidden8累计11169＋3723，重置output2只计本轮3723；官方/G历史另外记录。initial.pth的step0和空Adam须单独实际strict重建，不能用terminal的3723恢复见证代替。新权重按实际原生Acc@0.50保留最佳，闭合非best不归档；官方PV/G、V99必需链和活动恢复文件保护。

fresh-context SOURCE_ONLY初审PASS后，真实readonly前置probe发现旧controller的字段为completed/exit_code、脚本误读status。失败发生在mkdir/upload/GPU之前、0优化步；保留真实错误和初审版本，最小修正两字段并同上下文补审PASS、0未解决阻断。模型/runner/规格没有改动；归属same-family/provisional，backend未attest。sourcePASS不是GPU或精度证据。

实际M0于{launch['time_cst']}提交，已pgrep核对controller703128；两组各2真实更新，检查中性分布等于所选参考、实际raw输入全部256范围、真实39空支撑保留先验、第二步内部梯度、父/R/评分/Mask冻结、CPU模型/Adam严格恢复。现有warm环境与算子缓存复用，不重装。sole observer44835计划11:04:28首查、后续每240s；估时900s。此发布不声称已完成预检、已正式训练或获得准确率。M1未开始；预计完整pair约22000s，首次观察21600s约结束前7min，再240s；依据实际吞吐校正，不重复controller/observer。代码/规格/失败/实际launch/review见{prefix}，fixture二进制和权重不上传Git。目标ACTIVE_UNMET，当前best4511距4754仍243条。
'''
"""+source[end:]
source=source.replace("new.count(b'## 20.376.80 ')==1","new.count(b'## 20.376.82 ')==1")
start=source.index("names=[path.name")
end=source.index("assert len(names)==len(set(names))",start)
source=source[:start]+"""names=[path.name for path in sorted(local.glob('*.py'))]
names+=['native_reference_spec.json','fused_mask_reference_spec.json','EXPERIMENT_PLAN.md',
    'research_contract.md','EXPERIMENT_TRACKER.md','ACTUAL_EXECUTION_TRACKER.md','MANIFEST.md',
    'invalid_reference_fixture.json','DEPLOY_PROBE_FAILURE.json','resource_check.json',
    'preflight_launch.json','PREFLIGHT_EXECUTION_SESSIONS.json','SOURCE_REVIEW_CALL.json',
    'SOURCE_REVIEW_FOLLOWUP_CALL.json','SOURCE_REVIEW.json','SOURCE_REVIEW.md',
    'PUBLISH_PREFLIGHT_REVIEW.json','PUBLISH_PREFLIGHT_REVIEW.md']
names += [path.name for path in sorted(local.glob('SOURCE_REVIEW_2026*'))]
"""+source[end:]
source=source.replace("'/root/autodl-tmp/mcln_published_evidence_20261005/pvground_mask_extent_diagnostic_20261006'",
    "'/root/autodl-tmp/mcln_published_evidence_20261005/pvground_mask_reference_20261006'")
source=source.replace(' Protected4511 completeMask extent read-only diagnostic: actual raw8 M0 accepted, formal9508 running; no new accuracy or weights.',
    ' Protected4511 native/fused Mask spatial-reference pair: SOURCE reviewed,actual2-step sanity launched; no accuracyyet.')
source=source.replace('Record accepted raw Mask extent preflight and actual formal launch',
    'Record reviewed Mask spatial-reference source and actual two-step sanity launch')
start=source.index("record=dict(time_cst=")
end=source.index("(local/'preflight_launch_publication.json')",start)
source=source[:start]+"""record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.82',heads=heads,
    handoff_sha256=digest,four_local_and_remote_equal=True,github_main=heads[0],
    new_accuracy_result=False,preflight_controller=launch['process'],observer_native_session_id=44835,
    source_verdict=source['verdict'],payload_count=len(payloads),raw_npz_published=False,
    protected_trained_model_hits=[5616,4511])
"""+source[end:]
start=source.index("state.update(time_cst=")
end=source.index('state_path.write_text',start)
source=source[:start]+"""state.update(time_cst=record['time_cst'],latest_publication=str(local/'preflight_launch_publication.json'),
    status='MASK_REFERENCE_PREFLIGHT_ACTUAL_LAUNCH_PUBLISHED',owned_gpu_job_active=True,active_reviewer=None,
    reference_preflight_launch=str(local/'preflight_launch.json'),reference_observer_native_session_id=44835,
    handoff_section=record['section'],handoff_sha256=digest,published_heads=heads,
    next_action='Wait sole44835; collect actual M0 closure before fresh full fit. No new accuracy or best yet.')
"""+source[end:]
source=source.replace('Doc80 actual8row rawCPU M0 accepted and sameQuery9508 read-only diagnostic launched; one formal observer near estimated finish.',
    'Doc82 actualMask-reference M0 launched;controller703128/soleobserver44835;fullfitnotstarted.')
ast.parse(source)
(root/'publish_preflight_launch.py').write_text(source,encoding='utf-8')
print('PREFLIGHT_PUBLICATION_PREPARED_NOT_EXECUTED')
