"""Prepare an append-only terminal-evidence publication using the closed lifecycle."""
import ast
from pathlib import Path

root=Path(__file__).resolve().parent
source=(root/'publish_formal_launch_v3.py').read_text(encoding='utf-8')
start=source.index("source=json.loads((local/'SOURCE_REVIEW.json').read_bytes())")
end=source.index("state_path=local.parent",start)
source=source[:start]+"""audit=json.loads((local/'analysis/EXPERIMENT_AUDIT.json').read_bytes())
acceptance=json.loads((local/'analysis/TERMINAL_ACCEPTANCE.json').read_bytes())
summary=json.loads((local/'analysis/SUMMARY.json').read_bytes())
publication_review=json.loads((local/'PUBLISH_TERMINAL_REVIEW.json').read_bytes())
assert audit['verdict'] in ('PASS','WARN') and not audit['blocking_findings']
assert audit['execution_scope']=='TERMINAL_SOURCE_AND_ACTUAL_ARTIFACTS'
assert acceptance['offline_not_new_trained_result'] and acceptance['trained_best_hits']==[5616,4511]
assert publication_review['execution_scope']=='SOURCE_ONLY'
assert publication_review['verdict'] in ('PASS','WARN') and not publication_review['blocking_findings']
for item in publication_review['reviewed_files']:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
"""+source[end:]
source=source.replace('formal_launch_publication.json','terminal_publication.json').replace("=='20.376.79'","=='20.376.80'")
start=source.index("prefix='refine-logs/")
end=source.index("assert chr(65533) not in section",start)
source=source[:start]+"""prefix='refine-logs/pvground_mask_extent_diagnostic_20261006/terminal/'
section=f'''

## 20.376.81 当前4511模型完整Mask范围诊断闭合：同Query exact4848，未当作训练成绩（{stamp}）

承接§80。实际9508只读任务于{summary['actual_controller']['finished_cst']}闭合，controller699895、child699896、exit0；sole observer59532和CPU分组34764均已闭合，不重启旧进程。重建同一官方PV＋原G＋受保护4511几何头，eval/no_grad；0优化器、0更新、0新权重。完整256候选保留，原生bbs选中的同一Query，原语义头每批仅调用一次，Box与Mask身份一致。GT只在前向及选择之后用于IoU/Mask诊断。

| 同Query范围 | Acc@0.25命中 | Acc@0.50命中 | 严格修复/破坏/净变化 |
|---|---:|---:|---:|
| 当前学习框 |5616|4511|—|
| 全融合Mask sigmoid>0.5实际成员精确范围 |5598|4848|713/376/+337|
| 同一前景各轴0.5%/99.5%线性分位范围 |5590|4781|663/393/+270|

exact@0.25修复174/破坏192/净−18；q005修复167/破坏193/净−26。两种Mask范围均39条无效，实际均为空支撑，诊断计IoU0，没有用原框补齐。原历史9508逐行身份、点hash、rootGT、选中Query和严格标签一致；505行IoU有微小浮点差异、最大约8.106e−6，不能写成逐位复现。

自身Query与融合Mask都>0.5的5124条中，原严格4169→exact4577，修复644/破坏236/净408；其中原框不合格955条，exact修复644。公共融合好、自身Query不好仅9条，严格5→5；两Mask都不合格4366条，exact严格328→258，净−70。此为当前模型离线GT分组，资格不进入部署；不能把GT分组变成推理门控，也不能把644条换算成新网络预期收益。目标体积四分位下exact严格净变化170/93/10/64，分组阈值/并列处理保留在实际CPU报告。

实际推理1755.08秒、总1882.98秒；峰值allocator4321393664B、reserved5739905024B。1189份NPZ共450284771B，完整收集1197文件462713589B，权重0。独立fresh-context terminal SOURCE_AND_ACTUAL审查{audit['verdict']}、0阻断，全部1197 SHA/bytes、9508成员极值/Mask交并/分位邻节点/IoU/修复破坏及分组均重算，阈值翻转0。完整原始50000点排序重放仅M0的8条；正式9508核验保存的成员证据，并非全原始点复放。审查属于same-family/provisional、backend未attest。单seed与反复开发验证的范围限制保留。数据和NPZ留本地/服务器，不上传Git，仅源与摘要、实际收集清单和审查记录。

SUMMARY、NARRATIVE_REPORT与TERMINAL_EXECUTION_TRACKER保留审查前的pending/in-progress快照以保持被审原字节；最终审查状态以TERMINAL_ACCEPTANCE.json和已返回的EXPERIMENT_AUDIT.json/md为准，不将旧启动快照写成服务器当前状态。

结论：完整预测Mask中存在现有学习框未充分利用的严格几何信息，但376条破坏不支持无条件换框。本次不是新训练模型、不是V99能力已内化，metricbest仍5616/4511，距4754尚差243条。下一项源码正在fresh SOURCE_ONLY审查，计划以完整Mask范围作几何参考，对照原粗框参考；两组共同保留hidden8并重置output2，原粗框保留6维先验，参考改变采样位置及六面坐标系，监督预算保持。39条实际空支撑说明网络无法形成参考时需要保留原粗框先验；这是有效性处理，不是GT质量/双源/版本选择。该新实验尚未部署或获得精度，先真实两步GPU预检，再固定预算训练。研究目标ACTIVE_UNMET，PV-Ground主线、原生唯一bbs及全256候选不变。
'''
"""+source[end:]
start=source.index("names=[path.name")
end=source.index("assert len(names)==len(set(names))",start)
source=source[:start]+"""names=['publish_terminal.py','prepare_terminal_publication.py','record_terminal_acceptance.py',
       'TERMINAL_EXECUTION_TRACKER.md','FORMAL_CLOSED_SESSIONS.json','PUBLICATION_EXECUTION_SESSIONS.json',
       'FORMAL_ESTIMATE_EXPIRED_PROGRESS.json','formal_wait.json',
       'analysis/SUMMARY.json','analysis/NARRATIVE_REPORT.md','analysis/TERMINAL_ACCEPTANCE.json',
       'analysis/TERMINAL_REVIEW_CALL.json','analysis/EXPERIMENT_AUDIT.json','analysis/EXPERIMENT_AUDIT.md',
       'complete/formal_status.json','complete/formal.exit','complete/formal/receipt.json',
       'complete/formal/CPU_SUMMARY.json','complete/formal/FAILURE_BREAKDOWN.json',
       'complete/formal/imports.json','complete/formal/load.json',
       'PUBLISH_TERMINAL_REVIEW.json','PUBLISH_TERMINAL_REVIEW.md']
"""+source[end:]
source=source.replace("'/root/autodl-tmp/mcln_published_evidence_20261005/pvground_mask_extent_diagnostic_20261006'",
    "'/root/autodl-tmp/mcln_published_evidence_20261005/pvground_mask_extent_diagnostic_20261006/terminal'")
source=source.replace(' Protected4511 completeMask extent read-only diagnostic: actual raw8 M0 accepted, formal9508 running; no new accuracy or weights.',
    ' Protected4511 fullMask diagnostic9508 closed: offline exact4848/q4781, no trainedbest promotion; fresh terminal WARN0blocks.')
source=source.replace("Record accepted raw Mask extent preflight and actual formal launch","Record closed9508 Mask extent diagnostic and fresh integrity audit")
start=source.index("record=dict(time_cst=")
end=source.index("(local/'terminal_publication.json')",start)
source=source[:start]+"""record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.81',heads=heads,
            handoff_sha256=digest,four_local_and_remote_equal=True,github_main=heads[0],
            new_trained_accuracy_result=False,offline_extent_diagnostic_complete=True,
            protected_trained_model_hits=[5616,4511],offline_exact_hits=[5598,4848],offline_quantile_hits=[5590,4781],
            terminal_verdict=audit['verdict'],payload_count=len(payloads),raw_npz_published=False)
"""+source[end:]
start=source.index("state.update(time_cst=")
end=source.index('state_path.write_text',start)
source=source[:start]+"""state.update(time_cst=record['time_cst'],latest_publication=str(local/'terminal_publication.json'),
             status='MASK_EXTENT_CLOSED_ACCEPTED_PUBLISHED_REFERENCE_SOURCE_REVIEW',owned_gpu_job_active=False,
             active_reviewer='/root/pvg_mask_reference_source_review',
             handoff_section=record['section'],handoff_sha256=digest,published_heads=heads,
             next_action='Complete new reference SOURCE review; actual2-update checks before training. Best4511 unchanged.')
"""+source[end:]
source=source.replace('Doc80 actual8row rawCPU M0 accepted and sameQuery9508 read-only diagnostic launched; one formal observer near estimated finish.',
    'Doc81 actual9508 extent diagnostic closed and fresh audit accepted; offline4848/q4781 not trainedbest; source reference pair pending.')
source=source.replace("Reports=[", "Reports=[")
source=source.replace("new.count(b'## 20.376.80 ')==1","new.count(b'## 20.376.81 ')==1")
ast.parse(source)
(root/'publish_terminal.py').write_text(source,encoding='utf-8')
print('TERMINAL_PUBLICATION_PREPARED_NOT_EXECUTED')
