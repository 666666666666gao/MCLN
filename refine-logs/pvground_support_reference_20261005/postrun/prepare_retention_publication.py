"""Prepare an append-only source/tooling publication; no remote operations."""
import ast
from pathlib import Path

root = Path(__file__).resolve().parents[1]
destination = root / 'postrun/publish_retention_tools.py'
assert not destination.exists()
source = (root / 'postrun/publish_fit_launch.py').read_text(encoding='utf-8')


def change(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


source = change(source, "local/'preflight_launch_publication.json'", "local/'fit_launch_publication.json'")
source = change(source, "local/'SOURCE_REVIEW.json'", "local/'RETENTION_REVIEW.json'")
source = change(source, "local/'SOURCE_REVIEW_CALL.json'", "local/'RETENTION_REVIEW_CALL.json'")
source = change(source, "assert not (local/'fit_launch_publication.json').exists()", "assert not (local/'retention_tools_publication.json').exists()")
begin = source.index("proofs={arm:")
end = source.index("client=paramiko.SSHClient()", begin)
source = source[:begin] + '''for entry in review['reviewed_files']:
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
payloads[prefix+'.gitattributes']=b'** -text\\n'
assert all('.aris' not in name and not name.endswith(('.pt','.pth')) for name in payloads)
''' + source[end:]
source = change(source, 'Own/fused support reference two-step GPU sanity passed; same-start formal pair launched, no new accuracy result.', 'Support-reference closed-result and strict-best retention tools SOURCE reviewed; formal fit unchanged, tools not executed.')
source = change(source, 'Record passed reference sanity and launch paired formal comparison', 'Record reviewed reference closeout and strict-best retention tools')
source = source.replace("'section':'20.376.77'", "'section':'20.376.78'")
source = source.replace("local/'fit_launch_publication.json'", "local/'retention_tools_publication.json'")
# The previous publication is the existing formal-launch receipt.
source = change(source, "old_publication=json.loads((local/'retention_tools_publication.json').read_bytes())", "old_publication=json.loads((local/'fit_launch_publication.json').read_bytes())")
source = change(source, "status='SUPPORT_REFERENCE_FORMAL_FIT_ACTIVE_PUBLISHED',owned_gpu_job_active=True,active_reviewer=None,", "status='SUPPORT_REFERENCE_FORMAL_FIT_ACTIVE_PUBLISHED',owned_gpu_job_active=True,active_reviewer=None,\n    support_reference_retention_source_review_pending=False,support_reference_retention_source_review_complete=True,\n    support_reference_postrun_tools_prepared=True,support_reference_postrun_tools_executed=False,")
source = change(source, "workspace/'memory/2026-10-05.md'", "workspace/'memory/2026-10-06.md'")
source = change(source, 'Doc77 own/fused support reference actual two-step sanity passed and formal pair launched; no new accuracy result.', 'Doc78 closed-result/strict-best retention tools SOURCE reviewed and published, not executed; active formal fit unchanged and no new accuracy result.')
ast.parse(source)
destination.write_text(source, encoding='utf-8')
print(destination)
