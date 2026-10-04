"""Prepare a small publication from the proven predecessor publisher structure."""
import ast
from pathlib import Path

local = Path(__file__).resolve().parent
target = local / 'publish_terminal_audit.py'
assert not target.exists()
source = (local / 'publish_terminal_pending_audit.py').read_text(encoding='utf-8')
source = source.replace('terminal_pending_publication.json', 'terminal_audit_publication.json')
source = source.replace("previous = json.loads((readback / 'review_publication.json').read_bytes())",
                        "previous = json.loads((local / 'terminal_pending_publication.json').read_bytes())")
start = source.index('face_prefix = ')
end = source.index('new = old + section.encode', start)
replacement = """face_prefix = 'refine-logs/pvground_face_conditioned_20261004/'
readback_prefix = 'refine-logs/pvground_geometry_readback_20261004/'
audit = json.loads((local / 'analysis/EXPERIMENT_AUDIT.json').read_bytes())
assert audit['verdict'] == 'WARN' and not audit['blocking_issues']
assert audit['primary_files_read'] == 70
call = json.loads((readback / 'FULL_SOURCE_REVIEW_CALL.json').read_bytes())
assert call['status'] == 'ACTUALLY_CALLED_RUNNING'
payloads = {}
for path in sorted((local / 'analysis').glob('EXPERIMENT_AUDIT*')):
    payloads[face_prefix + 'analysis/' + path.name] = path.read_bytes()
for name in ('ACTUAL_TERMINAL_REVIEWER_CALL.json', 'ACTUAL_TERMINAL_REVIEWER_RESPONSE.txt',
             'finalize_terminal_audit.py', 'publish_terminal_audit.py', 'prepare_audit_publication.py'):
    payloads[face_prefix + name] = (local / name).read_bytes()
for name in ('run_readback_preflight.py', 'readback_preflight_controller.py',
    'create_remote_readback_source.py', 'launch_readback_preflight_authorized.py',
    'prepare_preflight_bundle.py', 'EXPERIMENT_PLAN_READBACK.md',
    'evidence_hidden_preflight_template.json', 'evidence_visible_preflight_template.json',
    'PREFLIGHT_BUNDLE_SOURCE_CHECK.json', 'prepare_full_source_review.py',
    'FULL_SOURCE_REVIEW_REQUEST.txt', 'FULL_SOURCE_REVIEW_BINDINGS.json',
    'FULL_SOURCE_AST_CHECK.json', 'FULL_SOURCE_REVIEW_CALL.json', 'record_full_source_call.py'):
    payloads[readback_prefix + name] = (readback / name).read_bytes()
for path in sorted((readback / 'runtime_bundle').iterdir()):
    payloads[readback_prefix + 'runtime_bundle/' + path.name] = path.read_bytes()
assert all(len(raw) < 100000000 for raw in payloads.values())
assert not any('/.aris/' in name or name.endswith('.pth') for name in payloads)
stamp = datetime.datetime.now().astimezone().isoformat()
section = f'''

## 20.376.52 六面终态独立审计完成与固定最佳几何回写预检源码（{stamp}）

新的experiment-audit已经实际完成：WARN、无阻断项，same-family/provisional；A/B/C/F为PASS，D/E为WARN。独立读取70份主文件、50287条NDJSON并完成CPU重算，正式last/bbs仍为face5615/4496、flat5616/4506，目标5615/4754未达。WARN涉及已继承但未调用的辅助函数、单seed开发验证范围、两架构容量不同、跨进程数值差异，以及Mask/oracle仅保存标量复核、未重放原始Mask与全候选框。未发现GT伪造、指标拼接或不存在的结果。没有因此新增训练或重评估；精度与权重清理仍以§51真实终态为准。

下一项选择保留的4506平铺分布头作为几何提供者，固定官方PV、原G及几何头参数与eval状态；仅新回写单元学习。两组保持相同参数、完整文本、Query、六面角色和256候选，仅改变44维几何证据可见/置零。原生最终语义子头在精修后只调用一次，Mask与对比投影保留原路径。现已补齐实际CPU/两步GPU预检caller、严格三段权重工厂、孤立源覆盖、顺序控制器和部署入口；新32份完整来源审查已实际调用，当前仍待终态。旧部分源码PASS不覆盖新字节；此处只有Python3.7语法与源码准备，未CPU构建、未GPU更新、未启动正式回写训练，更没有新精度结果。

预检将检查零残差原生一致、框/Mask/对比冻结、语义头单次调用、实际bbs分数/排序/梯度、独立末层CE+G梯度、两步更新与内存模型/优化器恢复。不会创建临时磁盘权重。未来正式对照预算须同时记录有效batch与更新次数：29778条fit各一次、batch8，3722个完整batch加尾批2，共3723次更新；改变有效batch会改变同遍历下更新次数，不以学习率缩放代替预算核对。当前无Nr/Sr、教师、质量损失或部署双源结果。
'''
"""
# Nested section literal above is stored as text, not executed by this preparer.
source = source[:start] + replacement + source[end:]
source = source.replace("new_directories = sorted({str(Path(name).parent).replace('\\\\', '/') for name in payloads\n    if name.startswith(face_prefix + 'complete/') or name.startswith(face_prefix + 'analysis/')},\n    key=lambda name: (name.count('/'), name))",
                        "new_directories = [readback_prefix + 'runtime_bundle']")
source = source.replace('Record face boundary terminal and prepare isolated geometry readback checks',
                        'Record completed face integrity audit and full readback preflight source')
source = source.replace("section='20.376.51'", "section='20.376.52'")
source = source.replace("predecessor=str(readback / 'review_publication.json')", "predecessor=str(local / 'terminal_pending_publication.json')")
source = source.replace('terminal_audit_pending_at_publication=True', "terminal_audit_pending_at_publication=False, terminal_integrity_verdict='WARN', terminal_integrity_blockers=0, full_readback_source_review_pending=True")
source = source.replace("handoff_section='20.376.51'", "handoff_section='20.376.52'")
source = source.replace('doc51. Audit pending at this snapshot', 'doc52. Actual terminal audit WARN/no blockers; full readback source review pending')
source = source.replace('Fresh terminal audit pending; no revised readback CPU/GPU/accuracy pass.',
                        'Actual terminal audit WARN/no blockers, full32-file readback source gate pending; no CPU/GPU/accuracy pass.')
source = source.replace('CPU recount complete, fresh integrity audit pending.', 'CPU recount and fresh integrity audit WARN/no blockers complete; readback full source gate pending.')
source = source.replace('doc51 fourlocal+remote raw equal.', 'doc52 fourlocal+remote raw equal.')
ast.parse(source)
target.write_text(source, encoding='utf-8')
print('AUDIT_PUBLICATION_PREPARED_ONLY')
