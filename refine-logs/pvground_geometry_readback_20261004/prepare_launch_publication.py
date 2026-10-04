"""Prepare publication of the actual source gate and launched preflight."""
import ast
from pathlib import Path

local = Path(__file__).resolve().parent
face = local.parent / 'pvground_face_conditioned_20261004'
target = face / 'publish_readback_preflight_launch.py'
assert not target.exists()
source = (face / 'publish_terminal_audit.py').read_text(encoding='utf-8')
source = source.replace("assert not (local / 'terminal_audit_publication.json').exists()",
                        "assert not (local / 'readback_launch_publication.json').exists()")
source = source.replace("previous = json.loads((local / 'terminal_pending_publication.json').read_bytes())",
                        "previous = json.loads((local / 'terminal_audit_publication.json').read_bytes())")
start = source.index('call = json.loads(')
end = source.index('new = old + section.encode', start)
replacement = """review = json.loads((readback / 'READBACK_FULL_SOURCE_REVIEW.json').read_bytes())
launch = json.loads((readback / 'readback_preflight_launch.json').read_bytes())
assert review['verdict'] == 'PASS' and not review['blocking_findings']
assert review['execution_scope'] == 'SOURCE_ONLY'
assert launch['execution_status'] == 'LAUNCHED_NOT_COMPLETED'
payloads = {}
for name in ('READBACK_FULL_SOURCE_REVIEW.md', 'READBACK_FULL_SOURCE_REVIEW.json',
    'ACTUAL_FULL_SOURCE_REVIEW_RESPONSE.txt', 'FULL_SOURCE_REVIEW_CALL.json',
    'finalize_full_source_review.py', 'readback_preflight_launch.json',
    'readback_preflight_resource_check.json', 'readback_remote_source_receipt.json',
    'SEALED_READBACK_SOURCE_PORT.json', 'evidence_hidden_preflight_spec.json',
    'evidence_visible_preflight_spec.json', 'wait_readback_preflight_authorized.py',
    'collect_readback_preflight.py', 'prepare_launch_publication.py'):
    payloads[readback_prefix + name] = (readback / name).read_bytes()
payloads[face_prefix + 'publish_readback_preflight_launch.py'] = (local / 'publish_readback_preflight_launch.py').read_bytes()
assert not any('/.aris/' in name or name.endswith('.pth') for name in payloads)
stamp = datetime.datetime.now().astimezone().isoformat()
section = f'''

## 20.376.53 固定4506几何回写完整预检源码审查通过并实际启动（{stamp}）

完整32份来源已由新的Codex原生审查员实际直接读取，SOURCE_ONLY PASS，无正确性阻断或非阻断项，same-family/provisional。审查员可选本地AST调用因Python启动器失败未执行；主端此前已实际执行Python3.7 AST检查，二者不混淆。PASS仅覆盖CPU构建和两步预检源码，不是CPU/GPU实测通过、正式训练授权见证或精度结果。实际最终响应与私有trace已保存；旧部分源码审查不替代这次完整审查。

预检控制器实际于{launch['time_cst']}启动，PID {launch['process'].split()[0]}，根目录为{launch['root']}。先evidence_hidden，后evidence_visible，每组CPU严格构建后仅做两次真实GPU更新。模型使用已保留的4506平铺分布头及原G/官方PV父链，父模型与几何保持冻结/eval，新回写单元学习。两组只改变44维证据是否可见，全256候选、完整文本、相同参数及原生唯一评分保留。源覆盖位于全新隔离目录，原已验证源码未改；sealed源SHA为{launch['sealed_source']['source_port_sha256']}。

启动时实测GPU无计算进程，旧控制器已关闭，数据盘剩余{launch['resources']['directory_free_bytes']}字节、系统盘{launch['resources']['system_free_bytes']}字节，均只代表启动快照。预检不创建磁盘权重。唯一只读观察器已安排首次22:25 CST附近、之后240秒复查，不重复SSH轮询、不重启任务。此发布尚未读取实际预检终态；不能把已启动写成两组通过或正式训练已经开始。保留最佳5616/4506及必要父链，ScanRefer目标5615/4754仍未达，无新Nr/Sr结果。
'''
"""
source = source[:start] + replacement + source[end:]
source = source.replace("new_directories = [readback_prefix + 'runtime_bundle']", 'new_directories = []')
source = source.replace('Record completed face integrity audit and full readback preflight source',
                        'Record passed full readback source gate and actual two-arm preflight launch')
source = source.replace("section='20.376.52'", "section='20.376.53'")
source = source.replace("predecessor=str(local / 'terminal_pending_publication.json')", "predecessor=str(local / 'terminal_audit_publication.json')")
source = source.replace('full_readback_source_review_pending=True', "full_readback_source_review_pending=False, full_readback_source_review='PASS_SOURCE_ONLY', actual_preflight_launched=True")
source = source.replace("(local / 'terminal_audit_publication.json').write_text", "(local / 'readback_launch_publication.json').write_text")
source = source.replace("latest_publication=str(local / 'terminal_audit_publication.json')", "latest_publication=str(local / 'readback_launch_publication.json')")
source = source.replace("github_publication_predecessor=str(local / 'terminal_audit_publication.json')", "github_publication_predecessor=str(local / 'readback_launch_publication.json')")
source = source.replace("handoff_section='20.376.52'", "handoff_section='20.376.53'")
source = source.replace('doc52.', 'doc53.').replace('doc52 fourlocal', 'doc53 fourlocal')
source = source.replace('full readback source review pending', 'full source gate PASS_SOURCE_ONLY; actual preflight launched, terminal pending')
source = source.replace('full32-file readback source gate pending', 'full32-file source gate PASS; actual preflight launched, terminal pending')
source = source.replace('readback full source gate pending', 'readback full source gate passed; actual preflight pending')
source = source.replace('Revised readback/factory AST-only, full source/runtime gate pending.', 'Full source gate passed; actual readback preflight launched, runtime terminal pending.')
ast.parse(source)
target.write_text(source, encoding='utf-8')
print('ACTUAL_READBACK_LAUNCH_PUBLICATION_PREPARED_ONLY')
