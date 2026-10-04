"""Make the concrete local/Git publication after two real SSH failures."""
import ast
from pathlib import Path


local=Path(__file__).resolve().parent
source=(local/'publish_quality_terminal_20261005.py').read_text(encoding='utf-8')
assert (local/'PUBLICATION_CAPACITY_ATTEMPT_2.json').exists()
assert not (local/'PUBLICATION_CAPACITY.json').exists()


def replace_once(old,new):
    global source
    assert source.count(old)==1,old
    source=source.replace(old,new)


replace_once('import paramiko\n','')
replace_once("receipt_path = local/'terminal_publication.json'","receipt_path = local/'terminal_local_publication.json'")
start=source.index('names = ['); end=source.index("for directory in ('complete','analysis'):")
source=source[:start]+"names = ['fit_wait.json','CLOSED_RESOURCES.json','check_closed_resources.py',\n    'PUBLICATION_CAPACITY_ATTEMPT_1.json','PUBLICATION_CAPACITY_ATTEMPT_2.json','CONSOLE_STATE_UNAVAILABLE.json',\n    'check_publication_capacity.py','LEGACY_MASK_QUALIFIED_MATCH_ROLE.json','recount_legacy_mask_match_roles.py',\n    'prepare_local_terminal_publication.py','publish_quality_terminal_20261005.py',Path(__file__).name]\n"+source[end:]
start=source.index('client = paramiko.SSHClient(); client.load_system_host_keys()')
end=source.index('heads=[]',start)
source=source[:start]+'''for name,raw in payloads.items():
    for repo in repos[:2]:
        destination=repo/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(raw)
for path in copies:
    path.write_bytes(new)
# No SSH writes or remote document guard change in this local publication.
'''+source[end:]
start=source.index("guard=workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py'")
end=source.index('assert all(path.read_bytes()==new for path in copies)',start)
source=source[:start]+"guard=workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py'\nassert guard.read_bytes().count(previous['handoff_sha256'].encode())==1\n"+source[end:]
replace_once('four_local_and_remote_equal=True','four_local_equal=True,remote_sync_pending=True')
replace_once("record['remote_complete_reuses_data_disk_originals']=True","record['remote_complete_reuses_data_disk_originals']=False\nrecord['remote_handoff_expected_sha256']=previous['handoff_sha256']\nrecord['remote_guard_unchanged']=True")
replace_once("status='QUALITY_TERMINAL_NEGATIVE_PUBLISHED'","status='QUALITY_TERMINAL_LOCAL_GITHUB_PUBLISHED_REMOTE_PENDING'")
replace_once("current_goal_turn_classification='PROGRESS_ACTUAL_QUALITY_TERMINAL_PUBLISHED'","current_goal_turn_classification='PROGRESS_ACTUAL_QUALITY_TERMINAL_LOCAL_PUBLISHED_REMOTE_PENDING'")
replace_once("state['closed_native_sessions'].append(84556)","assert 84556 in state['closed_native_sessions']\nstate['remote_sync_pending']=True\nstate['active_reviewer']=None\nstate['remote_handoff_sha256']=previous['handoff_sha256']")
replace_once('Latest actual publication terminal_publication.json','Latest actual local/Git publication terminal_local_publication.json; remote sync pending')
replace_once('rawfourlocal+remote exact. GoalACTIVE_UNMET','rawfourlocal exact, remote sync pending and guard unchanged. GoalACTIVE_UNMET')
old='鉴于实际系统盘仅约48MiB，远端发布complete证据目录直接链接数据盘现存闭合结果，不再复制约26.9MB日志到系统盘；逐份验证内容与本地收取记录相同。Git的本地证据与四份原始交接文档仍按原协议保留。发布前一次SSH banner连接失败在任何远端探测或写入前退出，单独记录PUBLICATION_CAPACITY_ATTEMPT_1.json；之后实际容量检查与发布收据分开，不将连接失败写成成功探测。'
new='本节先完成四份本地原始交接文档与GitHub发布，远端同步尚未执行。两次SSH在banner握手阶段退出，均未执行远端容量探测或写入；控制台浏览器也因app-server缺失而无法读取，不能据此猜测服务器已关机。失败见PUBLICATION_CAPACITY_ATTEMPT_1/2.json与CONSOLE_STATE_UNAVAILABLE.json，容量结果文件未生成。远端文档仍应为§20.376.59，同步guard保持该SHA不变，不把本地发布写成远端已更新。SSH恢复后将核对实际挂载位置，让远端complete证据引用数据盘原始结果，避免再次将约26.9MB日志复制到已仅剩约48MiB的系统盘。'
replace_once(old,new)
source=source.replace('"""Publish only the actual closed result and completed integrity review."""','"""Publish actual terminal evidence locally and to Git, with remote sync pending."""')
destination=local/'publish_quality_terminal_local_20261005.py'
assert not destination.exists()
ast.parse(source,feature_version=(3,7))
destination.write_text(source,encoding='utf-8')
print('Local terminal publisher prepared and AST checked; remote publisher superseded/unexecuted.')
