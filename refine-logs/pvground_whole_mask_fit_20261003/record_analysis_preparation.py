"""Record real local tool verification without changing the active experiment."""
import datetime
import hashlib
import json
from pathlib import Path


root = Path(__file__).parent
check = json.loads((root/'terminal_tools_source_check.json').read_bytes())
assert check['status'] == 'SOURCE_READY_NOT_CURRENT_TERMINAL_RESULT'
for name, digest in check['tool_sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
state = json.loads((root/'active_formal_continuation_state.json').read_bytes())
assert state['sole_waiter_native_session'] == 5205
state['time_cst'] = datetime.datetime.now().astimezone().isoformat()
state['waiter_status'] = 'VERIFIED_LIVE_NATIVE_WAIT_180S'
state['waiter_last_native_poll_chunk'] = '59bee8'
state['waiter_last_native_poll_session'] = 5205
state['terminal_analysis_preparation'] = check
state['terminal_collection_executed'] = False
state['terminal_analysis_executed'] = False
state['remote_observation_available'] = False
state['active_source_changed'] = False
state['goal_achieved'] = False
(root/'active_formal_continuation_state.json').write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')
note = '\n### PV完整范围来源对照：终态工具准备（{}）\n\n'.format(state['time_cst'])
note += ('原唯一原生观察器5205实际180秒poll返回LIVE（chunk59bee8），首个SSH观察仍按21:54:57计划。'
    '未新开观察器／SSH／模型／更新。新增collect_terminal.py与analyze_terminal.py仅在真实complete闭环之后执行；'
    '当前未收取／分析F终态。Py3.7语法检查及历史tail_fused正式9508行实际只读工具验证通过，'
    '重现旧4412→4406、修复15／破坏21，bbs／bbf选中框CPU阈值差0；这些不是本轮新成绩。'
    '当前工具哈希与诊断详见'+str(root/'terminal_tools_source_check.json')+'。活动source、batch8、29778各一次／3723更新、'
    'native＋G、400614参数头、全部256候选均未改。当前最佳仍原G5615／4495，目标未完成。\n')
daily = Path('C:/Users/gb/memory/2026-10-03.md')
assert daily.exists()
with daily.open('a', encoding='utf-8') as stream:
    stream.write(note)
print(json.dumps(dict(status='PREPARATION_RECORDED', waiter_session=5205,
    remote_observation_available=False, current_pair_terminal_result_available=False,
    model_updates=0, goal_achieved=False)))
