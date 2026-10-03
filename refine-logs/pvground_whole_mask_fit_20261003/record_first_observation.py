"""Record the sole waiter's first actual progress snapshot; no new remote query."""
import datetime
import json
from pathlib import Path


root = Path(__file__).parent
observation = json.loads((root/'observation_01.json').read_bytes())
waiting = json.loads((root/'wait.json').read_bytes())
assert observation['controller_alive'] and observation['status']['status'] == 'running'
assert observation['status']['stage'] == 'local_range/train'
assert observation['controller_exit'] is None and waiting['status'] == 'waiting'
progress = next(item['value'] for item in reversed(observation['progress'])
    if item['kind'] == 'PVG_TRAIN_PROGRESS')
rate = progress['cumulative_seconds']/progress['step']
remaining = (progress['total_steps']-progress['step'])*rate
estimated_loop_end = datetime.datetime.fromisoformat(observation['time_cst']) + datetime.timedelta(seconds=remaining)
state = json.loads((root/'active_formal_continuation_state.json').read_bytes())
state.update(time_cst=datetime.datetime.now().astimezone().isoformat(),
    goal_turn_result='PROGRESS_AND_VERIFIED_WAIT', goal_achieved=False,
    waiter_status='VERIFIED_LIVE_NATIVE_WAIT_180S', waiter_last_native_poll_chunk='5188a8',
    remote_observation_available=True, latest_actual_observation=observation,
    next_scheduled_observation_cst=waiting['next_scheduled_cst'],
    next_scheduled_delay_seconds=waiting['next_delay_seconds'],
    measured_seconds_per_update=rate, estimated_current_fit_loop_end_cst=estimated_loop_end.isoformat(),
    estimate_scope='current local_range fit loop only; terminal holdout, formal9508 and whole_range still remain',
    formal_result_available=False, terminal_collection_executed=False, terminal_analysis_executed=False,
    active_source_changed=False)
state['next_steps'] = [
    'Keep the same sole native waiter5205; next measured-time SSH observation is near00:32:51 CST',
    'Do not start another observer or source/control/model run before the actual stage ends',
    'The new terminal tools are source-verified only; execute collection after controller closed complete and four phases verified',
    'Then analyze and run the fresh experiment-audit on actual artifact paths; publish actual results once',
    'Controller retains strict metric best/protected parents and removes verified owned nonbest; keep all256 inference candidates',
    'Condition subsequent support-token/six-face/readback/adaptation design on source-pair results; no normalization rerun']
(root/'active_formal_continuation_state.json').write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')
note = '\n### PV完整范围来源对照：第一次真实进度（{}）\n\n'.format(observation['time_cst'])
note += ('原唯一观察器5205读取实际controller403326／child403332：local_range/train、320／3723更新，'
    'cumulative907.897465秒，约{:.6f}秒／更新；数据盘2621943808字节，尚无owned .pth（未到512步）。'
    '控制器LIVE、无exit，completed[]表示整个train阶段未闭环，不表示没有真实更新。'
    '本轮仍无正式新成绩，最佳原G5615／4495。首阶段样本遍历推算约{}；该估计不含其后6887终态评估、'
    '9508正式评估及完整whole组，不作为整对照完成时间。原waiter已自行计划{}（{}秒后、训练遍历预计结束前约3分钟）。'
    '不新开观察器／训练、不改source/spec；terminal工具已有实际历史行验证但F终态未收取／分析。'
    '本次原生180秒poll返回chunk5188a8／session5205继续LIVE。'
    '目标仍ACTIVE／UNMET，三模块与Nr／Sr尚未成立。\n').format(rate, estimated_loop_end.isoformat(),
        waiting['next_scheduled_cst'], waiting['next_delay_seconds'])
daily = Path('C:/Users/gb/memory/2026-10-03.md')
with daily.open('a', encoding='utf-8') as stream:
    stream.write(note)
print(json.dumps(dict(status='ACTUAL_FIRST_PROGRESS_RECORDED', stage='local_range/train',
    completed_updates=progress['step'], total_updates=progress['total_steps'],
    seconds_per_update=rate, estimated_fit_loop_end_cst=estimated_loop_end.isoformat(),
    next_observation_cst=waiting['next_scheduled_cst'], formal_results_available=False,
    goal_achieved=False)))
