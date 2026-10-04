"""Keep the actual supplemental native review message in the private trace."""
import datetime
import json
from pathlib import Path

local = Path(__file__).resolve().parent
trace = local/'.aris/traces/experiment-bridge/2026-10-04_boundary_source_run01'
message = ('请将以下已准备的 lifecycle 文件作为实际补充 primary paths 读入并纳入 reviewed_files；原始请求列出的实现与规范保持不变。'
    '目录 C:\\Users\\gb\\.codex\\tmp\\pvground_boundary_distribution_20261004 下：prepare_lifecycle_helpers.py、lifecycle_preparation.json、'
    'observe_preflight_authorized.py、wait_preflight_authorized.py、collect_preflight_terminal.py、launch_formal_authorized.py。'
    '它们是既有授权观察/收取/启动工具的最小路径适配，均未执行远端动作。正式启动要求两组实际 preflight，并继承 GPU lock、'
    '原G保护和非最佳自有权重清理；请核对具体 mutation 路径。补充消息会保留实际工具记录。')
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),actual_tool='collaboration.send_message',
    actual_arguments=dict(target='/root/pvg_boundary_distribution_source_review',message=message),
    actual_tool_receipt={},source_primary_files_changed=False,remote_calls=0)
with (trace/'002-lifecycle-followup.request.json').open('x',encoding='utf-8') as stream:
    stream.write(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
state = dict(time_cst=record['time_cst'],status='SOURCE_IMPLEMENTED_ACTUAL_FRESH_REVIEW_RUNNING',
    fresh_source_task='/root/pvg_boundary_distribution_source_review',
    native_preflight_started=False,native_preflight_result_available=False,
    fit_started=False,formal_metrics_available=False,original_g_retained=True,
    primary_score='native last/bbs',primary_threshold=0.5,
    planned_fit_rows_per_arm=29778,planned_updates_per_arm=3723,
    head_parameters={'residual':400614,'distribution':456102},effective_batch=8,seed=2027,
    goal_status='ACTIVE_UNMET',weights_created=0,weights_deleted=0,
    next_steps=['Receive actual fresh code review','Resolve concrete blockers if any',
        'Run the actual serial two-step boundary probes','Only after success launch bounded comparison'])
(local/'active_continuation_state.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(actual_followup_preserved=True,status=state['status'],native_forwards=0,remote_calls=0)))
