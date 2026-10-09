"""Record local static preparation; no SSH, GPU or result claims."""
import ast
import datetime
import hashlib
import json
from pathlib import Path


root = Path(__file__).resolve().parent
stamp = datetime.datetime.now().astimezone().isoformat()
sources = {}
for name in ('analyze_identity_campaign.py', 'check_identity_terminal_recovery.py'):
    path = root / name
    source = path.read_text(encoding='utf-8')
    ast.parse(source, filename=str(path), feature_version=(3, 7))
    sources[name] = dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), python37_ast=True)
split = Path('C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/scanrefer_local_visual_mesh_pair_20260906_v1/split_protocol.json')
assert hashlib.sha256(split.read_bytes()).hexdigest() == '06d0b20a848be97827a4e0257b074afcfb91a04fc8e448f7de34eef453294461'
partitions = json.loads(split.read_bytes())['row_ids']
assert len(partitions['fit']) == 29778 and len(partitions['holdout']) == 6887
record = dict(status='SOURCE_PREPARED_STATIC_ONLY', generated_cst=stamp, sources=sources,
    split_protocol=str(split), original_campaign_pid=973371, original_collector_session=9320,
    original_collector_wait_cell13_consumed=True, new_remote_query=False,
    nn_source_changed=False, cpu_metric_recount_executed=False, terminal_recovery_executed=False,
    fresh_source_review_status='PENDING', best_updated=False, goal_complete=False,
    initial_preparation_attempt='Stopped before metadata writes because guessed local split path was absent; corrected to existing archived source path, no remote operation.')
path = root / 'TERMINAL_CHECK_PREPARATION.json'
assert not path.exists()
path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
tracker = root / 'EXPERIMENT_TRACKER.md'
with tracker.open('a', encoding='utf-8') as stream:
    stream.write('\n终点检查源码已准备并通过Python3.7语法解析，仅为静态检查。完整CPU重算、正式文件完整模型／Adam恢复、实际结果审查均未执行；不改变当前最佳或正在运行的原实验。\n')
state_path = Path('C:/Users/gb/.codex/tmp/pvground_query_supported_geometry_20261005/active_continuation_state.json')
state = json.loads(state_path.read_bytes())
state['terminal_validation_preparation'] = dict(path=str(path), status=record['status'], fresh_source_review='PENDING', actual_results=False)
state['support_identity_experiment']['active_wait_exec_cell'] = None
state['current_goal_turn_classification'] = 'PROGRESS_PREPARED_REQUIRED_FORMAL_CPU_RECOUNT_AND_TERMINAL_RECONSTRUCTION_SOURCE'
state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
daily = Path('C:/Users/gb/memory/2026-10-09.md')
with daily.open('a', encoding='utf-8') as stream:
    stream.write('\nPV-Ground ' + stamp + ': 终点CPU重算与正式8状态delta完整1299状态／Adam恢复源码已准备，Python3.7 AST通过；未执行实际结果重算或GPU恢复，未改原973371训练。原collector9320本地等待句柄仍存在，旧functions13已消费，12:12首次查询不变；需要鲜独立上下文源码审查与终态后结果审查，不提前晋级。\n')
print(json.dumps(record), flush=True)
