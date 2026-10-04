"""Record the actual completed fresh audit and close stale live-handle pointers."""
import datetime
import hashlib
import json
from pathlib import Path

local = Path(__file__).resolve().parent
audit = json.loads((local / 'analysis/EXPERIMENT_AUDIT.json').read_bytes())
assert audit['verdict'] == 'WARN' and not audit['blocking_issues']
assert audit['primary_files_read'] == 70
response = '''审计完成：**WARN，同系暂定**；A/B/C/F PASS，D/E WARN，无阻断项。

70 个主文件和 50,287 条 NDJSON 已核验，主产物未改动。正式 `bbs`：face **5615/4496**，flat **5616/4506**；目标未达，收据支持保留 flat 最优权重。报告明确限定单种子开发集、容量变化及标量复核范围。

已生成：

- [审计报告](</D:/Program Files/UserCache/gb/codex/tmp/pvground_face_conditioned_20261004/analysis/EXPERIMENT_AUDIT.md>)
- [结构化审计](</D:/Program Files/UserCache/gb/codex/tmp/pvground_face_conditioned_20261004/analysis/EXPERIMENT_AUDIT.json>)
- 同目录 CPU 重算脚本、结果及完整读取身份清单。

输出格式、56 处证据引用和文件哈希核验通过。
'''
trace = local / '.aris/traces/experiment-audit/2026-10-04_face_terminal_run01'
assert not (local / 'ACTUAL_TERMINAL_REVIEWER_RESPONSE.txt').exists()
for path in (local / 'ACTUAL_TERMINAL_REVIEWER_RESPONSE.txt', trace / '001-terminal-integrity.response.md'):
    path.write_text(response, encoding='utf-8')
stamp = datetime.datetime.now().astimezone().isoformat()
call_path = local / 'ACTUAL_TERMINAL_REVIEWER_CALL.json'
call = json.loads(call_path.read_bytes())
call.update(status='ACTUALLY_COMPLETED', recorded_at=stamp, verdict='WARN', blocking_issues=0,
            reviewer_generated_at=audit['generated_at'])
call_path.write_text(json.dumps(call, indent=2) + '\n', encoding='utf-8')
meta_path = trace / 'run.meta.json'
meta = json.loads(meta_path.read_bytes())
meta.update(status='ACTUALLY_COMPLETED', recorded_at=stamp, verdict='WARN', blocking_issues=0)
meta_path.write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
with (trace / 'events.jsonl').open('a', encoding='utf-8') as stream:
    stream.write(json.dumps(dict(time_cst=stamp, event='actual_final_response_recorded',
        agent='/root/pvg_face_terminal_integrity', verdict='WARN',
        response_sha256=hashlib.sha256(response.encode()).hexdigest())) + '\n')
state_path = local / 'active_continuation_state.json'
state = json.loads(state_path.read_bytes())
state.update(status='FACE_TERMINAL_AUDITED_READBACK_FULL_SOURCE_PREPARATION', time_cst=stamp,
    active_terminal_reviewer=None, fresh_terminal_audit_complete=True, terminal_integrity_verdict='WARN',
    terminal_integrity_blockers=0, native_observer_session_id=None, remote_controller_pid=None,
    sole_live_native_session_id=None, sole_live_exec_cell_id=None, native_observer_closed=True,
    native_publisher_session_closed=88576, controller_alive=False, observed_phase='complete',
    observer_resume_rule='All old observer and publisher handles closed. Do not resume806/84531/88576.',
    next_actions=['Publish actual WARN audit using terminal_pending_publication.json predecessor.',
        'Complete isolated frozen4506 readback runner and fresh full source gate.',
        'Only after source gate perform actual CPU factory and two-update GPU preflight.'])
state_path.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
(local / 'NEXT_CONTINUATION.md').write_text('''# Current continuation — all old observer handles closed

Read active_continuation_state.json and terminal_pending_publication.json first.
Latest publication: main405c6d5, doc§20.376.51, four local+remote raw copies equal.
Face fit completed20:08:33 CST. Formal native last/bbs5615/4496; current best remains5616/4506.
Owned904318-byte worse face terminal was verified and deleted. No weight downloads or failed archives.
Fresh terminal integrity agent completed: WARN, no blockers, same-family/provisional; 70 files/50287 records.
Use analysis/EXPERIMENT_AUDIT.md/.json; actual final response/trace saved. Publish these next.
Exec806/native84531/collector29290/publisher88576 CLOSED. Never resume/restart them; no GPU job currently running.

Readback source is isolated and unexecuted: parent is protected4506 flat distribution, original G+geometry frozen/eval,
only readback learns. Full source factory/preflight runner are being prepared; fresh full source gate pending.
Current drafts do not establish CPU/GPU, actual optimizer restoration, or accuracy.
Keep all256 candidates, one native bbs decision, same Query Box/Mask; no teacher/new quality loss/GT gate.
Future visibility control keeps same capacity/full text/queries but zeros44-D geometry evidence.
Both formal arms must keep29778 rows once, B8/tail2,3723 updates, seed2027; changing effectivebatch alters update budget.
Target5615/4754 remains unmet. Nr/Sr later after ScanRefer structure choice; goalACTIVE_UNMET.
''', encoding='utf-8')
with (Path('C:/Users/gb') / 'memory/2026-10-04.md').open('a', encoding='utf-8') as stream:
    stream.write('\n' + stamp + ' Fresh face terminal audit actually completed WARN/no blockers;70 files/50287 records, native face5615/4496 vs best5616/4506. Scope single seed/development/capacity confound/saved Mask-oracle scalars. Actual response private trace saved. Publisher88576 closed main405c6d5/doc51 raw copies equal; no live GPU job. Readback full source/runtime still pending; goalACTIVE_UNMET.\n')
print(json.dumps(dict(status=state['status'], actual_audit='WARN', blockers=0,
    publisher_closed=88576, live_gpu_job=False, target_met=False)), flush=True)
