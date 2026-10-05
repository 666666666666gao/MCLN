"""Independently recount a closed control's native9508 rows; no model replay."""
import hashlib
import json
from pathlib import Path

from geometry_result_metrics import paired, summarize


root=Path(__file__).resolve().parent
intake=json.loads((root/'CONTROL_CLOSED_INTAKE.json').read_bytes())
assert intake['status']=='CLOSED_CONTROL_RECEIPTS_READ' and intake['formal_rows']==9508
assert not intake['downloaded_weights'] and not intake['inference_or_optimizer_replayed']
directory=root/'closed_control_receipts'
raw=(directory/'control/formal/rows.jsonl').read_bytes()
identity=intake['files']['control/formal/rows.jsonl']
assert len(raw)==identity['bytes'] and hashlib.sha256(raw).hexdigest()==identity['sha256']
rows=[json.loads(line) for line in raw.decode('utf-8').splitlines()]
assert len(rows)==9508 and [row['row_id'] for row in rows]==list(range(9508))
assert all(row['same_forward_geometry_exact'] and row['native_head_calls']==1 and row['diagnostic_native_head_replay_calls']==0 for row in rows)
receipt=json.loads((directory/'control/formal/receipt.json').read_bytes())
metrics=summarize(rows,receipt['metrics']['bbs'])
parent=root.parent/'pvground_query_supported_geometry_20261005/complete/query_supported/formal'
parent_raw=(parent/'rows.jsonl').read_bytes()
parent_receipt=json.loads((parent/'receipt.json').read_bytes())
assert hashlib.sha256(parent_raw).hexdigest()==parent_receipt['rows_sha256']
parent_rows=[json.loads(line) for line in parent_raw.decode('utf-8').splitlines()]
parent_metrics=summarize(parent_rows,parent_receipt['metrics']['bbs'])
assert [parent_metrics['rec_hits25'],parent_metrics['rec_hits50']]==[5614,4509]
record=dict(status='CLOSED_CONTROL_FULL_ROWS_CPU_RECOUNTED',formal_finished_cst=receipt['time_cst'],
            formal_rows=9508,control=metrics,parent=parent_metrics,
            versus_protected_parent={str(t):paired(parent_rows,rows,t) for t in (.25,.5)},
            formal_rows_sha256=identity['sha256'],full_pair_complete=False,
            fresh_terminal_integrity_review_pending=True,best_weight_promoted=False,
            inference_or_optimizer_replayed=False,weights_deleted=0,
            scope='Single closed native_gt control; member_gt results pending. Native9508 is development validation. Box thresholds CPU-recomputed; candidate and Mask summaries are scalar recounts. No new multi-dataset or significance claim.')
assert not (root/'CONTROL_CPU_RECOUNT.json').exists()
(root/'CONTROL_CPU_RECOUNT.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=record['status'],control_hits=[metrics['rec_hits25'],metrics['rec_hits50']],
      versus_parent=record['versus_protected_parent'],strategy_result_pending=True)),flush=True)
