import ast
import datetime
import hashlib
import json
from pathlib import Path
from geometry_result_metrics import summarize

root=Path(__file__).resolve().parent
parent=root.parent/'pvground_boundary_distribution_20261004/complete/distribution/formal'
raw=(parent/'rows.jsonl').read_bytes()
receipt=json.loads((parent/'receipt.json').read_bytes())
assert hashlib.sha256(raw).hexdigest()==receipt['rows_sha256']
rows=[json.loads(line) for line in raw.decode('utf-8').splitlines()]
metric=summarize(rows,receipt['metrics']['bbs'])
assert metric['rows']==9508 and [metric['rec_hits25'],metric['rec_hits50']]==[5616,4506]
assert metric['same_query_refinement']['0.5']==dict(coarse_hits=4495,final_hits=4506,repairs=20,damages=9,net=11)
for name in ('collect_formal_authorized.py','geometry_result_metrics.py','analyze_closed_formal.py'):
    ast.parse((root/name).read_text(encoding='utf-8'),filename=name)
record=dict(time_cst=datetime.datetime.now().astimezone().isoformat(),status='HISTORICAL_PARENT_CPU_RECOUNT_PASS',
    historical_only=True,new_fit_result=False,model_replayed=False,weights_loaded=0,rows=9508,
    parent_hits=[5616,4506],parent_refinement=metric['same_query_refinement'],
    cpu_box_threshold_changes=metric['cpu_box_threshold_changes'],parent_rows_sha256=receipt['rows_sha256'])
(root/'METRIC_TOOLS_CHECK.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record),flush=True)
