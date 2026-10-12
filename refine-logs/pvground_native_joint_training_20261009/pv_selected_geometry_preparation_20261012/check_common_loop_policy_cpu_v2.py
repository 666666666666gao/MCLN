"""Check extracted future loop policy on actual epoch histories and defined cases."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
path = root / 'source_overlay_v2/main_utils.py'
raw = path.read_bytes()
tree = ast.parse(raw.decode('utf-8'))
nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
         and node.name in ('native_metric_order', 'consecutive_strict_declines')]
assert len(nodes) == 2
scope = {}
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes)), str(path), 'exec'), scope)
declines = scope['consecutive_strict_declines']
order = scope['native_metric_order']
history_path = root.parent / 'face_native_gpu_v2_20261011/normal_training/EPOCH3_ANALYSIS.json'
history = json.loads(history_path.read_bytes())
assert [item['epoch'] for item in history['metrics']] == [0, 1, 2, 3]
assert all(item['rows'] == 9508 and item['primary_score'] == 'last/bbs'
           for item in history['metrics'])
actual_face_history = [item['hits050'] for item in history['metrics']]
assert actual_face_history == [4920, 4599, 4625, 4476]
checks = []
for values, expected in (([4777], 0), ([4777, 4750], 1), ([4777, 4750, 4749], 2),
                         ([4777, 4750, 4750], 0), ([4777, 4750, 4751], 0),
                         (actual_face_history, 1)):
    observed = declines([dict(hits050=value) for value in values])
    assert observed == expected
    checks.append(dict(hits050=values, consecutive_declines=observed))
qualified = dict(hits025=5705, hits050=4945)
unqualified_strict_higher = dict(hits025=5704, hits050=5000)
assert order(qualified) > order(unqualified_strict_higher)
assert order(dict(hits025=5705, hits050=4946)) > order(qualified)
assert order(dict(hits025=5706, hits050=4945)) > order(qualified)
assert order(dict(hits025=5706, hits050=4945)) == order(dict(hits025=5706, hits050=4945))
result = dict(status='COMMON_FRESH_NORMAL_POLICY_CPU_PASS', checks=checks,
    actual_epoch3_analysis_source=str(history_path), actual_epoch3_analysis_sha256=hashlib.sha256(history_path.read_bytes()).hexdigest(),
    policy_source_sha256=hashlib.sha256(raw).hexdigest(), synthetic_cases_are_engineering_not_accuracy=True,
    actual_face_metrics_for_decline_example_only=True,
    complete_model_forward=False, actual_training_loop_executed=False,
    optimizer_resume_verified=False, max_epochs=8, patience=2, no_neural_forwards=True,
    current_MCLN_training_changed=False, normal_training_launched=False)
(root / 'COMMON_LOOP_POLICY_CHECK_V2.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status=result['status'], checks=len(checks), new_neural_forwards=0)))
