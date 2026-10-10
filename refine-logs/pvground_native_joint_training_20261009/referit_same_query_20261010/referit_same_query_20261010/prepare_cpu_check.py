"""Prepare a bounded CPU evaluator check using the existing reviewed warm runtime."""
import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parent
prior = root.parent / 'referit_data_interface_cpu_20261010'
old_spec = json.loads((prior / 'CHECK_SPEC.json').read_bytes())
(root / 'BASE_WARM_SOURCE_HASHES.json').write_bytes((prior / 'WARM_SOURCE_HASHES.json').read_bytes())
spec = dict(warm_source=old_spec['warm_source'], env_spec_sha256=old_spec['env_spec_sha256'],
    CPU_only=True, CUDA_VISIBLE_DEVICES='', fixture_cases=5,
    fixture_points=4, fixture_queries=256, fixture_tokens=256,
    dataset_constructors=0, PV_model_constructors=0, criterion_calls=0,
    optimizer_steps=0, saved_weights=0, current_training_queries=0,
    formal_accuracy=None, full_benchmark_evaluation_completed=False,
    GPU_training_admission=False)
(root / 'CHECK_SPEC.json').write_text(json.dumps(spec, indent=2)+'\n')
for path in root.glob('*.py'):
    tree = ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 7))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and
                target.id == 'remote_code' for target in node.targets):
            ast.parse(ast.literal_eval(node.value), feature_version=(3, 7))
print(json.dumps(dict(status='NATIVE_SAME_QUERY_EVALUATOR_CPU_SOURCE_PREPARED',
    fixture_cases=5, formal_accuracy=None, GPU_training_admission=False)))
