"""Prepare bounded CPU checks; no active model change or new GPU task."""
import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parent
data = root / 'query_mask_assignment_20261011'
source_review = json.loads((data / 'source_review/SOURCE_REVIEW.json').read_bytes())
assert source_review['execution_scope'] == 'SOURCE_ONLY' and not source_review['blocking_findings']
original = json.loads((root / 'matcher_mask_role_cpu_20261011/CHECK_SPEC.json').read_bytes())
hashes = json.loads((data / 'SOURCE_HASHES.json').read_bytes())
spec = dict(execution_scope='ISOLATED_NATIVE_MATCHER_AND_BOXES_ONLY_CRITERION_CPU',
    warm_source=original['warm_source'], env_spec_sha256=original['env_spec_sha256'],
    original_losses_sha256=original['source_sha256']['models/losses.py'],
    query_losses_sha256=hashes['models/losses.py'], import_binding_sha256=original['import_binding_sha256'],
    constructed_cases=['root_only', 'root_anchor', 'mixed_gt_batch'],
    query_count=256, token_count=256, superpoint_count=4, input_point_count=6,
    seed=2027, threads=1, CUDA_VISIBLE_DEVICES='',
    expected_matcher_calls=33, expected_boxes_only_criterion_calls=6,
    check_scope=['Default and text equal unchanged warm matcher', 'Query Mask mutations alter direct cost and assignment',
        'All GT retain unique one-to-one assignments', 'No-Mask intermediate path unchanged',
        'Native SetCriterion boxes-only loss routes direct box gradients to newly assigned queries'],
    excluded=['PV model forward', 'real dataset', 'full native loss', 'Mask loss', 'optimizer', 'GPU',
        'benchmark accuracy', 'current main training query', 'active source edit', 'new normal training'],
    estimated_cpu_seconds=10, estimated_transport_seconds=60,
    next_main_training_observation_cst='2026-10-11T05:00:40.968644+08:00')
path = data / 'CPU_CHECK_SPEC.json'
assert not path.exists()
path.write_text(json.dumps(spec, indent=2) + '\n')
for path in (data / 'check_query_mask_assignment_cpu.py', data / 'run_cpu_authorized.py'):
    ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 7))
print(json.dumps(dict(status='QUERY_MASK_ASSIGNMENT_CPU_CHECK_PREPARED', execution_started=False)))
