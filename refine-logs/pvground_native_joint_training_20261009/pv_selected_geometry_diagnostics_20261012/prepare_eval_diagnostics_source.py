"""Add compact diagnostics to the future isolated native evaluation loop."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
stage = json.loads((root / 'SOURCE_STAGE_R3_RECEIPT.json').read_bytes())
original = root.parent / 'source/train_dist_mod.py'
raw = original.read_bytes()
assert hashlib.sha256(raw).hexdigest() == stage['source_files']['train_dist_mod.py']['sha256']
source = raw.decode('utf-8').replace('\r\n', '\n')
anchor = 'from src.grounding_evaluator import GroundingEvaluator\n'
assert source.count(anchor) == 1
source = source.replace(anchor, anchor + 'from selected_geometry_eval_diagnostics import SelectedGeometryEvalDiagnostics\n')
anchor = '        # NOTE Main eval branch\n        test_loader = tqdm(test_loader, ascii=True)'
assert source.count(anchor) == 1
source = source.replace(anchor, '        diagnostics = SelectedGeometryEvalDiagnostics(args.log_dir, epoch)\n' + anchor)
anchor = '                    evaluator.evaluate(end_points, prefix)'
assert source.count(anchor) == 1
source = source.replace(anchor, anchor + '\n                    diagnostics.append(end_points)')
anchor = "        return dict(rows=9508, hits025=int(evaluator.dets[keys[0]]),\n                    hits050=int(evaluator.dets[keys[1]]), primary_score='last/bbs',\n                    same_complete_model=True)"
assert source.count(anchor) == 1
source = source.replace(anchor, anchor.replace('return dict', 'metrics = dict')
    + '\n        diagnostics.finish(metrics)\n        return metrics')
ast.parse(source)
directory = root / 'source_overlay_v4'
directory.mkdir(exist_ok=True)
(directory / 'train_dist_mod.py').write_bytes(source.encode())
helper = (root / 'selected_geometry_eval_diagnostics.py').read_bytes()
ast.parse(helper.decode())
(directory / 'selected_geometry_eval_diagnostics.py').write_bytes(helper)
for name in ('launch_gpu_preflight_pair_authorized.py', 'launch_normal_arm_authorized.py'):
    source = (root / name).read_text(encoding='utf-8')
    assert source.count('SOURCE_STAGE_R3_RECEIPT.json') == 1
    assert source.count('source_review/SOURCE_REVIEW_LAUNCHERS.json') == 1
    source = source.replace('SOURCE_STAGE_R3_RECEIPT.json', 'SOURCE_STAGE_R4_RECEIPT.json')
    source = source.replace('source_review/SOURCE_REVIEW_LAUNCHERS.json', 'source_review/SOURCE_REVIEW_DIAGNOSTICS.json')
    ast.parse(source)
    (root / (Path(name).stem + '_v4.py')).write_bytes(source.encode())
record = dict(status='COMPACT_NATIVE_EVAL_DIAGNOSTICS_PREPARED_NOT_EXECUTED',
    original_train_source_sha256=hashlib.sha256(raw).hexdigest(),
    modified_train_source_sha256=hashlib.sha256((directory / 'train_dist_mod.py').read_bytes()).hexdigest(),
    new_diagnostic_source_sha256=hashlib.sha256(helper).hexdigest(),
    neural_calls=0, GPU_calls=0, active_MCLN_source_changed=False,
    deployed=False, actual_formal_accuracy=None, new_model_parameters=0,
    raw_candidate_arrays_saved=False, source_files_after_future_stage=118)
(root / 'EVAL_DIAGNOSTIC_SOURCE_PREPARATION.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
print(json.dumps(record))
