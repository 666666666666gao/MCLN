"""Reuse immutable completed runner; change only controlled boundary package."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

local = Path(__file__).resolve().parent
prior = local.parent / 'pvground_range_head_only_20261004'
assert not (local / 'source_preparation.json').exists()
source = prior / 'run_range_head_only.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == 'e4fdf9bddb90e1709f439a80e55cbe86136f5cd059fb76190a104df28d748b95'
text = source.read_text(encoding='utf-8')

def replace(before, after):
    global text
    assert text.count(before) == 1, before
    text = text.replace(before, after)

replace('"""Keep original G state fixed; learn only its existing range refiner."""',
        '"""Keep original G fixed; compare residual and supervised six-face distribution."""')
replace("    assert spec['support_arm'] in ('local_range', 'whole_range')\n    assert spec['use_whole_range'] == (spec['support_arm'] == 'whole_range')\n",
        "    assert spec['support_arm'] == 'whole_range' and spec['use_whole_range']\n"
        "    assert spec['boundary_mode'] in ('residual', 'distribution')\n"
        "    assert spec['boundary_loss_weight'] == (0.0 if spec['boundary_mode'] == 'residual' else 1.0 / 7)\n"
        "    from pvground_boundary_box_refiner import install_boundary_refinement, distribution_loss\n")
replace("    from pvground_whole_mask_box_refiner import install_whole_mask_refinement\n    install_whole_mask_refinement(model, spec['use_whole_range'])\n    assert sum(p.numel() for p in model.candidate_box_refiner.parameters()) == 400614\n",
        "    install_boundary_refinement(model, spec['boundary_mode'])\n"
        "    expected_head_parameters = 400614 if spec['boundary_mode'] == 'residual' else 456102\n"
        "    assert spec['head_parameters'] == expected_head_parameters\n"
        "    assert sum(p.numel() for p in model.candidate_box_refiner.parameters()) == expected_head_parameters\n")
assert text.count('head_parameters=400614') == 4
text = text.replace('head_parameters=400614', "head_parameters=expected_head_parameters, boundary_mode=spec['boundary_mode'], boundary_loss_weight=spec['boundary_loss_weight']")
replace("        assert terminal['head_only']\n", "        assert terminal['head_only']\n"
        "        assert terminal['boundary_mode'] == spec['boundary_mode']\n"
        "        assert terminal['boundary_loss_weight'] == spec['boundary_loss_weight']\n")
replace("        loss=native+correction\n", "        edge = native.new_zeros(())\n"
        "        boundary_counts = {}\n"
        "        if spec['boundary_mode'] == 'distribution':\n"
        "            edge, boundary_counts = distribution_loss(predictions, batch, matching[1])\n"
        "            assert torch.isfinite(edge)\n"
        "        loss=native+correction+spec['boundary_loss_weight']*edge\n"
        "        assignment_counts.update(boundary_counts, boundary_loss=float(edge),\n"
        "            boundary_weighted_loss=float(spec['boundary_loss_weight']*edge),\n"
        "            size_floor_axis_count=int(predictions['boundary_size_floor_count']))\n")
replace("            assert geometry_output_gradient > 0\n", "            assert geometry_output_gradient > 0\n"
        "            if spec['boundary_mode'] == 'distribution':\n"
        "                edge_gradients = torch.autograd.grad(edge, parameters, retain_graph=True, allow_unused=True)\n"
        "                edge_output_gradient = float(edge_gradients[-2].norm())\n"
        "                assert edge_output_gradient > 0\n"
        "                assignment_counts['edge_alone_output_gradient'] = edge_output_gradient\n"
        "            assert (predictions['last_pred_size'] > 0).all()\n")
start = text.index('            pair_outputs = []\n')
end = text.index('            preflight_call_witnesses.append(call_witness)', start)
text = text[:start] + (
    "            replay = read(*captured[0])\n"
    "            torch.cuda.synchronize(); pair_replay_seconds = time.perf_counter() - replay_begin\n"
    "            assert torch.equal(replay[0], captured[0][2])\n"
    "            assert torch.equal(replay[1], captured[0][3].clamp(min=1e-6))\n"
    "            if spec['boundary_mode'] == 'distribution':\n"
    "                from pvground_boundary_box_refiner import expected_offset\n"
    "                assert (expected_offset(captured[0][4]['boundary_logits']) == 0).all()\n"
    "            del captured, replay\n"
) + text[end:]
replace("            differences[key] = float((with_p3[key] - without_p3[key]).abs().max())\n            assert torch.equal(with_p3[key], without_p3[key]), key\n",
        "            reference = without_p3[key].clamp(min=1e-6) if key == 'last_pred_size' else without_p3[key]\n"
        "            differences[key] = float((with_p3[key] - reference).abs().max())\n"
        "            assert torch.equal(with_p3[key], reference), key\n")
replace("        support_statistics['negative_coarse_size_elements'] = int((with_p3['p3_coarse_size'] < 0).sum())\n",
        "        support_statistics['negative_coarse_size_elements'] = int((with_p3['p3_coarse_size'] < 0).sum())\n"
        "        support_statistics['raw_coarse_size_elements_below_floor'] = int((with_p3['p3_coarse_size'] < 1e-6).sum())\n")
replace('same_cached_inputs_zero_head_pair_exact=True,', 'same_cached_inputs_zero_head_common_floor_exact=True,')
replace("                    assert torch.equal(predictions['last_pred_size'],predictions['p3_coarse_size'])\n",
        "                    assert torch.equal(predictions['last_pred_size'],predictions['p3_coarse_size'].clamp(min=1e-6))\n")
replace("                    rows.append(record);stream.write(json.dumps(record)+'\\n')\n",
        "                        record[mode]['size_floor_axes'] = int(predictions['boundary_size_floored'][bid,q].sum())\n"
        "                        if spec['boundary_mode'] == 'distribution':\n"
        "                            from pvground_boundary_box_refiner import expected_offset\n"
        "                            logits = predictions['boundary_logits'][bid,q]\n"
        "                            probability = logits.softmax(-1)\n"
        "                            record[mode]['face_offsets'] = expected_offset(logits).cpu().tolist()\n"
        "                            record[mode]['face_entropy'] = -(probability * logits.log_softmax(-1)).sum(-1).cpu().tolist()\n"
        "                    rows.append(record);stream.write(json.dumps(record)+'\\n')\n")
# Parenthesize the unary tensor operation before converting to a Python list.
text = text.replace("= -(probability * logits.log_softmax(-1)).sum(-1).cpu().tolist()", "= (-(probability * logits.log_softmax(-1)).sum(-1)).cpu().tolist()")
replace("    if spec['use_whole_range']:\n", "    if spec['boundary_mode'] == 'distribution':\n")
ast.parse(text, feature_version=(3, 7))
(local / 'run_boundary_fit.py').write_text(text, encoding='utf-8', newline='\n')
modules = ['whole_model_preflight_checks.py', 'pvground_whole_mask_box_refiner.py', 'whole_mask_range.py',
    'pvground_candidate_box_refiner.py', 'pvground_tail_support_box_refiner.py', 'pvground_tail_preflight.py',
    'pvground_semantic_assignment.py', 'pvground_source_query.py', 'pvground_observation_query.py',
    'pvground_task_observation_query.py', 'initial_range_comparison.py']
for name in modules:
    shutil.copyfile(prior / name, local / name)
    assert (prior / name).read_bytes() == (local / name).read_bytes()
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
base_spec = json.loads((prior / 'whole_range_spec.json').read_bytes())
for arm in ('residual', 'distribution'):
    spec = dict(base_spec)
    spec.update(boundary_mode=arm, head_parameters=400614 if arm == 'residual' else 456102,
        boundary_loss_weight=0.0 if arm == 'residual' else 1.0 / 7,
        root='/root/autodl-tmp/pvground_boundary_preflight_20261004/' + arm,
        paired_control_root='/root/autodl-tmp/pvground_boundary_fit_20261004/residual',
        preflight_root='/root/autodl-tmp/pvground_boundary_preflight_20261004/' + arm,
        preflight_only_at_install=True)
    spec['whole_range_files'] = dict(base_spec['whole_range_files'])
    spec['whole_range_files']['pvground_boundary_box_refiner.py'] = sha(local / 'pvground_boundary_box_refiner.py')
    (local / (arm + '_preflight_spec.json')).write_text(json.dumps(spec, indent=2) + '\n', encoding='utf-8')
receipt = dict(status='LOCAL_SOURCE_PREPARED_NOT_EXECUTED',
    inherited_runner_sha256=sha(source), source_files={p.name: sha(p) for p in sorted(local.glob('*.py'))},
    unchanged_module_count=len(modules), native_forwards=0, optimizer_updates=0, weights_created=0,
    source_review_complete=False, real_preflight_complete=False, accuracy_result_available=False)
(local / 'source_preparation.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps(receipt))
