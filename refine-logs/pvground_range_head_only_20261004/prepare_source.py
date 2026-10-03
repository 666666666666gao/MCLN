"""Minimal stable-G protocol change; reuse the completed source experiment."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

local = Path(__file__).parent
prior = local.parent/'pvground_whole_mask_fit_20261003'
assert not (local/'source_preparation.json').exists()
source = prior/'run_whole_mask_fit.py'
text = source.read_text(encoding='utf-8').replace('\r\n','\n')
assert hashlib.sha256(source.read_bytes()).hexdigest() == 'd862a861d05f0411ed76c18b78684b485fe1fdfaa8a0c00d3250ce9a9e8cfeac'

def replace(before, after):
    global text
    assert text.count(before)==1, before
    text=text.replace(before, after)

replace('"""Same original G: local fused support with whole-Mask range information on/off."""',
        '"""Keep original G state fixed; learn only its existing range refiner."""')
replace("    model.cuda()\n", "    assert spec['head_only']\n"
    "    for name, parameter in model.named_parameters():\n"
    "        parameter.requires_grad_(name.startswith('candidate_box_refiner.'))\n"
    "    core_names = set(initial) - p3_names\n"
    "    model.cuda()\n")
replace("    assert len(trainable)==interface['trainable_tensors'] + len(p3_names)\n",
    "    assert set(trainable) == p3_names and len(trainable) == 10\n")
replace("        selected = set(trainable) | {n for n, _ in model.named_buffers() if n in initial}\n",
    "        assert terminal['head_only']\n"
    "        selected = set(trainable)\n")
start=text.index('            semantic_gradients = torch.autograd.grad(')
end=text.index("            assignment_counts['full_model_forward_seconds']",start)
text=text[:start]+(
    "            assert not (predictions['loss_ce']+predictions['loss_sem_align']+correction).requires_grad\n"
    "            mask_outputs = tuple(predictions['last_pred_masks'] + predictions['sp_last_pred_masks'] + predictions['adaptive_weights'])\n"
    "            assert all(not value.requires_grad for value in mask_outputs)\n"
    "            assert not predictions['whole_mask_range_evidence'].requires_grad\n"
    "            assignment_counts['direct_geometry_output_gradient'] = geometry_output_gradient\n"
    "            assignment_counts['semantic_loss_requires_grad'] = False\n"
    "            assignment_counts['mask_outputs_require_grad'] = False\n"
    "            assignment_counts['range_observation_requires_grad'] = False\n"
) + text[end:]
replace("        if update:optimizer.step()\n",
    "        assert all(parameter.grad is None for parameter in frozen.values())\n"
    "        if update:optimizer.step()\n")
replace("        reset_rng(spec['seed']);model.train()\n",
    "        reset_rng(spec['seed']);model.eval();model.candidate_box_refiner.train()\n")
start=text.index("                assert record['geometry_to_mask_output_gradient'] > 0\n")
end=text.index("            record['p3_gradients']=gradients",start)
text=text[:start]+text[end:]
replace("        selected=set(trainable)|{n for n,_ in model.named_buffers() if n in initial}\n",
    "        assert all(torch.equal(model.state_dict()[name].detach().cpu(), initial[name]) for name in core_names)\n"
    "        assert not model.training\n"
    "        selected=set(trainable)\n")
replace("            p3=True,p2=False,direct_semantic_to_p3_gradients_zero=True,\n",
    "            p3=True,p2=False,head_only=True,original_g_state_unchanged=True,\n")
replace("            direct_native_mask_loss_to_refiner_gradients_zero=True,\n",
    "            upstream_parameters_frozen=True,upstream_running_state_eval=True,\n")
replace("    reset_rng(spec['seed']);dataset.augment=True;dataset.augment_det=True;model.train()\n",
    "    reset_rng(spec['seed']);dataset.augment=True;dataset.augment_det=True\n"
    "    model.eval();model.candidate_box_refiner.train()\n")
replace("    selected_names=set(trainable)|{n for n,_ in model.named_buffers() if n in initial}\n",
    "    selected_names=set(trainable)\n")
replace("                    whole_range_model=True, head_parameters=400614)\n",
    "                    whole_range_model=True, head_parameters=400614, head_only=True)\n")
replace("    assert all(torch.equal(p.detach().cpu(),initial[n]) for n,p in frozen.items())\n",
    "    assert all(torch.equal(model.state_dict()[name].detach().cpu(),initial[name]) for name in core_names)\n")
replace("                   whole_range_model=True, head_parameters=400614)\n",
    "                   whole_range_model=True, head_parameters=400614, head_only=True,\n"
    "                   original_g_state_unchanged=True, upstream_running_state_eval=True)\n")
replace('    from pvground_tail_preflight import observed_forward, native_mask_loss_routes\n',
        '    from pvground_tail_preflight import observed_forward\n')
replace('    from whole_model_preflight_checks import whole_range_loss_routes, optimizer_restore_exact\n',
        '    from whole_model_preflight_checks import optimizer_restore_exact\n')
ast.parse(text, feature_version=(3,7))
(local/'run_range_head_only.py').write_text(text,encoding='utf-8',newline='\n')
modules=['whole_model_preflight_checks.py','pvground_whole_mask_box_refiner.py','whole_mask_range.py',
    'pvground_candidate_box_refiner.py','pvground_tail_support_box_refiner.py','pvground_tail_preflight.py',
    'pvground_semantic_assignment.py','pvground_source_query.py','pvground_observation_query.py',
    'pvground_task_observation_query.py','initial_range_comparison.py']
for name in modules:
    shutil.copyfile(prior/name,local/name)
spec=json.loads((prior/'whole_range_spec.json').read_bytes())
spec.update(root='/root/autodl-tmp/pvground_range_head_only_preflight_20261004/whole_range',
    head_only=True,comparison='same original G fixed / only range head learns',
    preflight_only_at_install=True,
    paired_control_root='/root/autodl-tmp/pvground_range_head_only_fit_20261004/local_range')
(local/'preflight_spec.json').write_text(json.dumps(spec,indent=2)+'\n',encoding='utf-8')
record=dict(status='PREPARED_NOT_LAUNCHED',original_runner=str(source),
    original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    current_sha256=hashlib.sha256((local/'run_range_head_only.py').read_bytes()).hexdigest(),
    unchanged_model_modules=modules,model_updates=0,weights_created=0,
    changes=['only refiner parameters trainable','original modules eval during head training',
        'verify all original state unchanged','head-only delta/optimizer restoration',
        'frozen Mask and semantic outputs recorded rather than differentiated'])
(local/'source_preparation.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record))
