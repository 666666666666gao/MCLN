"""Derive the isolated runner from the closed native-target continuation."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
previous = root.parent / 'pvground_auxiliary_target_20261005'
assert not (root / 'run_geometry_fit.py').exists()
published = json.loads((previous / 'terminal_publication.json').read_bytes())
assert published['retained_best']['hits'] == [5616, 4511]


def change(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


runner = (previous / 'run_geometry_fit.py').read_text(encoding='utf-8')
runner = change(runner, 'Protected4509 continuation; native versus member auxiliary geometry targets.',
    'Protected4511 continuation; fixed versus own/fused-support reference.')
runner = change(runner, "assert spec['geometry_hits50']==4509 and spec['head_only']",
    "assert spec['geometry_hits50']==4511 and spec['head_only']")
runner = change(runner, "assert spec['auxiliary_target_mode'] in ('native_gt','member_gt')",
    "assert spec['auxiliary_target_mode']=='native_gt'\n"
    "    assert isinstance(spec['reference_enabled'],bool)\n"
    "    assert spec['reference_loss_weight']==(1.0 if spec['reference_enabled'] else 0.0)\n"
    "    assert sha(output.parent/'support_reference.py')==spec['support_reference_sha256']")
runner = change(runner, 'from pvground_boundary_box_refiner import distribution_loss',
    'from pvground_boundary_box_refiner import distribution_loss, BoundaryBoxRefiner\n'
    '    from support_reference import install_support_reference, reference_localization_loss')
runner = change(runner, '    model.cuda()\n    for parameter in model.parameters():',
    "    if spec['reference_enabled']:\n"
    "        install_support_reference(model)\n"
    "    model.cuda()\n    for parameter in model.parameters():")
runner = change(runner, "assert len(trainable)==10 and all(name.startswith('candidate_box_refiner.') for name in trainable)",
    "expected_states=12 if spec['reference_enabled'] else 10\n"
    "    expected_parameters=459180 if spec['reference_enabled'] else 456102\n"
    "    assert len(trainable)==expected_states and all(name.startswith('candidate_box_refiner.') for name in trainable)")
runner = change(runner, 'assert sum(parameter.numel() for parameter in trainable.values())==456102',
    'assert sum(parameter.numel() for parameter in trainable.values())==expected_parameters')
runner = change(runner, '    GroundingEvaluator=evaluator_module.GroundingEvaluator',
    "    GroundingEvaluator=evaluator_module.GroundingEvaluator\n"
    "    assert sha(evaluator_path)==spec['native_evaluator_sha256']\n"
    "    imported['native_evaluator']=str(evaluator_path)\n"
    "    write_json(output/'imports.json',dict(files=imported,sha256={key:sha(value) for key,value in imported.items()}))")
runner = change(runner, "        assert terminal['auxiliary_target_mode']==spec['auxiliary_target_mode']",
    "        assert terminal['auxiliary_target_mode']==spec['auxiliary_target_mode']\n"
    "        assert terminal['reference_enabled']==spec['reference_enabled']\n"
    "        assert terminal['reference_loss_weight']==spec['reference_loss_weight']")
runner = change(runner,
    "                coarse = torch.cat([predictions['p3_coarse_center'],\n"
    "                                    predictions['p3_coarse_size'].clamp(min=1e-6)], -1)",
    "                reference_boxes=torch.cat([predictions['p3_coarse_center'],predictions['p3_coarse_size'].clamp(min=1e-6)],-1)\n"
    "                if spec['reference_enabled']:\n"
    "                    coarse=torch.cat([predictions['native_coarse_center'],predictions['native_coarse_size'].clamp(min=1e-6)],-1)\n"
    "                else:\n"
    "                    coarse=reference_boxes")
runner = change(runner, '                    coarse_iou = box_iou(coarse[bid], truth[bid])',
    '                    coarse_iou = box_iou(coarse[bid], truth[bid])\n'
    '                    reference_iou=box_iou(reference_boxes[bid],truth[bid])')
runner = change(runner, '                            coarse_box=coarse[bid, query].cpu().tolist(), coarse_iou=float(coarse_iou[query]),',
    '                            coarse_box=coarse[bid, query].cpu().tolist(), coarse_iou=float(coarse_iou[query]),\n'
    '                            reference_box=reference_boxes[bid,query].cpu().tolist(),reference_iou=float(reference_iou[query]),')
runner = change(runner, '        auxiliary_roots=native_roots if spec[\'auxiliary_target_mode\']==\'native_gt\' else batch[\'pre_jitter_root_box\']',
    '        auxiliary_roots=native_roots')
runner = change(runner, "        loss=native+correction+(1.0/7)*edge+spec['extra_geometry_weight']*extra",
    "        reference_loss=predictions['last_center'].sum()*0\n"
    "        if spec['reference_enabled']:\n"
    "            reference_loss=reference_localization_loss(predictions,batch,matches[1],qualified,set_criterion,auxiliary_roots)\n"
    "        loss=native+correction+(1.0/7)*edge+spec['extra_geometry_weight']*extra+spec['reference_loss_weight']*reference_loss")
runner = change(runner, "        witness={}\n        if args.mode=='preflight':",
    "        witness={}\n        if args.mode=='preflight':\n"
    "            if spec['reference_enabled']:\n"
    "                if torch.count_nonzero(geometry_head.reference.weight)==0 and torch.count_nonzero(geometry_head.reference.bias)==0:\n"
    "                    with torch.no_grad():\n"
    "                        arguments=list(head_inputs[0]);arguments[-1]=dict(arguments[-1])\n"
    "                        old_center,old_size=BoundaryBoxRefiner.forward(geometry_head,*arguments)\n"
    "                    assert torch.equal(old_center,predictions['last_center']) and torch.equal(old_size,predictions['last_pred_size'])\n"
    "                    witness['zero_reference_cached_old_head_exact']=True\n"
    "                ref_direct=torch.autograd.grad(reference_loss,(predictions['support_reference_center'],predictions['support_reference_size']),retain_graph=True)\n"
    "                assert all(torch.isfinite(value).all() and value.norm()>0 for value in ref_direct)\n"
    "                ref_parameter=torch.autograd.grad(reference_loss,geometry_head.reference.weight,retain_graph=True)[0]\n"
    "                assert torch.isfinite(ref_parameter).all() and ref_parameter.norm()>0\n"
    "                witness['reference_direct_geometry_gradient']=float(ref_parameter.norm())")
runner = change(runner, '            witness=dict(extra_alone_output_gradient=float(isolated[-2].norm()),\n'
    '                extra_direct_output_gradients_only_qualified=True)',
    "            witness.update(extra_alone_output_gradient=float(isolated[list(trainable).index('candidate_box_refiner.output.weight')].norm()),\n"
    '                extra_direct_output_gradients_only_qualified=True)')
runner = change(runner, "            matched_boundary_loss=float(edge),extra_geometry_loss=float(extra),",
    "            matched_boundary_loss=float(edge),extra_geometry_loss=float(extra),\n"
    "            reference_localization_loss=float(reference_loss),reference_enabled=spec['reference_enabled'],\n"
    "            reference_loss_weight=spec['reference_loss_weight'],")
runner = runner.replace('head_parameters=456102,head_state_tensors=10',
    'head_parameters=expected_parameters,head_state_tensors=expected_states')
runner = change(runner, "            geometry_parent_fit_updates=7446,total_geometry_fit_updates=7446+step_number,",
    "            geometry_parent_fit_updates=11169,total_geometry_fit_updates=11169+step_number,\n"
    "            reference_enabled=spec['reference_enabled'],reference_loss_weight=spec['reference_loss_weight'],\n"
    "            support_reference_sha256=spec['support_reference_sha256'],")
runner = change(runner, "            isolated_extra_geometry_gradient_verified=True,extra_geometry_weight=spec['extra_geometry_weight'],",
    "            isolated_extra_geometry_gradient_verified=True,extra_geometry_weight=spec['extra_geometry_weight'],\n"
    "            reference_enabled=spec['reference_enabled'],reference_loss_weight=spec['reference_loss_weight'],")
(root / 'run_geometry_fit.py').write_text(runner, encoding='utf-8')
(root / 'query_supported_geometry.py').write_bytes((previous / 'query_supported_geometry.py').read_bytes())
controller = (previous / 'controller.py').read_text(encoding='utf-8')
controller = controller.replace("('control','member_target')", "('control','support_reference')")
controller = change(controller, 'protected_best_hits=[5614,4509]', 'protected_best_hits=[5616,4511]')
(root / 'controller.py').write_text(controller, encoding='utf-8')
parent = json.loads((previous / 'control_spec.json').read_bytes())
evaluator = root.parent / 'pvground_auxiliary_target_20261005/analysis/EXPERIMENT_AUDIT.json'
audit = json.loads(evaluator.read_bytes())
assert audit['verdict'] in ('PASS','WARN') and not audit['blocking_findings']
# Actual inspected runtime-source copy used in the closed review.
source_evaluator = root.parent / 'pvground_runtime_bundle_20260908_v1/PV-Ground/src/grounding_evaluator.py'
assert source_evaluator.is_file()
for arm, enabled in (('control', False), ('support_reference', True)):
    spec = dict(parent)
    spec.update(root='/root/autodl-tmp/pvground_support_reference_20261005/' + arm,
        geometry_terminal=published['retained_best']['path'],geometry_terminal_sha256=published['retained_best']['sha256'],
        geometry_hits50=4511,geometry_parent_fit_updates=11169,total_geometry_fit_updates=14892,
        auxiliary_target_mode='native_gt',reference_enabled=enabled,reference_loss_weight=float(enabled),
        support_reference_sha256=hashlib.sha256((root/'support_reference.py').read_bytes()).hexdigest(),
        native_evaluator_sha256=hashlib.sha256(source_evaluator.read_bytes()).hexdigest())
    (root/(arm+'_spec.json')).write_text(json.dumps(spec,indent=2)+'\n',encoding='utf-8')
for file in root.glob('*.py'):
    ast.parse(file.read_text(encoding='utf-8'))
print(json.dumps({'status':'PREPARED_NOT_LAUNCHED','parent_hits':[5616,4511],
    'new_reference_parameters':3078,'strategy_parameters':459180,'control_parameters':456102}))
