"""Prepare the minimal two-reference continuation without changing warm runtime."""
import ast
import copy
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
previous=root.parent/'pvground_support_reference_20261005'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def replace_once(text,old,new):
    assert text.count(old)==1,old[:120]
    return text.replace(old,new,1)

text=(previous/'run_geometry_fit.py').read_text(encoding='utf-8')
text=replace_once(text,'"""Protected4511 continuation; fixed versus own/fused-support reference."""',
    '"""Protected4511 hidden states; native versus full fused-Mask reference."""')
text=replace_once(text,"choices=['preflight', 'train', 'formal']","choices=['preflight', 'train', 'initial_formal', 'formal']")
text=replace_once(text,"    assert isinstance(spec['reference_enabled'],bool)\n    assert spec['reference_loss_weight']==(1.0 if spec['reference_enabled'] else 0.0)\n    assert sha(output.parent/'support_reference.py')==spec['support_reference_sha256']",
    "    assert spec['reference_mode'] in ('native','fused_mask')\n    assert spec['common_output_reset']==['output.weight','output.bias']\n    assert sha(output.parent/'mask_reference.py')==spec['mask_reference_sha256']")
text=replace_once(text,'    from support_reference import install_support_reference, reference_localization_loss',
    '    from mask_reference import install_mask_reference, reference_bounds_witness')
text=replace_once(text,'    from query_supported_geometry import query_supported_geometry_loss',
    '    from query_supported_geometry import query_supported_geometry_loss\n    from check_invalid_reference import check_invalid_reference')
text=replace_once(text,"    if spec['reference_enabled']:\n        install_support_reference(model)",
    "    install_mask_reference(model,spec['reference_mode'])\n    load.update(reference_mode=spec['reference_mode'],common_output_reset=spec['common_output_reset'],\n        retained_hidden_state_tensors=8,reset_output_state_tensors=2)")
text=replace_once(text,"    expected_states=12 if spec['reference_enabled'] else 10\n    expected_parameters=459180 if spec['reference_enabled'] else 456102",
    '    expected_states=10\n    expected_parameters=456102')
text=replace_once(text,"    formal = args.mode == 'formal'", "    formal = args.mode in ('initial_formal','formal')")
text=replace_once(text,"    if formal:\n        terminal=torch.load", "    if args.mode=='formal':\n        terminal=torch.load")
text=replace_once(text,"        assert terminal['reference_enabled']==spec['reference_enabled']\n        assert terminal['reference_loss_weight']==spec['reference_loss_weight']",
    "        assert terminal['reference_mode']==spec['reference_mode']\n        assert terminal['common_output_reset']==spec['common_output_reset']\n        assert terminal['mask_reference_sha256']==spec['mask_reference_sha256']")
text=replace_once(text,"                if spec['reference_enabled']:\n                    coarse=torch.cat([predictions['native_coarse_center'],predictions['native_coarse_size'].clamp(min=1e-6)],-1)\n                else:\n                    coarse=reference_boxes",
    "                coarse=torch.cat([predictions['native_coarse_center'],predictions['native_coarse_size'].clamp(min=1e-6)],-1)\n                if formal:\n                    np.savez_compressed(directory/('batch_%05d.npz'%len(rows)),\n                        row_ids=batch['local_training_id'].cpu().numpy(),\n                        original_prior=coarse.cpu().numpy(),reference=reference_boxes.cpu().numpy(),\n                        final=boxes.cpu().numpy(),scores=score.cpu().numpy(),\n                        reference_valid=predictions['mask_reference_valid'].cpu().numpy(),\n                        root_gt=torch.cat([batch['center_label'][:,0,:3],batch['size_gts'][:,0]],-1).cpu().numpy())")
text=replace_once(text,"                            reference_box=reference_boxes[bid,query].cpu().tolist(),reference_iou=float(reference_iou[query]),",
    "                            reference_box=reference_boxes[bid,query].cpu().tolist(),reference_iou=float(reference_iou[query]),\n                            reference_valid=bool(predictions['mask_reference_valid'][bid,query]),")
text=replace_once(text,"        reference_loss=predictions['last_center'].sum()*0\n        if spec['reference_enabled']:\n            reference_loss=reference_localization_loss(predictions,batch,matches[1],qualified,set_criterion,auxiliary_roots)\n        loss=native+correction+(1.0/7)*edge+spec['extra_geometry_weight']*extra+spec['reference_loss_weight']*reference_loss",
    "        loss=native+correction+(1.0/7)*edge+spec['extra_geometry_weight']*extra")
start=text.index("            if spec['reference_enabled']:",text.index('        witness={}'))
end=text.index('            output_gradients=torch.autograd.grad(extra,',start)
text=text[:start]+"""            witness.update(reference_bounds_witness(predictions,batch))
            if torch.count_nonzero(geometry_head.output.weight)==0 and torch.count_nonzero(geometry_head.output.bias)==0:
                assert torch.equal(predictions['last_center'],predictions['geometry_reference_center'])
                assert torch.equal(predictions['last_pred_size'],predictions['geometry_reference_size'].clamp(min=1e-6))
                witness['neutral_initial_decode_equals_reference']=True
            # Reference changes the actual eligibility set. Keep the existing
            # empty-set loss and inspect its real output gradients without
            # imposing the old frame's nonempty/clipped-target fixture counts.
"""+text[end:]
text=replace_once(text,"            output_gradients=torch.autograd.grad(extra,\n                (predictions['last_center'],predictions['last_pred_size'],predictions['boundary_logits']),retain_graph=True)",
    "            output_gradients=torch.autograd.grad(extra+predictions['boundary_logits'].sum()*0,\n                (predictions['last_center'],predictions['last_pred_size'],predictions['boundary_logits']),retain_graph=True)")
text=replace_once(text,"            assert witness['extra_alone_output_gradient']>0", "            assert witness['extra_alone_output_gradient']>=0")
text=replace_once(text,"        norm=torch.nn.utils.clip_grad_norm_(tuple(trainable.values()),spec['clip_norm'])",
    "        if args.mode=='preflight':\n            witness['raw_parameter_gradient_norms']={name:float(parameter.grad.norm()) for name,parameter in trainable.items()}\n        norm=torch.nn.utils.clip_grad_norm_(tuple(trainable.values()),spec['clip_norm'])")
text=replace_once(text,"                    actual_empty_row_and_clipped_target_exercised=True,diagnostic_native_head_replay_calls=1)",
    "                    observed_empty_extra_rows=sum(count==0 for count in extra_counts['extra_row_counts']),\n                    observed_clipped_extra_faces=extra_counts['extra_boundary_outside'],diagnostic_native_head_replay_calls=1)")
text=replace_once(text,"            reference_localization_loss=float(reference_loss),reference_enabled=spec['reference_enabled'],\n            reference_loss_weight=spec['reference_loss_weight'],",
    "            reference_mode=spec['reference_mode'],")
text=replace_once(text,"    if formal:\n        evaluate('formal')\n        return", """    if formal:
        evaluate(args.mode)
        if args.mode=='initial_formal':
            assert not optimizer.state
            torch.save(dict(state_delta={name:value.detach().cpu() for name,value in model.state_dict().items() if name in selected},
                optimizer=optimizer.state_dict(),step=0,zero_update_architecture=True,
                reference_mode=spec['reference_mode'],common_output_reset=spec['common_output_reset'],
                retained_hidden_prior_updates=11169,reset_output_total_updates=0,
                checkpoint_sha256=spec['checkpoint_sha256'],base_terminal_sha256=spec['base_terminal_sha256'],
                geometry_terminal_sha256=spec['geometry_terminal_sha256'],source_port_sha256=spec['source_port_sha256'],
                mask_reference_sha256=spec['mask_reference_sha256'],spec_sha256=sha(args.spec)),str(output/'initial.pth'))
        return""")
text=replace_once(text,"        witnesses=[step(batch_cpu,True) for _ in range(2)]", """        witnesses=[step(batch_cpu,True) for _ in range(2)]
        invalid_fixture=check_invalid_reference(output.parent)
        assert witnesses[0]['neutral_initial_decode_equals_reference']
        assert witnesses[1]['raw_parameter_gradient_norms']['candidate_box_refiner.output.weight']>0
        assert all(value>0 for name,value in witnesses[1]['raw_parameter_gradient_norms'].items()
                   if not name.startswith('candidate_box_refiner.output.'))""")
text=replace_once(text,"            isolated_extra_geometry_gradient_verified=True,extra_geometry_weight=spec['extra_geometry_weight'],\n            reference_enabled=spec['reference_enabled'],reference_loss_weight=spec['reference_loss_weight'],",
    "            qualified_extra_output_gradient_scope_verified=True,extra_geometry_weight=spec['extra_geometry_weight'],\n            reference_mode=spec['reference_mode'],common_output_reset=spec['common_output_reset'],invalid_reference_fixture=invalid_fixture,")
text=replace_once(text,"            geometry_parent_fit_updates=11169,total_geometry_fit_updates=11169+step_number,\n            reference_enabled=spec['reference_enabled'],reference_loss_weight=spec['reference_loss_weight'],\n            support_reference_sha256=spec['support_reference_sha256'],",
    "            retained_hidden_prior_updates=11169,retained_hidden_total_updates=11169+step_number,\n            reset_output_total_updates=step_number,common_output_reset=spec['common_output_reset'],\n            reference_mode=spec['reference_mode'],mask_reference_sha256=spec['mask_reference_sha256'],")
assert 'support_reference' not in text and 'reference_enabled' not in text and 'reference_loss' not in text
ast.parse(text)
(root/'run_geometry_fit.py').write_text(text,encoding='utf-8')
(root/'query_supported_geometry.py').write_bytes((previous/'query_supported_geometry.py').read_bytes())

base=json.loads((previous/'control_spec.json').read_bytes())
for key in ('reference_enabled','reference_loss_weight','support_reference_sha256','geometry_parent_fit_updates','total_geometry_fit_updates'):
    del base[key]
base.update(mask_reference_sha256=digest(root/'mask_reference.py'),common_output_reset=['output.weight','output.bias'],
    retained_hidden_prior_updates=11169,retained_hidden_total_updates=14892,reset_output_total_updates=3723)
for arm,mode in (('native_reference','native'),('fused_mask_reference','fused_mask')):
    spec=copy.deepcopy(base)
    spec.update(root='/root/autodl-tmp/pvground_mask_reference_20261006/'+arm,reference_mode=mode)
    (root/(arm+'_spec.json')).write_text(json.dumps(spec,indent=2,sort_keys=True)+'\n',encoding='utf-8')

controller=(previous/'controller.py').read_text(encoding='utf-8')
controller=controller.replace('control_spec.json','native_reference_spec.json')
controller=controller.replace("('control','support_reference')","('native_reference','fused_mask_reference')")
controller=controller.replace("('train','formal')","('initial_formal','train','formal')")
ast.parse(controller)
(root/'controller.py').write_text(controller,encoding='utf-8')
print(json.dumps(dict(status='PREPARED_NOT_EXECUTED',files={name:digest(root/name) for name in
    ('run_geometry_fit.py','query_supported_geometry.py','controller.py','native_reference_spec.json','fused_mask_reference_spec.json')})))
