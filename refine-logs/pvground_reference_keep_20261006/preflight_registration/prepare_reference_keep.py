"""Prepare the single new loss contrast from the retained4848 complete state."""
import ast
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
old=root.parent/'pvground_mask_reference_20261006'
for name,path in {'selected_mask_reference_factory.py':old/'postrun/selected_mask_reference_factory.py',
    'query_supported_geometry.py':old/'query_supported_geometry.py','mask_reference.py':old/'mask_reference.py',
    'check_invalid_reference.py':old/'check_invalid_reference.py',
    'invalid_reference_fixture.npz':old/'invalid_reference_fixture.npz',
    'invalid_reference_fixture.json':old/'invalid_reference_fixture.json'}.items():
    (root/name).write_bytes(path.read_bytes())
source=(old/'run_geometry_fit.py').read_text(encoding='utf-8')
source=source.replace('Protected4511 hidden states; native versus full fused-Mask reference.',
    'Retained4848 complete Mask-reference state; original versus reference-preserving supervision.')
source=source.replace("assert spec['geometry_hits50']==4511 and spec['head_only']", "assert spec['starting_hits']==[5598,4848] and spec['head_only']")
source=source.replace("assert spec['reference_mode'] in ('native','fused_mask')", "assert spec['reference_mode']=='fused_mask' and spec['reference_keep_weight'] in (0.0,1.0)")
source=source.replace("assert sha(output.parent/'mask_reference.py')==spec['mask_reference_sha256']",
    "assert sha(output.parent/'mask_reference.py')==spec['mask_reference_sha256']\n"
    "    assert sha(output.parent/'reference_keep.py')==spec['reference_keep_sha256']\n"
    "    assert sha(output.parent/'selected_mask_reference_factory.py')==spec['selected_factory_sha256']")
source=source.replace("assert sha(spec['geometry_terminal']) == spec['geometry_terminal_sha256']", "assert sha(spec['selected_terminal'])==spec['selected_terminal_sha256']")
source=source.replace('from readback_model_factory import build_readback_model', 'from selected_mask_reference_factory import build_selected_mask_reference_model')
source=source.replace('    from query_supported_geometry import query_supported_geometry_loss',
    '    from query_supported_geometry import query_supported_geometry_loss\n    from reference_keep import reference_keep_loss')
begin=source.index('    model, config, initial, load = build_readback_model')
end=source.index('    model.cuda()',begin)
source=source[:begin]+'''    selected_payload=torch.load(spec['selected_terminal'],map_location='cpu')
    assert selected_payload['step']==0 and selected_payload['zero_update_architecture']
    assert selected_payload['reference_mode']=='fused_mask' and not selected_payload['optimizer']['state']
    model,config,load=build_selected_mask_reference_model(cfg,
        torch.load(official_weight['path'],map_location='cpu'),
        torch.load(spec['base_terminal'],map_location='cpu'),selected_payload,manifest['data_root'])
    assert load['full_state_tensors']==1304 and not load['old_geometry_checkpoint_required']
    assert torch.count_nonzero(model.candidate_box_refiner.output.weight)==0
    assert torch.count_nonzero(model.candidate_box_refiner.output.bias)==0
    load.update(status='pass',time_cst=datetime.datetime.now().astimezone().isoformat(),
        spec_sha256=sha(args.spec),selected_terminal=spec['selected_terminal'],
        selected_terminal_sha256=spec['selected_terminal_sha256'],reference_keep_weight=spec['reference_keep_weight'],
        common_output_reset=spec['common_output_reset'],retained_hidden_state_tensors=8,reset_output_state_tensors=2)
    write_json(output/'load.json',load)
''' + source[end:]
source=source.replace("('checkpoint_sha256','base_terminal_sha256','geometry_terminal_sha256','source_port_sha256')",
    "('checkpoint_sha256','base_terminal_sha256','selected_terminal_sha256','source_port_sha256')")
source=source.replace("geometry_terminal_sha256=spec['geometry_terminal_sha256']", "selected_terminal_sha256=spec['selected_terminal_sha256']")
source=source.replace("        assert terminal['reference_mode']==spec['reference_mode']", "        assert terminal['reference_mode']==spec['reference_mode']\n        assert terminal['reference_keep_weight']==spec['reference_keep_weight']")
source=source.replace('loss=native+correction+(1.0/7)*edge+spec[\'extra_geometry_weight\']*extra',
    "keep,keep_counts,keep_selected=reference_keep_loss(predictions,batch,matches[1])\n        loss=native+correction+(1.0/7)*edge+spec['extra_geometry_weight']*extra+spec['reference_keep_weight']*keep")
where="            witness.update(reference_bounds_witness(predictions,batch))"
new=where+'''
            keep_grad=torch.autograd.grad(keep,(predictions['last_center'],predictions['last_pred_size']),retain_graph=True)
            for bid,queries in enumerate(keep_selected):
                outside=torch.ones(256,dtype=torch.bool,device=queries.device);outside[queries]=False
                assert all((gradient[bid,outside]==0).all() for gradient in keep_grad)
            keep_output_grad=torch.autograd.grad(keep,geometry_head.output.weight,retain_graph=True)[0]
            witness.update(reference_keep_gradient_only_qualified=True,
                reference_keep_output_gradient_norm=float(keep_output_grad.norm()))'''
assert source.count(where)==1;source=source.replace(where,new)
source=source.replace("witness['neutral_initial_decode_equals_reference']=True", "witness['neutral_initial_decode_equals_reference']=True\n                assert float(keep)==0.0\n                witness['neutral_initial_reference_keep_exact_zero']=True")
source=source.replace('matched_boundary_loss=float(edge),extra_geometry_loss=float(extra),',
    "matched_boundary_loss=float(edge),extra_geometry_loss=float(extra),\n            reference_keep_loss=float(keep),reference_keep_counts=keep_counts,reference_keep_weight=spec['reference_keep_weight'],")
source=source.replace("extra_geometry_weight=spec['extra_geometry_weight'],\n            reference_mode=spec['reference_mode']", "extra_geometry_weight=spec['extra_geometry_weight'],reference_keep_weight=spec['reference_keep_weight'],\n            reference_mode=spec['reference_mode']")
source=source.replace("spec_sha256=sha(args.spec),extra_geometry_weight=spec['extra_geometry_weight'],", "spec_sha256=sha(args.spec),extra_geometry_weight=spec['extra_geometry_weight'],reference_keep_weight=spec['reference_keep_weight'],")
source=source.replace("extra_geometry_weight=spec['extra_geometry_weight'],primary_mode='bbs'", "extra_geometry_weight=spec['extra_geometry_weight'],reference_keep_weight=spec['reference_keep_weight'],primary_mode='bbs'")
# Retain historical total-update fields and the same original evaluator/fit order.
assert "spec['geometry_terminal']" not in source and 'build_readback_model(' not in source
(root/'run_reference_keep_fit.py').write_text(source,encoding='utf-8')
controller=(old/'controller.py').read_text(encoding='utf-8')
controller=controller.replace('native_reference_spec.json','control_spec.json').replace("('native_reference','fused_mask_reference')","('control','keep')")
controller=controller.replace('run_geometry_fit.py','run_reference_keep_fit.py').replace("('base_terminal','geometry_terminal')","('base_terminal','selected_terminal')")
controller=controller.replace('protected_best_hits=[5616,4511]','protected_best_hits=[5598,4848]')
(root/'controller.py').write_text(controller,encoding='utf-8')
base=json.loads((old/'fused_mask_reference_spec.json').read_bytes())
base.pop('geometry_terminal');base.pop('geometry_terminal_sha256');base.pop('geometry_hits50')
base.update(starting_hits=[5598,4848],selected_terminal='/root/autodl-tmp/pvground_mask_reference_20261006/fused_mask_reference/initial.pth',
    selected_terminal_sha256='2301e90a9391af4e4cd56edae09ce49db1cc8f67463e1c2568352ee2dafaff61',
    reference_keep_formula='expression-mean then batch-mean squared ReLU(reference_IoU-final_IoU)',
    reference_keep_qualification='native matches to own filtered GT; unmatched Query AND fused root masks>.5 without box-IoU cutoff',
    reference_keep_sha256=hashlib.sha256((root/'reference_keep.py').read_bytes()).hexdigest(),
    selected_factory_sha256=hashlib.sha256((root/'selected_mask_reference_factory.py').read_bytes()).hexdigest(),
    candidate_gate_hits=[5620,4764],no_multiseed=True)
remote='/root/autodl-tmp/pvground_reference_keep_20261006'
for arm,weight in [('control',0.0),('keep',1.0)]:
    spec=dict(base,root=remote+'/'+arm,reference_keep_weight=weight)
    (root/(arm+'_spec.json')).write_text(json.dumps(spec,indent=2)+'\n',encoding='utf-8')
for name in ('reference_keep.py','run_reference_keep_fit.py','controller.py','selected_mask_reference_factory.py'):
    ast.parse((root/name).read_text(encoding='utf-8'))
print(json.dumps(dict(status='REFERENCE_KEEP_PAIR_PREPARED_NOT_DEPLOYED',seed=2027,arms=['control','keep'],
    new_inference_parameters=0,retained_state=[5598,4848],geometry_parameters=456102)))
