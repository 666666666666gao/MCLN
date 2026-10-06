"""Prepare one zero-update three-forward measurement; preserve failed R1 source."""
import ast
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
debug=root/'initial_comparison';debug.mkdir(exist_ok=True)
original=(root/'referit_model_preflight.py').read_text(encoding='utf-8')
capture='''    hook_values=[]
    hook=model.x_query.register_forward_hook(lambda module,args,result:hook_values.append(result.detach().cpu().clone()))
    def capture(predictions):
        return dict(semantic=predictions['last_sem_cls_scores'].detach().cpu().clone(),
            center=predictions['last_center'].detach().cpu().clone(),size=predictions['last_pred_size'].detach().cpu().clone(),
            query=[x.detach().cpu().clone() for x in predictions['sp_last_pred_masks']],
            text=[x[0].detach().cpu().clone() for x in predictions['last_pred_masks']],
            alpha=[x.detach().cpu().clone() for x in predictions['adaptive_weights']],
            super_xyz=[x.detach().cpu().clone() for x in predictions['super_xyz_list']],
            x_query=hook_values[-1].clone(),query_xyz=predictions['query_points_xyz'].detach().cpu().clone())
    def differences(before,after):
        result=dict(semantic_max_abs=float((before['semantic']-after['semantic']).abs().max()),
            center_max_abs=float((before['center']-after['center']).abs().max()),
            size_max_abs=float((before['size']-after['size']).abs().max()),
            x_query_max_abs=float((before['x_query']-after['x_query']).abs().max()),
            query_xyz_max_abs=float((before['query_xyz']-after['query_xyz']).abs().max()),rows=[])
        for bid in range(8):
            a=before['query'][bid];b=after['query'][bid]
            fused_a=before['alpha'][bid]*before['text'][bid]+(1-before['alpha'][bid])*a
            fused_b=after['alpha'][bid]*after['text'][bid]+(1-after['alpha'][bid])*b
            result['rows'].append(dict(row_id=int(batch_cpu['local_training_id'][bid]),
                sample_dataset=batch_cpu['sample_dataset'][bid],query_logit_elements=a.numel(),
                query_mask_max_abs=float((a-b).abs().max()),query_mask_mean_abs=float((a-b).abs().mean()),
                query_mask_rms=float((a-b).square().mean().sqrt()),
                query_mask_original_allclose=bool(torch.allclose(a,b,rtol=1e-5,atol=1e-5)),
                query_mask_sign_changes=int((a.gt(0)!=b.gt(0)).sum()),
                fused_mask_sign_changes=int((fused_a.gt(0)!=fused_b.gt(0)).sum()),
                text_mask_max_abs=float((before['text'][bid]-after['text'][bid]).abs().max()),
                alpha_abs=float((before['alpha'][bid]-after['alpha'][bid]).abs()),
                super_xyz_max_abs=float((before['super_xyz'][bid]-after['super_xyz'][bid]).abs().max())))
        return result
'''
anchor='    model.cuda();model.eval();torch.cuda.reset_peak_memory_stats();reset_rng()\n'
assert original.count(anchor)==1
source=original.replace(anchor,capture+anchor)
anchor="        native_masks=[value.detach().clone() for value in native['sp_last_pred_masks']]\n"
assert source.count(anchor)==1
source=source.replace(anchor,anchor+"        native_a=capture(native)\n")
anchor='    del native\n'
assert source.count(anchor)==1
source=source.replace(anchor,anchor+'''    reset_rng()
    with torch.no_grad():
        native_b_output=model(dict(inputs));native_b=capture(native_b_output)
    del native_b_output
''')
cut="        semantic_error=float((predictions['last_sem_cls_scores']-native_semantic).abs().max())\n"
index=source.index(cut)
source=source[:index]+'''        installed=capture(predictions)
        reference_witness=reference_bounds_witness(predictions,batch)
    hook.remove();assert len(hook_values)==3
    record=dict(status='ACTUAL_ZERO_UPDATE_THREE_FORWARD_COMPARISON',
        time_cst=datetime.datetime.now().astimezone().isoformat(),dataset=dataset_name,reference_mode=spec['reference_mode'],
        model_forwards=3,optimizer_steps=0,formal_rows=0,new_weights_saved=0,
        identical_rng_before_each_forward=True,all_official_initial_states_exact=True,
        native_A_vs_native_B=differences(native_a,native_b),
        native_A_vs_installed_C=differences(native_a,installed),
        native_B_vs_installed_C=differences(native_b,installed),
        reference_witness=reference_witness,elapsed_seconds=time.perf_counter()-start,
        script_sha256=sha(__file__),spec_sha256=sha(args.spec))
    write_json(output/'comparison.json',record)
    print('ZERO_UPDATE_COMPARISON_COMPLETE '+json.dumps(record),flush=True)


if __name__=='__main__':main()
'''
source=source.replace('native=model(inputs)','native=model(dict(inputs))').replace('predictions=model(inputs)','predictions=model(dict(inputs))')
ast.parse(source)
(debug/'compare_initial.py').write_text(source,encoding='utf-8')
for name in ('referit_training_targets.py','pvground_referit_fit_dataset.py','mask_reference.py','query_supported_geometry.py'):
    (debug/name).write_bytes((root/name).read_bytes())
spec=json.loads((root/'preflight_spec.json').read_bytes())['runs']['nr3d_native']
spec.update(root='/root/autodl-tmp/pvground_referit_initial_comparison_20261006/nr3d_native',
    files={name:hashlib.sha256((debug/name).read_bytes()).hexdigest() for name in
        ('compare_initial.py','referit_training_targets.py','pvground_referit_fit_dataset.py','mask_reference.py','query_supported_geometry.py')})
(debug/'spec.json').write_text(json.dumps(spec,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status='ZERO_UPDATE_COMPARISON_SOURCE_PREPARED_NOT_EXECUTED',new_script_sha256=spec['files']['compare_initial.py'])))
