import ast
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
prior=root.parent/'pvground_final_quality_20261005'
source_path=prior/'runtime_bundle/run_final_quality_fit.py'
source=source_path.read_text(encoding='utf-8')
assert hashlib.sha256(source_path.read_bytes()).hexdigest()=='47457bc0a95161287e633aa423274de4f701a7cc482bfc95f9a890d9207a5379'
setup=source[:source.index('    evaluator_path = runtime')]
setup=setup.replace('"""Frozen4506 R fit with training-only native final-IoU differences.\n\nImplementation draft; source review and two-step preflight precede launch.\n"""','"""Protected4506 head continuation with Query-supported geometry targets."""')
setup=setup.replace("assert spec['geometry_hits50'] == 4506 and spec['geometry_frozen']", "assert spec['geometry_hits50']==4506 and spec['head_only']")
setup=setup.replace("assert spec['use_geometry_evidence'] is True and spec['quality_weight'] == 1.0", "assert spec['use_geometry_evidence'] is True and spec['extra_geometry_weight'] in (0.0,1.0)")
setup=setup.replace("assert sha(output / name) == digest, name", "assert sha(Path(spec['helper_root']) / name)==digest,name")
setup=setup.replace('    sys.path.insert(0, str(model_source))', "    sys.path.insert(0,spec['helper_root'])\n    sys.path.insert(0,str(output.parent))\n    sys.path.insert(0,str(model_source))")
start=setup.index('    from readback_preflight_checks import')
stop=setup.index('    imported =',start)
setup=setup[:start]+'''    from readback_preflight_checks import observed_readback_forward
    from pvground_semantic_assignment import semantic_assignment_correction
    from pvground_boundary_box_refiner import distribution_loss
    from query_supported_geometry import query_supported_geometry_loss
    from whole_model_preflight_checks import optimizer_restore_exact
'''+setup[stop:]
setup=setup.replace('    readback = model.boundary_evidence_readback', '''    for parameter in model.parameters():
        parameter.requires_grad_(False)
    geometry_head=model.candidate_box_refiner
    for parameter in geometry_head.parameters():
        parameter.requires_grad_(True)
    initial={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    model.eval()
    load.update(frozen_geometry_provider=False,frozen_parent_and_R=True,geometry_head_only=True,
        fresh_readback_optimizer_required=False,fresh_geometry_optimizer_required=True)
    write_json(output/'load.json',load)''')
setup=setup.replace("assert len(trainable) == 23 and all(name.startswith('boundary_evidence_readback.') for name in trainable)", "assert len(trainable)==10 and all(name.startswith('candidate_box_refiner.') for name in trainable)\n    assert sum(parameter.numel() for parameter in trainable.values())==456102")
setup=setup.replace('    core_names = set(initial)', '    core_names=set(initial)-set(trainable)')
middle='''    evaluator_path=runtime/'PV-Ground/src/grounding_evaluator.py'
    evaluator_spec=importlib.util.spec_from_file_location('pvground_official_evaluator',str(evaluator_path))
    evaluator_module=importlib.util.module_from_spec(evaluator_spec)
    evaluator_spec.loader.exec_module(evaluator_module)
    GroundingEvaluator=evaluator_module.GroundingEvaluator
    from native_root_bbs import native_root_bbs
    selected=set(trainable)
    optimizer=torch.optim.AdamW(tuple(trainable.values()),lr=spec['lr'],weight_decay=spec['weight_decay'])
    if formal:
        terminal=torch.load(str(output/'terminal.pth'),map_location='cpu')
        assert terminal['step']==3723 and terminal['spec_sha256']==sha(args.spec)
        assert terminal['head_only'] and terminal['extra_geometry_weight']==spec['extra_geometry_weight']
        for key in ('checkpoint_sha256','base_terminal_sha256','geometry_terminal_sha256','source_port_sha256'):
            assert terminal[key]==spec[key]
        assert set(terminal['state_delta'])==selected
        assert Counter(terminal['row_ids'])==Counter(json.loads(Path(manifest['split_protocol']).read_bytes())['row_ids']['fit'])
        restored=dict(initial,**terminal['state_delta'])
        model.load_state_dict(restored,strict=True)
        optimizer.load_state_dict(terminal['optimizer'])
        assert all(int(state['step'])==3723 for state in optimizer.state.values())
        assert all(torch.equal(value.detach().cpu(),restored[name]) for name,value in model.state_dict().items())
        write_json(output/'formal_restore.json',dict(status='pass',terminal_sha256=sha(output/'terminal.pth'),
            optimizer=optimizer_restore_exact(optimizer,terminal['optimizer']),strict_model_restore=True))
'''
helpers=source[source.index('    def loader(part, shuffle):'):source.index('    @torch.no_grad()\n    def evaluate(stage):')]
evaluation=source[source.index('    @torch.no_grad()\n    def evaluate(stage):'):source.index('    if formal:\n        evaluate(\'formal\')')]
start=evaluation.index('                # This is an explicitly separate fixed-Query diagnostic')
stop=evaluation.index('                boxes =',start)
evaluation=evaluation[:start]+evaluation[stop:]
evaluation=evaluation.replace("                    bypass_query = int(bypass_score[bid].argsort(descending=True)[0])\n",'')
evaluation=evaluation.replace("                        bypass_fixed_frame=dict(query=bypass_query, box=boxes[bid, bypass_query].cpu().tolist(),\n                            iou=float(iou[bypass_query])),\n",'')
evaluation=evaluation.replace("                        diagnostic_native_head_replay_calls=1)","                        diagnostic_native_head_replay_calls=0)")
start=evaluation.index('        direct = {}')
stop=evaluation.index('        receipt =',start)
evaluation=evaluation[:start]+evaluation[stop:]
evaluation=evaluation.replace("            fixed_frame_readback_effect=direct, primary_mode='bbs', primary_threshold=.5,", "            primary_mode='bbs',primary_threshold=.5,")
evaluation=evaluation.replace('READBACK_EVAL_COMPLETE','QUERY_SUPPORTED_GEOMETRY_EVAL_COMPLETE')
runner=setup+middle+helpers+evaluation+(root/'fit_body.py').read_text(encoding='utf-8')+"\n\nif __name__=='__main__':\n    main()\n"
ast.parse(runner,feature_version=(3,7))
assert 'native_final_quality_loss' not in runner and 'bypass_score' not in runner
(root/'run_geometry_fit.py').write_text(runner,encoding='utf-8')
spec=json.loads((prior/'quality_fit_spec.json').read_bytes())
for key in ('quality_weight','control_root','preflight_root','geometry_frozen'):
    spec.pop(key)
spec.update(root='/root/autodl-tmp/pvground_query_supported_geometry_20261005/control',
    helper_root='/root/autodl-tmp/pvground_final_quality_20261005/quality',head_only=True,
    extra_geometry_weight=0.0,primary_mode='bbs',primary_threshold=.5,
    additional_geometry_qualification='unmatched; candidate Query and fused root Mask IoU>.5; final Box IoU<=.5')
for arm,weight in (('control',0.0),('query_supported',1.0)):
    value=dict(spec,root='/root/autodl-tmp/pvground_query_supported_geometry_20261005/'+arm,extra_geometry_weight=weight)
    value['preflight_batch_index']=json.loads((root/'PREFLIGHT_PANEL.json').read_bytes())['chosen']['batch_index']
    (root/(arm+'_spec.json')).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
(root/'GENERATION.json').write_text(json.dumps(dict(source=str(source_path),source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
    runner_sha256=hashlib.sha256((root/'run_geometry_fit.py').read_bytes()).hexdigest(),python37_parsed=True,
    drafted_only=True,source_review_pending=True,optimizer_updates_executed=0),indent=2)+'\n',encoding='utf-8')
print(json.dumps({'runner_lines':len(runner.splitlines()),'status':'DRAFT_NOT_DEPLOYED'}))
