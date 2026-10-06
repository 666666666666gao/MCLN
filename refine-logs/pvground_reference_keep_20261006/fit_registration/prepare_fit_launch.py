"""Reuse the closed prior pair's deployment and collection, with this pair's gates."""
from pathlib import Path

root = Path(__file__).resolve().parent
prior = root.parent / 'pvground_mask_reference_20261006'
source = (prior / 'launch_geometry_fit_authorized.py').read_text(encoding='utf-8')
source = source.replace("local/'SOURCE_REVIEW.json'", "local/'FIT_SOURCE_REVIEW.json'")
source = source.replace("('native_reference','fused_mask_reference')", "('control','keep')")
source = source.replace("('native' if arm=='native_reference' else 'fused_mask')", "'fused_mask'")
source = source.replace("'native_reference_spec.json'", "'control_spec.json'")
source = source.replace("'fused_mask_reference_spec.json'", "'keep_spec.json'")
source = source.replace("'run_geometry_fit.py'", "'run_reference_keep_fit.py','reference_keep.py','selected_mask_reference_factory.py'")
source = source.replace("screen='pvg_mask_reference_fit_20261006'", "screen='pvg_reference_keep_fit_20261007'")
source = source.replace("trainable_parameters_by_arm=dict(native_reference=456102,fused_mask_reference=456102)", "trainable_parameters_by_arm=dict(control=456102,keep=456102)")
source = source.replace("trainable_state_tensors_by_arm=dict(native_reference=10,fused_mask_reference=10)", "trainable_state_tensors_by_arm=dict(control=10,keep=10)")
source = source.replace("protected_best_hits=[5616,4511]", "protected_best_hits=[5598,4848],candidate_gate_hits=[5620,4764],no_multiseed=True,")
source = source.replace("no_multiseed=True,,", "no_multiseed=True,")
source = source.replace("first_check_seconds=21600,later_poll_seconds=240,estimated_seconds=22000", "first_check_seconds=24300,later_poll_seconds=240,estimated_seconds=25200")
source = source.replace("Closed prior pair about5h plus2 initial9508 evaluations (actual diagnostic1883s each); same frozen-parent fit budget. Estimated6.1h, sole first observer check about7minutes before projected final closure. Calibrate with real progress if needed.", "Closed previous pair24792s (6h53min), same frozen-parent budget plus train-only keep loss. Estimate7h; first sole observation6h45min after launch, then240s near closure.")
source = source.replace("    assert proof['extra_geometry_weight']==1.0", "    assert proof['extra_geometry_weight']==1.0\n    assert proof['reference_keep_weight']==(0.0 if arm=='control' else 1.0)\n    assert proof['witnesses'][0]['neutral_initial_reference_keep_exact_zero']\n    assert proof['runner_sha256']==hashlib.sha256((local/'run_reference_keep_fit.py').read_bytes()).hexdigest()\n    assert proof['spec_sha256']==hashlib.sha256((local/(arm+'_spec.json')).read_bytes()).hexdigest()")
source = source.replace("        assert witness['auxiliary_target_mode']=='native_gt'", "        assert witness['auxiliary_target_mode']=='native_gt'\n        assert witness['reference_keep_gradient_only_qualified']\n        assert witness['reference_keep_weight']==proof['reference_keep_weight']")
source = source.replace("spec=json.loads", "assert proofs['keep']['witnesses'][1]['reference_keep_output_gradient_norm']>0\nspec=json.loads", 1)
source = source.replace("python=spec['runtime']+'/venv/bin/python'", "python=spec['runtime']+'/venv/bin/python'\nwith sftp.open(spec['selected_terminal'],'rb') as stream:\n    assert hashlib.sha256(stream.read()).hexdigest()==spec['selected_terminal_sha256']")
source = source.replace("root=root,process=process,screen=screen", "root=root,process=process,controller_pid=int(process.split()[0]),screen=screen")
compile(source, 'launch_fit_authorized.py', 'exec')
(root / 'launch_fit_authorized.py').write_text(source, encoding='utf-8')

collector = (prior / 'collect_closed_fit_authorized.py').read_text(encoding='utf-8').replace("'native_reference_spec.json'", "'control_spec.json'")
compile(collector, 'collect_closed_fit_authorized.py', 'exec')
(root / 'collect_closed_fit_authorized.py').write_text(collector, encoding='utf-8')

observer = (prior / 'observe_fit_authorized.py').read_text(encoding='utf-8').replace("'native_reference_spec.json'", "'control_spec.json'")
observer = observer.replace("launch=str(local/(stage+'_launch.json')),first_check_seconds=launch['first_check_seconds'],poll_seconds=240", "launch=str(local/(stage+'_launch.json')),observer_local_pid=os.getpid(),first_check_seconds=launch['first_check_seconds'],poll_seconds=240,\n    first_observation_cst=(datetime.datetime.fromisoformat(launch['time_cst'])+datetime.timedelta(seconds=launch['first_check_seconds'])).isoformat()")
observer = observer.replace("    time.sleep(240)", "    next_check=datetime.datetime.now().astimezone()+datetime.timedelta(seconds=240)\n    (local/'fit_observer_wait.json').write_text(json.dumps(dict(observer_local_pid=os.getpid(),remote_queries_performed=index,\n        next_observation_cst=next_check.isoformat()),indent=2)+'\\n',encoding='utf-8')\n    time.sleep(240)")
observer = observer.replace('QUERY_GEOMETRY_', 'REFERENCE_KEEP_')
compile(observer, 'observe_fit_authorized.py', 'exec')
(root / 'observe_fit_authorized.py').write_text(observer, encoding='utf-8')
print('FIT_LAUNCH_OBSERVER_COLLECTOR_PREPARED_NOT_EXECUTED')
