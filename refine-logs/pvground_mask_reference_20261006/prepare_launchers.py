"""Adapt the already closed SSH launch lifecycle to the reviewed new pair."""
import ast
from pathlib import Path

root=Path(__file__).resolve().parent
previous=root.parent/'pvground_support_reference_20261005'

def replace_once(text,old,new):
    assert text.count(old)==1,old[:100]
    return text.replace(old,new,1)

text=(previous/'deploy_preflight_authorized.py').read_text(encoding='utf-8')
text=replace_once(text,"previous=local.parent/'pvground_auxiliary_target_20261005'", "previous=local.parent/'pvground_mask_extent_diagnostic_20261006'")
text=replace_once(text,"assert json.loads((previous/'terminal_publication.json').read_bytes())['retained_best']['hits']==[5616,4511]",
    "assert json.loads((previous/'analysis/SUMMARY.json').read_bytes())['protected_trained_model_hits']==[5616,4511]")
text=text.replace('pvground_support_reference_20261005','pvground_mask_reference_20261006')
text=text.replace("('control', 'support_reference')","('native_reference','fused_mask_reference')")
text=text.replace('control_spec.json','native_reference_spec.json').replace('support_reference_spec.json','fused_mask_reference_spec.json')
text=text.replace("'support_reference.py'", "'mask_reference.py','check_invalid_reference.py','invalid_reference_fixture.npz','invalid_reference_fixture.json'")
text=replace_once(text,"root=Path(sys.argv[1]);closed=Path('/root/autodl-tmp/pvground_auxiliary_target_20261005')",
    "root=Path(sys.argv[1]);closed=Path('/root/autodl-tmp/pvground_mask_extent_diagnostic_20261006')")
text=replace_once(text,"assert json.loads((closed/'fit_status.json').read_bytes())['status']=='complete'\nassert (closed/'fit_controller.exit').read_text().strip()=='0'\nassert json.loads((closed/'weight_retention.json').read_bytes())['retained_best']['hits']==[5616,4511]",
    "closed_status=json.loads((closed/'formal_status.json').read_bytes())\nassert closed_status['completed'] and closed_status['exit_code']==0\nassert (closed/'formal.exit').read_text().strip()=='0'")
text=text.replace('pvg_support_reference_preflight_20261005','pvg_mask_reference_preflight_20261006')
text=replace_once(text,"estimated_seconds=1200, first_check_seconds=900, later_poll_seconds=240,",
    "estimated_seconds=900, first_check_seconds=720, later_poll_seconds=240,")
text=replace_once(text,"estimate_basis='Two sequential protected4511 reconstructions; own-support reference and extra reference gradient/replay check. No formal accuracy.'",
    "estimate_basis='Two warm protected4511 model reconstructions;2 updates each, all256 actual point extent check and39 recorded empty supports. No accuracy result.'")
ast.parse(text);(root/'deploy_preflight_authorized.py').write_text(text,encoding='utf-8')

text=(previous/'observe_preflight_authorized.py').read_text(encoding='utf-8')
text=text.replace('control_spec.json','native_reference_spec.json')
ast.parse(text);(root/'observe_preflight_authorized.py').write_text(text,encoding='utf-8')

text=(previous/'collect_preflight_authorized.py').read_text(encoding='utf-8')
text=text.replace("'support_reference.py'", "'mask_reference.py','check_invalid_reference.py','invalid_reference_fixture.json'")
text=text.replace('control_spec.json','native_reference_spec.json').replace('support_reference_spec.json','fused_mask_reference_spec.json')
text=text.replace("('control', 'support_reference')","('native_reference','fused_mask_reference')")
ast.parse(text);(root/'collect_preflight_authorized.py').write_text(text,encoding='utf-8')

text=(previous/'launch_geometry_fit_authorized.py').read_text(encoding='utf-8')
text=text.replace("('control','support_reference')","('native_reference','fused_mask_reference')")
text=replace_once(text,"    assert proof['head_parameters']==(459180 if arm=='support_reference' else 456102)\n    assert proof['head_state_tensors']==(12 if arm=='support_reference' else 10)\n    assert proof['reference_enabled']==(arm=='support_reference')\n    if arm=='support_reference':\n        assert proof['witnesses'][0]['zero_reference_cached_old_head_exact']\n        assert all(item['reference_direct_geometry_gradient']>0 for item in proof['witnesses'])\n    assert proof['all_parent_and_R_states_exact'] and proof['isolated_extra_geometry_gradient_verified']",
    "    assert proof['head_parameters']==456102 and proof['head_state_tensors']==10\n    assert proof['reference_mode']==('native' if arm=='native_reference' else 'fused_mask')\n    assert proof['common_output_reset']==['output.weight','output.bias']\n    assert proof['witnesses'][0]['neutral_initial_decode_equals_reference']\n    assert proof['invalid_reference_fixture']['actual_empty_support_rows_verified']==39\n    assert proof['all_parent_and_R_states_exact'] and proof['qualified_extra_output_gradient_scope_verified']")
text=replace_once(text,"        assert witness['actual_empty_row_and_clipped_target_exercised']",
    "        assert witness['actual_all256_raw_member_extent_verified']")
text=text.replace('control_spec.json','native_reference_spec.json').replace('support_reference_spec.json','fused_mask_reference_spec.json')
text=text.replace("'support_reference.py'", "'mask_reference.py','check_invalid_reference.py','invalid_reference_fixture.npz','invalid_reference_fixture.json'")
text=replace_once(text,"reserve=3*max(proof['serialization_bytes'] for proof in proofs.values())+256*1024**2",
    "reserve=3*max(proof['serialization_bytes'] for proof in proofs.values())+900*1024**2")
text=text.replace('pvg_support_reference_fit_20261005','pvg_mask_reference_fit_20261006')
text=replace_once(text,"updates_per_arm=3723,tail_batch_rows=2,geometry_parent_fit_updates=11169,\n    geometry_head_total_updates_at_terminal=14892,initial_holdout_rows=6887,terminal_holdout_rows=6887,formal_rows_per_arm=9508,\n    trainable_parameters_by_arm=dict(control=456102,support_reference=459180),trainable_state_tensors_by_arm=dict(control=10,support_reference=12),parent_and_R_frozen=True,",
    "updates_per_arm=3723,tail_batch_rows=2,retained_hidden_prior_updates=11169,\n    retained_hidden_total_updates_at_terminal=14892,reset_output_total_updates_at_terminal=3723,\n    common_output_reset=['output.weight','output.bias'],initial_holdout_rows=6887,terminal_holdout_rows=6887,\n    initial_formal_rows_per_arm=9508,terminal_formal_rows_per_arm=9508,\n    trainable_parameters_by_arm=dict(native_reference=456102,fused_mask_reference=456102),\n    trainable_state_tensors_by_arm=dict(native_reference=10,fused_mask_reference=10),parent_and_R_frozen=True,")
text=replace_once(text,"first_check_seconds=6200,later_poll_seconds=240,estimated_seconds=18000,\n    estimate_basis='Actual closed auxiliary-target pair16743.09s. Same data/evaluation; added3078 reference parameters and second support aggregation. Estimate5h, first check near control terminal.'",
    "first_check_seconds=21600,later_poll_seconds=240,estimated_seconds=22000,\n    estimate_basis='Closed prior pair about5h plus2 initial9508 evaluations (actual diagnostic1883s each); same frozen-parent fit budget. Estimated6.1h, sole first observer check about7minutes before projected final closure. Calibrate with real progress if needed.'")
ast.parse(text);(root/'launch_geometry_fit_authorized.py').write_text(text,encoding='utf-8')

plan=(root/'EXPERIMENT_PLAN.md').read_text(encoding='utf-8')
plan=plan.replace('180\ufffdC300','180-300').replace('@.50\ufffd\ufffd50%','@.50 >=50%')
(root/'EXPERIMENT_PLAN.md').write_text(plan,encoding='utf-8')
(root/'research_contract.md').write_text('PV-Ground single deployed network; original last/bbs, sameQueryBox/Mask, all256. Claim under test: full predicted instance range as spatial reference permits trained native strict-localization gains versus common-output-reset native-reference control and protected4511. Offline4848 is not a trained method result. Do not claim novelty or Nr/Sr gain before direct formal evidence.\n',encoding='utf-8')
(root/'EXPERIMENT_TRACKER.md').write_text('M0: PREPARED_NOT_DEPLOYED;2 actual updates per arm after source review.\nM1: NOT_STARTED; common reset native/fused spatial-reference pair.\nM2: NOT_STARTED; fresh terminal integrity and strict-best retention.\n',encoding='utf-8')
(root/'MANIFEST.md').write_text('2026-10-06: prepared mask_reference.py, native/fused specs, runner/controller and SSH lifecycle. No deployment, optimizer update or accuracy result. Actual39 empty-support fixture contains predicted support/member geometry and native prior, no GT fields.\n',encoding='utf-8')
print('PREPARED_SOURCE_LIFECYCLE_NOT_EXECUTED')
