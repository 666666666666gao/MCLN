"""Prepare closed-result collection and complete-head best retention only."""
import ast
from pathlib import Path

root=Path(__file__).resolve().parents[1]
previous=root.parent/'pvground_auxiliary_target_20261005'
target=root/'postrun/collect_formal_authorized.py'
assert not target.exists()
text=(previous/'collect_formal_authorized.py').read_text(encoding='utf-8')
assert text.count('local = Path(__file__).resolve().parent')==1
text=text.replace('local = Path(__file__).resolve().parent','local = Path(__file__).resolve().parents[1]')
target.write_text(text,encoding='utf-8')
text=(previous/'retain_metric_best.py').read_text(encoding='utf-8')


def change(before,after):
    global text
    assert text.count(before)==1,before
    text=text.replace(before,after)


change("assert str(root)=='/root/autodl-tmp/pvground_auxiliary_target_20261005'",
       "assert str(root)=='/root/autodl-tmp/pvground_support_reference_20261005'")
change("assert str(parent)=='/root/autodl-tmp/pvground_query_supported_geometry_20261005/query_supported/terminal.pth'",
       "assert str(parent)=='/root/autodl-tmp/pvground_auxiliary_target_20261005/control/terminal.pth'")
text=text.replace('member_target','support_reference')
change("assert payload['step']==3723 and payload['geometry_parent_fit_updates']==7446 and payload['total_geometry_fit_updates']==11169",
       "assert payload['step']==3723 and payload['geometry_parent_fit_updates']==11169 and payload['total_geometry_fit_updates']==14892")
change("assert len(state)==10 and set(state)==set(parent_state) and sum(value.numel() for value in state.values())==456102\n    assert all(name.startswith('candidate_box_refiner.') and value.shape==parent_state[name].shape for name,value in state.items())",
       "assert payload['reference_enabled']==(arm=='support_reference')\n"
       "    assert payload['reference_loss_weight']==(1.0 if arm=='support_reference' else 0.0)\n"
       "    from pvground_boundary_box_refiner import BoundaryBoxRefiner\n"
       "    from support_reference import SupportReferenceBoxRefiner\n"
       "    head=SupportReferenceBoxRefiner() if arm=='support_reference' else BoundaryBoxRefiner('distribution')\n"
       "    assert len(state)==(12 if arm=='support_reference' else 10)\n"
       "    assert sum(value.numel() for value in state.values())==(459180 if arm=='support_reference' else 456102)\n"
       "    assert all(name.startswith('candidate_box_refiner.') for name in state)\n"
       "    stripped={name[len('candidate_box_refiner.'):]:value for name,value in state.items()}\n"
       "    assert set(stripped)==set(head.state_dict())\n"
       "    head.load_state_dict(stripped,strict=True)\n"
       "    assert all(torch.equal(value,head.state_dict()[name]) for name,value in stripped.items())\n"
       "    assert all(state[name].shape==value.shape for name,value in parent_state.items())")
change("assert payload['auxiliary_target_mode']==('native_gt' if arm=='control' else 'member_gt')",
       "assert payload['auxiliary_target_mode']=='native_gt'\n    assert payload['support_reference_sha256']==spec['support_reference_sha256']")
change("assert len(payload['optimizer']['state'])==10 and all(int(value['step'])==3723 for value in payload['optimizer']['state'].values())",
       "assert len(payload['optimizer']['state'])==len(state) and all(int(value['step'])==3723 for value in payload['optimizer']['state'].values())")
change("decision['hits']['protected_geometry_parent']==[5614,4509]","decision['hits']['protected_geometry_parent']==[5616,4511]")
change('best_delta_has_all10_geometry_states=True','best_delta_has_all_geometry_states=True,strict_CPU_head_restore=True')
change('spec=json.loads((root/\'control_spec.json\').read_bytes())',
       "spec=json.loads((root/'control_spec.json').read_bytes())\n"
       "sys.path.insert(0,spec['helper_root'])\n"
       "sys.path.insert(0,str(root))\n"
       "assert hashlib.sha256((root/'support_reference.py').read_bytes()).hexdigest()==spec['support_reference_sha256']\n"
       "assert all(hashlib.sha256(Path(spec['helper_root'],name).read_bytes()).hexdigest()==digest for name,digest in spec['runner_files'].items())")
(root/'postrun/retain_metric_best.py').write_text(text,encoding='utf-8')
text=(previous/'retain_metric_best_authorized.py').read_text(encoding='utf-8')
change('local=Path(__file__).resolve().parent','local=Path(__file__).resolve().parents[1]')
change("summary['status']=='ACTUAL_CLOSED_ROWS_ANALYZED' and summary['training_order_exact']",
       "summary['status']=='ACTUAL_CLOSED_REFERENCE_ROWS_ANALYZED' and summary['fit_order_exact']")
change("assert not (summary['scanrefer_target_pass'] and not summary['metric_best_target_pass']), 'Metric-best winner conflicts with the existing dual-threshold target; preserve candidates before cleanup.'",
       "assert not (summary['scanrefer_target_pass'] and not summary['metric_best_target_pass']), 'Keep candidates when strict-best selection conflicts with the stated development line.'\n"
       "assert all(not any(system['cpu_selected_coarse_reference_box_threshold_flip_counts'].values()) for system in summary['systems'].values())")
change("(local/'retain_metric_best.py').read_bytes()","(local/'postrun/retain_metric_best.py').read_bytes()")
(root/'postrun/retain_metric_best_authorized.py').write_text(text,encoding='utf-8')
for path in (root/'postrun').glob('*.py'):
    ast.parse(path.read_text(encoding='utf-8'))
print('Prepared closed collector, complete10/12-state CPU head restore and post-audit retention; not executed.')
