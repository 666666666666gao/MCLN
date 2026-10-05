"""Reuse the existing fit lifecycle, changing only this comparison's bindings."""
import ast
from pathlib import Path

root = Path(__file__).resolve().parent
assert not (root / 'launch_geometry_fit_authorized.py').exists()
text = (root.parent / 'pvground_query_supported_geometry_20261005/launch_geometry_fit_authorized.py').read_text(encoding='utf-8')
text = text.replace("'query_supported'", "'member_target'").replace("'query_supported_spec.json'", "'member_target_spec.json'")
old = "assert proof['extra_geometry_weight']==(0.0 if arm=='control' else 1.0)"
assert text.count(old) == 1
text = text.replace(old, "assert proof['extra_geometry_weight']==1.0\n    assert proof['native_data_and_member_target_exact']")
old = "assert witness['actual_empty_row_and_clipped_target_exercised']"
assert text.count(old) == 1
text = text.replace(old, old + "\n        assert witness['auxiliary_target_mode']==('native_gt' if arm=='control' else 'member_gt')")
text = text.replace("pvg_query_geometry_fit_20261005", "pvg_auxiliary_target_fit_20261005")
text = text.replace('geometry_parent_fit_updates=3723', 'geometry_parent_fit_updates=7446')
text = text.replace('geometry_head_total_updates_at_terminal=7446', 'geometry_head_total_updates_at_terminal=11169')
text = text.replace('protected_best_hits=[5616,4506]', 'protected_best_hits=[5614,4509]')
text = text.replace('estimated_seconds=15700', 'estimated_seconds=16400')
text = text.replace("Actual previous fit including initial/terminal6887 evaluation6427.53s plus formal9508 evaluation1422.47s, two sequential arms. Geometry backward may change throughput.",
                    "Actual closed responsibility pair16295.32s, same frozen456102 head, input and evaluation budget. Member-target qualification may change backward throughput.")
ast.parse(text)
(root / 'launch_geometry_fit_authorized.py').write_text(text, encoding='utf-8')
print('fit launcher prepared, no launch')
