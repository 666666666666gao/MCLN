from pathlib import Path

root=Path(__file__).resolve().parent
source=(root/'diagnose_closed_face_coupling_cpu.py').read_text()
assert "output=root/'face_coupling_diagnostic_20261010'" in source
source=source.replace("output=root/'face_coupling_diagnostic_20261010'","output=root/'face_coupling_exact_20261010'")
source=source.replace('own_selected_fused_mask_qualified_strict_errors','selected_fused_mask_qualified_strict_errors')
anchor="bound_limited=upper_bound<=.5-1e-6"
assert source.count(anchor)==1
extra='''# For fixed other axes, IoU3D is a ratio of affine functions between
# this axis's GT-face crossings. It is monotone or constant on each segment.
# A global maximum thus occurs on the Cartesian product of these breakpoints.
import itertools
exact_coupled=np.zeros(9508,dtype=np.float64)
best_gate=np.zeros((9508,3),dtype=np.float64)
for choice in itertools.product(range(len(candidates)),repeat=3):
    gate=np.stack([candidates[choice[axis]][:,axis] for axis in range(3)],1)
    current_low=m_low+gate*lower_slopes
    current_high=m_high+gate*upper_slopes
    box=np.concatenate([(current_low+current_high)/2,current_high-current_low],1)
    current=iou(box,gt)
    better=current>exact_coupled
    best_gate[better]=gate[better]
    exact_coupled=np.maximum(exact_coupled,current)
assert (exact_coupled<=upper_bound+1e-8).all()
assert (exact_coupled<=independent_iou+1e-8).all()
assert max(float((iou(box,gt)-exact_coupled).max()) for box in precise.values())<1e-6
exact_limited=exact_coupled<=.5-1e-6
'''
source=source.replace(anchor,extra+anchor)
source=source.replace("evidence=bound_limited & independent_good","evidence=exact_limited & independent_good")
source=source.replace("coupled_projection_upper_bound_below_half=int((select & bound_limited).sum()),",
    "coupled_projection_upper_bound_below_half=int((select & bound_limited).sum()),\n        exact_continuous_coupled_oracle_below_half=int((select & exact_limited).sum()),\n        exact_continuous_coupled_oracle_above_half=int((select & (exact_coupled>.5+1e-6)).sum()),")
source=source.replace("'extremal_iou','coupled_3d_upper_bound','GT_independent_face_projection_iou','opposed_axes',",
    "'extremal_iou','coupled_3d_upper_bound','GT_independent_face_projection_iou','exact_coupled_iou','opposed_axes',")
source=source.replace("upper_bound[index],independent_iou[index],int(opposed[index].sum()),int(evidence[index])])",
    "upper_bound[index],independent_iou[index],exact_coupled[index],int(opposed[index].sum()),int(evidence[index])])")
source=source.replace("status='CLOSED_SELECTED_QUERY_FACE_COUPLING_OFFLINE_DIAGNOSTIC'",
    "status='CLOSED_SELECTED_QUERY_EXACT_FACE_COUPLING_OFFLINE_DIAGNOSTIC'")
source=source.replace("maximum_observed_bound_violation=observed_max_violation,",
    "maximum_observed_bound_violation=observed_max_violation,\n    exact_coupled_method='216 Cartesian GT-crossing/endpoint combinations; coordinate-wise piecewise linear-fractional optimum',\n    exact_coupled_oracle_hits=[int((exact_coupled>t+1e-6).sum()) for t in (.25,.5)],\n    independent_face_projection_hits=[int((independent_iou>t+1e-6).sum()) for t in (.25,.5)],\n    first_diagnostic_group_label_corrected='Previous own_selected_fused label actually used only fused Mask IoU; own Query qualification is not established',")
source=source.replace("(output/'FACE_COUPLING_DIAGNOSTIC.json')","(output/'FACE_COUPLING_EXACT_DIAGNOSTIC.json')")
target=root/'diagnose_closed_face_coupling_exact_cpu.py'
assert not target.exists()
target.write_text(source)
print('EXACT_CONTINUOUS_COUPLED_DIAGNOSTIC_PREPARED_NO_NEURAL_CALLS')
