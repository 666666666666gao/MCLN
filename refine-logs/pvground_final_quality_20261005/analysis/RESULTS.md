# Final native-quality supervision: actual closed result

Actual finished CST: 2026-10-05T06:57:55.648903+08:00. Primary: native last/bbs Acc@0.50.

| System | Hits@.25 | Hits@.50 | Acc@.25% | Acc@.50% |
|---|---:|---:|---:|---:|
| protected_geometry_parent | 5616 | 4506 | 59.0660 | 47.3917 |
| native_g_control | 5615 | 4477 | 59.0555 | 47.0867 |
| final_quality | 5606 | 4460 | 58.9609 | 46.9079 |

Quality vs completed control: repairs 14, damages 31, net -17.

Quality vs protected parent: repairs 72, damages 118, net -46.

Quality same-frame R forward: repairs 72, damages 118, net -46.

All256 / one native score / same Query Box+Mask. Actual row order matched; B8/29778 once/3723 updates.
Only the fixed-weight quality term differs from the completed control; parents remain frozen/eval.
Independent terminal integrity review pending. This analysis is not reviewer PASS.
One new quality arm; reused completed native+G control. Single seed2027. 6887 pretrained-seen holdout separately from native9508. Fixed-frame head replay is a forward diagnostic, not independently trained ablation. Parent states exact/frozen; independent complete-forward tensors not claimed bitwise identical. GT coverage/volume groups offline only. No Nr/Sr results or novelty proof.
