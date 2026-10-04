# Revised isolated source; full factory/runner gate pending

The prior source-review PASS binds the old draft bytes only. This revision has
not been source-reviewed or executed in a native model. The active face fit and
its sources remain unchanged; its actual formal outcome will select the provider.

The unused bbs helper now follows the native per-map reduction/addition order.
No actual score or ranking mismatch was previously measured. The future real
batch check compares all256 scores, ranks and logit gradients to native per-row
reductions. No second score is introduced for deployment.

An explicit evidence-visibility control zeros the44-D geometry input before its
encoder. Both arms keep the same parameter layout, full encoded text, candidate
Query, face embeddings and native semantic head. It changes neither geometry
outputs nor native Mask responsibility, and retains all256 candidates. This is
a planned same-provider, same-start/budget structural control, not a no-readback
training ablation or quality-loss experiment.

readback_preflight_checks.py prepares actual-model checks for semantic call order,
unchanged geometry/Mask/contrast outputs, native bbs equality, and the isolated
native ScanRefer CE+G route before and after the first output update. It is not
a constructed factory, an actual GPU preflight, optimizer/serialization proof,
or accuracy result. The full future runner must also perform zero-residual
equality, frozen state checks and exact model/optimizer save/restore, then pass
a fresh full source gate before GPU execution. No Nr3D/Sr3D port is approved.
