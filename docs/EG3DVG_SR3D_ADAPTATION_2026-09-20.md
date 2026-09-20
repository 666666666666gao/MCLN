# EG-3DVG Sr3D conditional native adaptation

Recorded 2026-09-20 19:09 CST. This is a queued protocol, not a completed model result.

## Entry conditions fixed before outcomes

The live controller first waits for the fixed Nr3D adaptation endpoint and its independent evaluation audit. If either Nr3D REC count is below 4726/4059 of 7899, it records `deferred_for_nr_priority` and exits without Sr3D training. This leaves the next training allocation available for the unmet Nr3D objective.

If Nr3D meets both counts, the controller waits for the original ScanRefer checkpoint's zero-update Sr3D evaluation and audit. If that evaluation already reaches both protected Sr3D counts, 12139/10335 of 17726, it records `skipped_zero_update_already_meets_protected` and does not train. Otherwise it records `run_fixed_adaptation` and proceeds to the GPU preflight.

An exit code of zero for either skip is not evidence that training ran. Read `decision.json`, the fit receipt, and the evaluation audit separately. The already queued Sr3D zero-update evaluation remains independent of this training decision.

## Fixed experiment

- Complete author EG-3DVG, initialized from the released ScanRefer epoch69 checkpoint, not the Nr3D endpoint. Parent SHA256: `785f46ca595aed3a06efdfc46dc1c4b0493b212ccb66062a76f519324fd24f7e`.
- Author Sr3D train annotations: 65846 expressions in 1018 scenes. Joint ScanNet detection: 1199 samples repeated ten times. Total: 77836 rows, exactly once.
- Seed2027, batch8, final batch4 retained, 9730 updates, saved fixed order. AdamW, main LR1e-4, backbone LR1e-5, weight decay0.0005, gradient clip0.1, constant LR. Preserve author parameter trainability and all native losses, including negative-expression forward and Mask losses.
- Two-batch GPU training preflight must pass; discard its weights and reload the original author checkpoint for the fixed fit. No intermediate checkpoint selection, seed/LR search, or added C/D/G modules.
- Preserve original Sr3D relation-based augmentation. Do not transplant the Nr3D view-word repair. The only model training-interface repair exports the already computed `super_xyz_list` required by the unchanged native loss.
- Reuse the verified Torch1.12+cu116 environment. Save a single replaceable recovery checkpoint every512 steps and at the fixed endpoint, with optimizer/RNG state; this is not validation-based selection.
- Endpoint evaluation: 8-row execution preflight, then all17726 validation expressions, then independent CPU reconstruction and paired comparison with the zero-update Sr3D run. Primary output is author last/bbs. Same GT scene object boxes, predicted classes, native averaged regression/predicted-mask box, and native object-support score multiplication as the zero-update control. Diagnostic heads do not replace the primary output.
- Only REC Acc@.25/.50 determines acceptance. Mask diagnostics and native Mask computation remain. This protocol is not an object-box-free single-stage setting or a reproduction using an author Sr3D-specific weight.

## Evidence and execution state

Root: `/root/autodl-tmp/mcln_eg3dvg_sr3d_adapt_20260920_v1`.

Controller6128 launched at19:06:58 CST and was alive at19:09:05. At that observation it was waiting for the fixed entry conditions: no Sr3D GPU preflight, training update, or new REC result.

Spec SHA256: `89fc730fc2a1cf18e1970ae87d2faa142d799171f117f7bb2da4175a82578011`.

Training Dataset SHA256: `c45cd27aac71c8e65ba0713a0332625670423220d9ff5544df0e8f6752663699`.

Model source SHA256: `b65e77b9e3dd095591e37cbd88f8a6c3179b3499f4ed027569f9a33870438ee1`.

The actual controller's decision prefix passed five isolated CPU fixture cases: Nr below either threshold defers; Nr at both thresholds plus Sr at both protected counts skips; Sr one hit below either count proceeds. These synthetic checks execute no model and provide no accuracy evidence. See `refine-logs/eg3dvg_sr3d_adapt_20260920_v1/entry_policy_checks.json`; controller SHA256 `b1269e0a3e3729283315224b0f7392c0ba8c7b22da70c0c48ce6438798ef71ec`.

At19:09:05 the active Nr3D job had logged704/5614 steps, with finite loss and gradients. Estimated Nr endpoint plus evaluation: approximately23:40–23:45 CST, subject to measured runtime. Sr3D zero-update evaluation takes an estimated55–65 minutes afterward. If the conditional Sr3D fit is needed, its9730 updates plus full evaluation are approximately another9 hours; it is not currently running.

The three-dataset objective remains incomplete. Existing V99 results and all author-baseline results remain separately attributed.
