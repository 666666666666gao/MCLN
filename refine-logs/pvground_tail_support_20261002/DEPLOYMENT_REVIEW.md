**PV-Ground deployment review — PASS for preflight only**

No blocking issue was found in `deploy_preflight.py`. This code gate covers isolated source/spec preparation, strict CPU restores for both arms, and the raw arm's disposable preflight launch. It does **not** establish runtime readiness, passed GPU preflight, or permission to treat full training as passed.

Reviewed at 2026-10-02T16:57:29.854404+08:00. Actual reviewer: `gpt-6-astra`, reasoning `max`; `review_independence: same-family`; `acceptance_status: provisional`.

**Checks that passed**

- All round-2 reviewed references still match their recorded bytes. The deployment script checks those references and this deployment review before connecting. This report includes the concrete deployer and parent-spec snapshots; reviewed model/runner/controller files were not changed.
- Before the first remote mutation (`root.mkdir`, line 54), the embedded check requires actual current-P3 status `complete`, absent new root/source, no compute PIDs from `nvidia-smi`, and source-size-based free space. Uploads, source preparation and phase/spec writes follow that gate. The P3 run/source is never a write target.
- The new root and source are `/root/autodl-tmp/pvground_tail_support_20261002` and `/root/autodl-tmp/pvground_tail_support_source_20261002_v1/PV-Ground`. Preparation copies the original September17 G/D source into this separate directory. It does not reuse the P2 model source.
- Both arms retain the configured original G terminal under `mcln_pvground_scanrefer_finetune_20260918_semantic_assignment_v1`, SHA256 `0575dfae333dabdf288a470fc09d8967f9867d1853a5f61d9cfc3cbcf7964522`. They are passed through the reviewed runner's strict author/G CPU restore before launch. The script requires successful exit and the restore marker from both; these successes have not yet occurred in this review.
- All four inherited G helper files were inspected and their actual hashes match `parent_spec.json`. The root uploads are read back byte-for-byte, and the same seven `pvground_*` modules are copied into each phase directory. This matches both the runner's output-local hash checks and the shared root import path. Candidate, tail and preflight helper hash fields are supplied explicitly; generated source-port path/hash replaces the inherited P2 source descriptor.
- Four in-memory spec assemblies preserve the original G, runtime/interfaces, seed 2027, batch 8, one fit pass and LR/backbone LR 1e-5. The reviewed runner retains native weight-decay/clip and split/update checks. `p2` is explicitly false and `p3` true. The two arms use the same 14-channel tail architecture/order; their only spec differences are root, arm label, support flag and control-reference path.
- `tail_raw` uses the older G control for initial/order checks. `tail_fused` references the **new** `tail_raw`, and the reviewed controller requires its completed 9508-row formal receipt before later fused training. The old G control cannot substitute for the paired raw result.
- The sole background launch is `controller.py --arm tail_raw --preflight-only`. It runs the planned two-update disposable preflight and no full train/formal phase. Both CPU restores precede it. Fused GPU preflight and full training remain unlaunched. No deletion, process-kill, package installation or package-download action is present; the SFTP downloads are source/restore receipts.
- The helper obtains the SSH password from the existing environment, uses system host keys without automatic unknown-host acceptance, and quotes remote paths/arguments and environment values. No credential wrapper or environment values were read by the reviewer; no concrete credential-boundary defect was encountered.

**Local verification and limits**

Standard-library AST/compile checks passed for the deployment helper, embedded remote check and four inherited G helpers. The exact spec-update expressions were evaluated in memory; the future generated source-port hash alone was left pending. Static checks confirmed all four remote guards precede the first mkdir and all duplicate source/root/parent literals agree. All 24 round-2 snapshot references match.

No deployment code or Paramiko import was executed. No SSH/GPU, installation or implementation edit occurred. Actual remote P3/GPU/disk status, new source readback, CPU restores, raw/fused numerical preflights, memory measurements and full experiment results remain pending. The PASS applies to this concrete preflight-only deployment code.

PGTS-W001 remains nonblocking: the inherited `verify_native_replacement` path is unreachable from the current update calls. No fresh G reconstruction/gradient witness is claimed.

**Reviewed byte snapshots**

These follow the existing project trace convention and include the concrete deployment code consumed by the deployment gate.

- `C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\deploy_preflight.py` (7453 bytes): `09ef38ec50ccea7e64dcea847ce2e3a15fa208cfb9b34014399f44247730fa3b`
- `C:\Users\gb\.codex\tmp\pvground_p2_semantic_20261002\parent_spec.json` (3169 bytes): `3635eaefd44b13b5d5f4c8651f935f6dea06b4533dd0142dec08952d26079a4e`
- `C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\CODE_REVIEW_ROUND2.json` (14424 bytes): `f1dc6c57a73d5b30dfc6b7be2d80497183cb7db159f2e99161ba4c21324cf64d`
- `C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\CODE_REVIEW_ROUND2.md` (7128 bytes): `431e866f8a0d29c90cb85b71492d8f414e267fba836668d4ce474d7a6149a0cd`
- `C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\prepare_pvground_tail_support_source.py` (2218 bytes): `7487405bfd6865838720613c046a6c055d13f4a8d107cc5d26ea66b86e493f8e`
- `C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\run_pvground_tail_support.py` (38567 bytes): `483a7db4e0d11d8b79af63ea6e02f6d9455bed218190efc17d42e48929648346`
- `C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\run_pvground_tail_support_control.py` (4120 bytes): `21774197eb1ee5a9c4e7742953c1d496e75d6192c80c82de24eb6084398bdcb6`
- `C:\Users\gb\.codex_pvground_cs_20261002\scripts\pvground_candidate_box_refiner.py` (3767 bytes): `e971346230d0e0547139823c106e48f970c58a0b0ef26ef46b75beaa672f6212`
- `C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\pvground_tail_support_box_refiner.py` (3655 bytes): `665e94c150492a9fc3d52ddca7da2b7dca7d7ddbd981e759640f15846bb3f757`
- `C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\pvground_tail_preflight.py` (2898 bytes): `823f27d14e333a2fb8f15e6bd41a81abe3717c34a4adcb77aebbbb050ebcdbdb`
- `C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\PLAN.md` (4954 bytes): `5e8999f10ec2a3b6e69e71496f5edce863e3131912c33baefd002143f42973ff`
- `C:\Users\gb\.codex\tmp\pvground_p2_semantic_20261002\pvground_source_query.py` (3025 bytes): `e0fcc6a30402ebeba7a9ffa1be907515350b64e408602a8b47b702e83a5d05ab`
- `C:\Users\gb\.codex\tmp\pvground_p2_semantic_20261002\pvground_observation_query.py` (7683 bytes): `cc92906f9b6f388c8da702f555a099faf07d619fefa2772bfb229693ebae7742`
- `C:\Users\gb\.codex\tmp\pvground_p2_semantic_20261002\pvground_task_observation_query.py` (2770 bytes): `39349bf1fe8f3f51149f0872911020ecd0ede98fc9a8278cecb5ae466ed8643d`
- `C:\Users\gb\.codex\tmp\pvground_p2_semantic_20261002\pvground_semantic_assignment.py` (5065 bytes): `3bf9c5f50095e7e447af91e7f61a34c930bf04ec8590f7e4da54134b859cf773`
