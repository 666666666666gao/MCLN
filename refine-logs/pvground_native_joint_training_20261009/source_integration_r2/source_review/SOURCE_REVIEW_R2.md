# Native PV normal joint training — source review R2

**Overall verdict: WARN. Correction verdict: PASS. B1 is closed at source level; no confirmed source blockers remain in this revision.** Runtime admission and actual M0 remain pending.

This is a continuation by the **same original fresh review task** `/root/pvg_native_joint_source_20261009`, not a newly independent review. Execution scope is **SOURCE_ONLY**. Requested route remains `gpt-6-astra / max`; actual model/effort remain **UNATTESTED**, with **same-family / provisional** attribution.

## R2 change and B1 resolution

All **14 computational source files** were independently enumerated and SHA-256 checked against the sealed R1 input manifest. **Only `source/main_utils.py` changed**:

- R1 SHA-256: `62245d57455a2c175f91358030cf5befb0293d28b36cfe7d65bfd66ca5d3a10c`
- R2 SHA-256: `b1706a8c63923c2b8fc470155081615930ee446dd3c465055fdfc3485b308f4a`

The independently generated diff exactly matches `RESUME_CORRECTION_R2.diff`. All 14 current hashes match `RESUME_CORRECTION_R2.json`; its previous-receipt and previous-review hashes also reconcile. All R1 sealed artifacts remain unchanged. Among the 42 original R1 input paths, the only changed path is the authorized `source/main_utils.py`.

`source/main_utils.py:188–190` now handles training resume by assigning the absolute checkpoint directory to `args.log_dir` and requiring its retained `best.pth` to exist. Therefore, when later metrics do not improve, the unchanged final return at line 386 resolves to the actual already-retained best file in the same directory. Latest checkpoint writes also stay there. The code does not relabel the resumed latest weights as the best model.

For a fresh run or evaluation, the previous timestamp-directory construction and creation remain exactly the same in the `else` branch (lines 191–198). AST comparison confirms the entire module is otherwise unchanged, including full model/optimizer/scheduler/RNG restoration, metric ordering, update order, and best/latest save logic.

This closes R1's concrete missing-best-path control-flow defect. The existence guard and control-flow inspection are **not** an actual checkpoint load or a successful training-resume experiment. No model or checkpoint was opened.

## Normal-training plan and initialization declarations

`NORMAL_TRAINING_PLAN.md` and `NORMAL_TRAINING_DIRECTION.json` consistently describe the intended original trainable core plus registered Support/Span modules, forward-produced final boxes, unified native criterion, dynamic current matching, native all-256 last/bbs ranking, and frozen RoBERTa policy. They explicitly state that hard Mask/member references remain discrete, normal joint training has not started, and effectiveness is unproven. The source bodies implementing those paths are unchanged from R1.

The proposed three-epoch, single-seed, single-rank recipe and proposed learning rates are a plan; no loader counts, scheduler realization, GPU admission or resulting accuracy are established by these documents. The direction JSON's `R1_SEAL_PENDING_RESUME_DIRECTORY_DEFECT_FOUND` status is its historical recording state, not the current R2 verdict. `METRIC_PREPARATION.json` likewise remains the preserved pre-correction receipt; `RESUME_CORRECTION_R2.json` is the current 14-file receipt.

The additionally supplied source-only initialization records were read and bound:

- `init_manifests/whole_support.json`
- `init_manifests/extremal_support.json`
- `INIT_MANIFEST_PREPARATION.json`
- `prepare_native_init_manifests.py`

Their declared official/G/support SHA identities, G/support paths, parent-retention decision, environment-spec hash and provenance-spec hash match the still-unchanged pair specification sealed in R1. The two manifests differ only in `span_source_mode`. Both use ScanRefer/seed 2027, enable G and selected-mask supervision, and set `span_checkpoint` and its hash explicitly to null.

Thus the earlier absence of a concrete initialization declaration is resolved. The current manifests request a fresh Span output initialized to zero, consistent with the unchanged source constructor. They do **not** establish actual remote file existence, actual checkpoint bytes, G delta's exact expected key set, actual constructor counts, numerical initial equivalence, or successful parameter loading. The optional pretrained-Span lineage branch is not exercised by these null-Span declarations.

## Remaining conditions

R1's substantive source findings carry forward by exact hashes, with the corrected B1 exception and the new initialization declarations above. These remain pending evidence requirements, not newly observed code defects:

- Exact G delta key-set reconciliation and actual parent payload/lineage validation.
- Runtime source overlay, data/evaluator/YAML import binding and warm Torch/compiled-dependency admission.
- One-rank launch binding; actual train/validation expression and batch counts and point/superpoint ordering.
- Actual constructor/trainable counts, initial outputs, native-plus-additions losses and gradients.
- Actual complete checkpoint serialization/restoration, epoch-boundary RNG behavior, memory, throughput and storage capacity.
- Actual M0, normal joint training, full9508 results and direct evidence for three effective contributions.

No SSH, Torch or model import, pickle/checkpoint loading, GPU operation, installation, credential read, training, source edit, or weight update was performed. The frozen-pair source/specification was not changed. This review grants neither training success nor M0 approval.

## Evidence

`R2_STATIC_VERIFICATION.json` records **19 passing byte/hash/AST/declaration checks**. `INPUT_MANIFEST_R2.json` binds the current sources and requested review/declaration artifacts; exact bytes are in `r2_snapshots`. All 14 computational sources parse under the existing Python 3.10 interpreter using stdlib only.

`R2_INDEPENDENT_MAIN_UTILS.diff` records the independently recomputed change. `RAW_RESPONSE_R2.md` contains this full response. `R2_NATIVE_TOOL_RECEIPTS.json` contains the actual tool receipts for this follow-up, including the initial terminal encoding artifacts and the subsequent correctly decoded stdlib read. No R2 native command failed. `SOURCE_REVIEW_R2.seal.json` seals this review and binds the preserved R1 seal.

**All actual runtime, resume, training, accuracy, M0 and three-effective-contributions flags remain false.**

