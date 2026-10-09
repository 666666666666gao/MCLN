SOURCE_ONLY engineering source review — native_joint_v2 R1

Verdict: FAIL. Two concrete source-admission gaps remain. No reviewed program, SSH, Torch/model construction, GPU, checkpoint load, or training was executed by this reviewer. Earlier actual CPU evidence is read as evidence only.

Bv2.1 — engineering_controller.py:41 uses receipt.get('new_optimizer_steps', 2). The extremal helper emits actual_updates, so this controller assertion checks a default 2 instead of the actual field. Use explicit new_optimizer_steps==0 for the recovery stage and actual_updates==2 for the extremal stage. The helper itself still enforces two normal-trainer updates; this finding does not allege fewer actual updates.

Bv2.2 — launch_engineering_authorized.py:25–27 does not bind NATIVE_SOURCE_PORT.json or VSA_CONFIGURATION_WITNESS.json to audited_input_hashes. Those current files select remote source hashes and CPU eligibility. Add both to the existing local review hash loop. The current inputs are consistent; this finding is a missing launch identity contract, not observed drift.

Wv2.1 — native_joint_preflight.py:242 emits initial_span_zero=False, but the initial checks prove a finite bounded gate and correct mixing formula, not nonzero predicted gates. Record retained-checkpoint initialization and exact identity instead. No extra gate-nonzero acceptance requirement is needed.

The whole helper creates a new tester log directory before assigning the old checkpoint, reads the SHA-bound 841675936-byte checkpoint, constructs the original architecture and strictly restores model, Adam, scheduler and all four RNG families. It has no dataset iteration, forward, training update or new save path. Its restoration assertion tail and the extremal tail are identical to the previously reviewed complete recovery assertions.

The extremal helper preserves ordinary train_one_epoch for two real native batches and uses the initializer's SHA/mode/support-parent/strict-load checks for retained f9898c5dc7a5e97bf152421efc9072c23ef255618c64477a4bfacf996f2b3b19. It records diagnostics before acceptance, retains per-step core/backbone/support gradients, and requires both span output and internal aggregate gradients plus actual parameter updates. The initial formula corresponds to the existing model. Full checkpoint/optimizer/scheduler/four-RNG recovery is unchanged. No engineering state is a formal-training input.

Only train_dist_mod.py differs among 14 computational modules: import copy and per-construction deepcopy. Both specs match the deployed port. The actual CPU receipt is exit 0 with shared VSA layout counts 126→186 and copied layouts 126/126, CUDA uninitialized, zero forwards/optimizer steps. It does not establish full-model GPU recovery.

The launcher uses a fresh independent engineering directory, verifies the current 116 port hashes and four parent checkpoint hashes, checks one idle A100 and 846014332+67108864 free bytes, then starts the serial controller under flock --no-fork using a direct Popen PID. A successful process handoff is not M0 completion. Remote/witness syntax was parsed for Python 3.7 without imports or execution. No normal-training or cleanup stage is present.

Pending: actual GPU recovery of the old whole checkpoint; actual retained-head extremal initialization, two updates, gradients, full save/recovery, and independently audited receipts. Serialized state equality does not claim mid-epoch sampler reproduction or accuracy. Initialization/budget change remains separate from the configuration defect.

Attribution: original audit task was fresh; this is the same-task continuation, not a new fresh review. Requested gpt-6-astra/max; actual model/effort/route UNATTESTED, same-family/provisional.

All input identities are full Path.resolve() absolute paths mapped to bare SHA256 in SOURCE_ENGINEERING_REVIEW.json. Original failures and prior seals are unchanged. Native tool receipts, reviewer tool errors/corrections, source diff, static verification and raw input snapshots are preserved in RAW_ENGINEERING_REVIEW_R1.
