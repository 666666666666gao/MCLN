# Conditional native Nr3D evaluation for the L1 endpoint

The evaluation entry point is implemented and CPU checked. It has not run a
GPU forward, read a trained L1 endpoint, or produced a new formal metric.
The active L1 v2 training files and registered comparison remain unchanged.

`scripts/run_nr3d_l1_native_formal_pair.py` can evaluate the registered
position-key endpoint only after `verify_terminal_run()` recomputes a passing
screen from the complete, hashed L1 terminal rows. The screen still requires
at least 10 net REC@0.25 hits against both the terminal text-key control and the
protected start, with no REC@0.50, either Mask-threshold or mean Mask IoU
regression. The shared function was extracted from the existing summary CLI;
the rule and CLI output are unchanged. These 6172 rows belong to scenes the
frozen backbone has already seen and are only an added-module holdout.

The future formal manifest must pin the actual training receipt, training
manifest, position artifact, protected parent, frozen source, evaluation data,
historical CLI/config and addon files. No eligible formal manifest or GPU job
has been created. The existing terminal reader verifies the saved position
mode, 6687 steps, parent SHA and artifact SHA before attaching a frozen matrix.

The runner uses the original `TrainTester.get_loaders()` and
`TrainTester._main_eval_branch()` with 7899 validation rows, B16 and 4 workers.
Only checkpoint/log/output destinations may differ from the historical config.
For each batch it evaluates the position intervention, removes the temporary
attachment and evaluates the protected model on the same batch. The original
native epoch consumes the returned protected outputs; the position outputs use
a separate instance of the same native `GroundingEvaluator`.

Earlier current-source native audits produced REC 4478 / 3763 hits, while the
protected historical deployment recorded 4475 / 3759. The new paired control
must be reported separately from that historical result; its gain cannot be
inflated by attributing the existing source/run difference to L1.

Both arms retain native size clamping, detector-overlap filtering, source-choice
scores and Mask selection. REC filtering and Mask query selection are distinct
existing paths. The saved row therefore records their separate Query IDs and
IoUs, the Mask IoU conditional on the REC query and on the legal box oracle,
and Top-16/32/64/256 coverage before and after filtering. Oracle values are
diagnostics and never enter forward scores or select a deployable result.

The runner checks identical earlier sampling indices and pre-intervention
decoder boxes, exact parent parameter/buffer identity before and after the
epoch, and unchanged addon weights. Each arm's reconstructed row totals must
equal its native evaluator's REC hits, Mask hits and summed Mask IoU. It then
reports paired fixes/breaks and scene bootstrap intervals, without applying
the module-screen threshold as a new formal-benchmark rule. The script never
replaces a protected artifact or marks a result promoted.

On the server's actual Python 3.7.11 / PyTorch 1.10.2 environment, all 11 relevant
CPU tests passed in 3.65 s. They cover the unchanged screen, complete synthetic
receipt validation against both controls, changed-row rejection, and parity
with the actual frozen native evaluator. The native examples specifically
exercise distinct REC/Mask queries, no legal REC candidate, and a metric
accidentally attributed to the other arm. All examples are synthetic;
CUDA_VISIBLE_DEVICES was empty. Python 3.7 compilation passed for the deployed
test files and helpers. This is not an end-to-end GPU evaluation test.

Evidence is in `refine-logs/l1_native_formal_pair_cpu_20260905_v1/`.
The separate CPU directory is
`/root/autodl-tmp/mcln_l1_native_formal_pair_cpu_20260905_v1`.
The runner SHA256 is
`11a7bd0d6cc38be67d02a0f80d22deebe6dc7eb7db190372cf185fc484c7afc7`.

Next: let L1 finish its fixed endpoint and both terminal arms, verify the
receipt and recompute the screen. A passing screen permits an isolated native
formal pair; a failed screen seals this L1 run without a control promotion,
earlier-checkpoint selection or parameter sweep. Native formal behavior and
the paired accuracy result remain unverified until that conditional run.
