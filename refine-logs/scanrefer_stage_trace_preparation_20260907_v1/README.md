# ScanRefer stage trace preparation

This directory contains CPU unit-test evidence for `scripts/trace_scanrefer_readout_stages.py`.
It contains no real scene trace, training result, or new REC measurement.

The helper consumes a completed `JointRecReadout` forward and the native query
selection from the original evaluator. It reuses the deployed Parent, Geometry,
and V99 policy functions and requires every final score to equal the recorded
runtime score, including validity masking. It separates the ungated V99 proposal
from the final Pareto decision. Query indices identify slots in that forward;
they do not establish persistent instance identity.

Six tests passed in the original `bdetr` environment with CUDA hidden. The tests
cover global query mapping, tie ordering, geometry validity, accepted and vetoed
proposals, exact score verification, and absence of input mutation or gradients.
The synthetic fixture is only an executable contract check.

`dependency_binding.json` binds the existing runtime sources. Eight files match
the canonical worktree byte for byte. `train_dist_mod.py` has a previously added
native model-factory option in the worktree; the three relevant runtime functions
match by Python AST. `runtime_diff.txt` records that complete difference.

The current corrected-mesh training and formal evaluation snapshots are unchanged.
No live hook, duplicate evaluation, or GPU job was installed by this preparation.
Actual stage measurements require a later isolated forward after the fixed run.
That runner must bind row/scene/root identity, input point hash, checkpoint and
data hashes, then compute GT metrics only after recording the GT-free choices.
The old `rows.json` and `native_rows.json` lack the intermediate choices needed
to reconstruct stage-level repairs and damage.
