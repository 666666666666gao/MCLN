# Source review R2: native MCLN two-update engineering check

Verdict: **WARN**, with **0 blocking findings** and one retained nonblocking note. The controller environment issue from R1 is **resolved**. This is the same reviewer continuing the source review after a minimal patch; the result remains **same-family / provisional**. Requested model/effort is `gpt-6-astra` / `max`; actual model, effort and backend identity remain **UNATTESTED**.

Only `check_controller.py` changed. The new code at lines 11-15 loads the same runtime `env_spec.json`, copies `os.environ`, applies `environment['env']`, prepends the admitted source and its `pointnet2` directory to `PYTHONPATH`, and sets `CUDA_VISIBLE_DEVICES=0`, `WORLD_SIZE=1`, `RANK=0`, `LOCAL_RANK=0`. The sole child `Popen` at line 23 receives `env=variables`. This is the completed transfer controller's environment setup for a GPU stage. The update does not change the child command, update count, waiting behavior, failure propagation or single-launch scope.

## Retained nonblocking note

`check_native_mcln_portable_training.py:93` still calls the native `TrainTester.get_datasets`. That factory constructs a train dataset and an unused validation dataset. The two batch-8 updates iterate only `Subset(train, range(16))`; no validation example enters model forward, loss, optimization, evaluation or tuning. Claims must distinguish **no validation sample use in these updates** from **zero validation-file access**. The latter is not true of this constructor. No change to this native construction was requested for R2.

## Checks and unchanged conclusions

- Compared the revised controller with the archived R1 controller. The diff contains only the `os` import, existing environment setup and `env=variables` argument.
- Checked every existing audited input digest: only the controller changed. The training check and launcher remain byte-identical. Rechecked all **736** entries in the existing source manifest; each size and digest still matches.
- Actual local Python **3.7.6** parsed and compiled the controller, training check, launcher and embedded remote body. The check confirmed one `Popen` receiving `env=variables`. None of these scripts was executed; neither PyTorch nor the native factory was imported.
- The R1 source conclusions remain applicable: checksum-bound E71 core plus exact A/B states; real native augmented training inputs and task labels; the integrated final boxes and Query Masks consumed by the native loss; all native trainable core parameters plus 57634 A/B parameters included once in AdamW; the original RoBERTa freeze; clip 0.1 and optimizer-before-scheduler order; and exactly two updates guarded by finite/nonzero gradients and changed A/B elements.
- The file-backed recovery still saves and reloads complete model, optimizer, scheduler and Python/NumPy/CPU-Torch/all-CUDA RNG state, with recursive value checks in the same process. The result still explicitly excludes cold restart, formal training completion and accuracy. It does not execute a subsequent update after the restore.
- The unchanged launcher still requires the prior full run complete with exit 0, its child absent, no GPU compute process, existing source-manifest equality, a new side output directory and the existing nonblocking GPU lock. It has one launch path and no retries.

The already reviewed Torch 1.10.2 optimizer, scheduler, RNG and clipping APIs remain unchanged. Their pinned upstream references are retained in `SOURCE_REVIEW.json`; this revision made no new web or remote request. The launcher is still intended for the parent's explicit local Python 3.11.14 interpreter; the remote child remains the existing Python 3.7 runtime.

The updated `SOURCE_REVIEW.json` retains 27 static absolute-path bindings and changes only the controller digest to `b91d806fbc42a675f15ba23037007fbe53ef5eba2116af9682c684dfd6346f6f`. No report, archive, observation or mutable progress file was added to `audited_input_hashes`. R1 remains in `reviewed_v1/` and its original trace and timestamped outputs; R2 has separate request/input/response metadata and timestamped reports.

No SSH, GPU query, neural forward/backward, optimizer update, training run, credential-content read or source modification by the reviewer occurred. This is a source gate only. Actual positive A/B task gradients, memory fit and successful saved-state restoration still require the engineering run's real result.
