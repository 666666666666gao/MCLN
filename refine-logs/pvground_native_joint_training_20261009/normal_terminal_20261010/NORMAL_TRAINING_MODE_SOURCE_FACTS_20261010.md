# Normal training mode source facts

Scope: local published-source inspection only. No current-server training query, model execution, runtime mode capture, experiment change, or fix.

- `source/models/pv_ground.py:174` loads RoBERTa; lines 175–176 disable its parameter gradients.
- `source/main_utils.py:439` calls `model.train()` at the start of each native training epoch. No `train()` override or `text_encoder.eval()` call was found in the 14-file overlay.
- The text encoder is called in the model forward at `source/models/pv_ground.py:304`. The trainable text projector also contains Dropout(0.1).
- `source/models/pv_ground.py:621` sets BatchNorm1d/2d momentum to 0.1. Native training mode permits running-statistic updates; evaluation switches the whole model to eval at `source/train_dist_mod.py:198`.

Thus "frozen RoBERTa" in the current protocol describes its parameters. This source inspection does not establish fixed text features during training, actual per-layer runtime flags, or measured E0/E2 normalization-buffer differences. These are native behaviors, not a newly identified implementation bug.

The observed accuracy regression is the actual full-9508 E0→E1→E2 result, not a synthetic unit-test failure. Its exact feedback loop is the existing native training/full validation command; it is not a seconds-long minimal precision repro. The diagnosing-bugs phases for causal hypothesis testing, instrumentation, fixes and regression tests are deferred: this step records source facts only, does not diagnose the cause, and does not pretend to have a fast precision test. Active training remains unchanged. After terminal results, inspect actual saved states before attributing changes to normalization or text stochasticity; C-off remains the first prepared training control.

No relevant CONTEXT.md or ADR was found for the overlay. Historical candidate-headroom audit JSON files were not treated as architecture decisions.
