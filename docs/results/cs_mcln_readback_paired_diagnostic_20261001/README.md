# R fixed-prediction paired diagnostic

This is a read-only diagnostic of the final R residual on one saved trained R checkpoint. A single model forward supplies all boxes, Masks and geometry evidence. The eval-mode final semantic head is replayed on the pre-R Query; every other output is shared. Actual native evaluator scores/ranking are captured directly. It records changed Query selections, hits, repairs/damages, first-qualified ranks, raw256 coverage and descriptive expression-length groups.

CPU fixture passed on 2026-10-01: real head/R modules, zero/nonzero readback, unchanged box/Mask objects and BatchNorm buffers, exact native score/rank capture, zero updates and no CUDA initialization. The fresh max-reasoning source review passed provisionally within the same model family. These are engineering evidence, not ScanRefer GPU or accuracy results.

The waiting controller was launched at 16:10 CST, PID220304. It waits until 2026-10-04 01:45 CST, then confirms the formal R job has exited successfully, checks all 21 complete epochs and the preset best rule, and checks GPU idle. It runs first12 validation sanity before full9508. If training remains alive, it waits 300 seconds. It uses frozen training modules from a separate diagnostic folder; active training files/configuration are unchanged. This diagnostic must finish before starting the common-numeric CS GPU control.

Current training is only published through R E6. No paired GPU result exists. Current R21 ETA is October4 01:43 CST; allow approximately 45–60 minutes after release for sanity and full validation, based on the previous complete CS replay. This is a schedule estimate, not a completion receipt.

Interpretation limit: bypassing R within a jointly trained checkpoint measures the direct final forward effect. It does not replace the common-numeric independently trained no-R control, prove training causality or prove validation coverage was preserved relative to E71. GT is used only offline for IoU/coverage; the raw256 oracle is not deployable.
