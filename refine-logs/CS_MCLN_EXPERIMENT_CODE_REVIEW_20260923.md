# CS-MCLN experiment code review, 2026-09-23

Reviewer: fresh secondary Codex, `gpt-6-astra`, max reasoning.
Review independence: same-family. Acceptance: provisional static PASS; no GPU run was performed by reviewer.

Initial blocking findings were the permissive generic E71 checkpoint load and absence of a paired ScanRefer runner. The new dedicated runner verifies all retained E71 state names, shapes and dtypes; removes exactly nine SourceChoice tensors; allows only the three CS module state prefixes as new; uses a native-only scoring/evaluator contract and distinct new/core/backbone LR groups. The reviewer identified and we fixed the data-root trailing slash required by the existing RoBERTa path join, the final full-zero cosine epoch, and missing epoch-boundary resume.

Final review found no remaining blocking **code** defect. It confirmed that validation only contains ScanRefer, ScanNet ×10 is train-only, the native `last/bbs` Top-1 0.25/0.50 evaluator has 9508 rows, both arms use the same seed/order/budget, and the CS preflight compares zero-update outputs and checks output/internal gradients plus Mask-to-box dependency. The reviewer asked that preflight timing be labeled as diagnostic-inclusive; this was fixed.

Deployment and full training remain contingent on actual A100 preflight: real-input zero-update equivalence, two AdamW updates, capacity, gradients, and measured optimizer-checkpoint size versus available disk space. No accuracy claim follows from this static review.

User then requested retaining the best formal ScanRefer weight. The same reviewer rechecked the added per-epoch 9508-row evaluation, dual-threshold rank, one atomic model-only `best.pth`, and one atomic full-state `latest.pth`. Recovery now reads metrics from `best.pth`, handles a first-epoch best saved before latest, and refreshes the display JSON. Final re-review found no remaining blocking code issue. Actual disk capacity is still a preflight gate.
