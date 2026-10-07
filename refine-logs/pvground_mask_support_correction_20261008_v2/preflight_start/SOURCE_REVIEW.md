# SOURCE_ONLY rescue review: PASS

Generated: 2026-10-08T03:01:03.824793+08:00

Same-context follow-up under experiment-bridge RESCUE_ON_FAILURE; same-family/provisional. Actual backend/model/effort remain UNATTESTED.0blocking and0nonblocking findings for the v2 import repair and M0-only launch preparation.

The original M0 is still **FAILED**, exit1 at2026-10-08T02:23:14.613804+08:00. Its archived traceback stops at the external `torch_scatter` import before model construction/forward/optimizer work. The earlier source review missed this dependency mismatch; its SOURCE_ONLY PASS did not validate imports and does not erase the real failure. The log, closed wait record and their bound failure manifest remain preserved.

V2 replaces that import with `from models.losses import dice_loss, sigmoid_focal_loss, scatter_mean`. The actual ported criterion exports `scatter_mean` from its existing `utils.scatter_util` at line23. Both actual source digests match the98-file port. The separately generated deterministic variant is not selected. The loss body is AST-identical after excluding imports; other7core files and thePVoverlay are byte-identical. There is no package installation, fallback, environment rebuild, new loss, target change or optimizer change.

The prior same-native-forward cached integration witness, originalGT/Hungarian responsibility,5/1/10/2 coefficients, equal heads and complete evaluation order remain intact. The new root/model/source-port paths and all bindings agree. The local warm ledger requires existing runtime reuse and real forward/backward, capacity and restore checks before formal training.

The reviewed deployer launches onlyv2M0 under the existing lock and includes `INITIAL_IMPORT_FAILURE.json`. Its sole observer uses first720seconds/later240seconds. `launch_fit_authorized.py` is excluded from this acceptance and from the deployment payload. OldDoc98 publication repair is outside this review.

Actual validation:13M0-scoped Python files plus2embedded remote programs parsed with Python3.7 grammar;8new-runner and17existing-helper bindings checked;62reviewed paths/bytes/SHA256values recorded in `SOURCE_REVIEW.json`. Original57-binding FAIL and53-binding supplement records are retained. No old6887checker rerun, SSH/network/torch/GPU/model/checkpoint/NN/source-edit action occurred.

V2runtime import, real2-step M0, gradients, frozen state, capacity, CPU rebuild, GPU head/Adam restore and integrated native equality are still NOTRUN here. Source PASS supports attempting the real M0 only; it is not a runtime or performance PASS.
