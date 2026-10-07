# Transfer resume source review

**Verdict: PASS — 0 blocking findings.** Scope: `SOURCE_ONLY`. Reviewed at 2026-10-07 22:32:06 +08:00.

The minimal resume change implements the requested continuation. Existing local files are skipped only after both byte count and SHA256 match. Missing files use `prefetch(file_size=...)`, are verified, and are then written locally. Unequal existing files stop collection without overwrite.

The remote inventory code is exactly unchanged from the original collector: 14 fixed artifacts and exactly 191 NPZ slices, for 205 files. The completed diagnosis, exit-zero and zero-update/zero-weight checks remain. The supplied receipt preserves the original interpretation: targeted development error cases, without a new accuracy or training claim.

Before SSH, the resume requires the original collector's closure record and the measured slow-transfer reason. **Actual closure remains a launch prerequisite:** this reviewer did not observe or stop the running collector, and `transfer_stop.json` was outside the specified five-file review scope. The executor must close the original collector before recording that fact and starting the resume.

The SSH target, system host-key verification and quoted remote command are unchanged. The embedded remote program performs stdlib inventory/hash reads; SFTP opens files read-only. The collected source files are copied as data. No NN replay, model/weight load, training, weight transfer, deletion or remote source edit is introduced. Final `INTAKE.json` is written only after the entire artifact loop succeeds.

Validation: both local scripts and both embedded remote programs pass Python 3.8 AST parsing; all three supplied JSON files parse. No target script was executed. No SSH, transfer, credentials/AUTH inspection, NN operation or deletion occurred. Prefetch throughput and completed collection are not attested by this source review.

| Reviewed file | Bytes | SHA256 |
|---|---:|---|
| `C:\Users\gb\.codex\tmp\pvground_support_boundary_cases_20261007\resume_closed_collection_authorized.py` | 3543 | `fc8e9b1755e291ee57716572852201f47170aabf713c0db0eeae7a58c6031b97` |
| `C:\Users\gb\.codex\tmp\pvground_support_boundary_cases_20261007\collect_closed_authorized.py` | 3002 | `f50316c274b8a637de806fda67cb947b30ed75e004ac5e36c981adb47a09522f` |
| `C:\Users\gb\.codex\tmp\pvground_support_boundary_cases_20261007\wait.json` | 563 | `252195d103d433fec20db75a893a1d42bc12b7ab8e0292503cd5b2f031b3a559` |
| `C:\Users\gb\.codex\tmp\pvground_support_boundary_cases_20261007\diagnostic_spec.json` | 3740 | `28aed20f5eb182960d861c53ef299356d79e668ea1863e81f1f5a718a8817479` |
| `C:\Users\gb\.codex\tmp\pvground_support_boundary_cases_20261007\complete\receipt.json` | 644 | `29c5a102032d36cd7929769f9bc7437a4590f0ffc309c80262e6b19d5af772ae` |

Attribution: fresh delegated Codex reviewer `/root/pvg_transfer_resume_source_review`; `review_independence: same-family`, `acceptance_status: provisional`. Requested model/effort: `gpt-6-astra` / `max`. Actual model, effort and backend: `UNATTESTED`; no cross-family acceptance is claimed.
