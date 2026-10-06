# Doc90 fit publication source review

- Reviewed at: 2026-10-07T00:34:39.490+08:00
- Execution scope: `SOURCE_ONLY`
- Verdict: **PASS**; blocking findings: **0**.
- Same established review context and model family; provisional. Requested Astra/max is not backend attestation. No independent-backend or runtime-success attestation.

This bounded administrative review covers `prepare_fit_publication.py` and `publish_fit_authorized.py`. It checks their published claims against existing closed M0 receipts and their publication protocol against actual Doc89. It does not reopen NN/loss correctness or execute a launcher, observer or publisher. The separate FIT_SOURCE_REVIEW gate owns its final primary source identities. Its source/spec/report files and mutable active/progress state are not duplicated in this administrative hash gate.

Actual evidence and claim boundary:

- Existing M0 observer is closed, controller exit is 0, status is complete, completed arms are control/keep, finished at `2026-10-07T00:20:36.869134+08:00`. Both local per-arm receipts equal the copies embedded in `preflight_wait.json`: pass, 2 optimizer updates, 456102 geometry parameters / 10 states, 0 weight files, no accuracy result.
- Each M0 receipt records exact parent/R and native-data witnesses, neutral decode and keep loss 0 initially, qualified direct keep gradient greater than 0 on the second witness, exact Adam keys/groups/moments/steps after the existing in-memory serialization check, 39 actual empty-support fixture rows and all-256 raw-member extent witnesses. Doc90 preserves cross-process floating-point qualifications. These are existing receipt claims; this reviewer did not rerun them.
- Formal-launch files were initially expected future gates. During this review the parent reported actual successful launch and authorized reading their now-existing immutable receipts. `fit_launch.json` records controller 776929 at `2026-10-07T00:31:46.522037+08:00`, status `TWO_ARM_FIT_LAUNCHED_NOT_COMPLETED`, no accuracy result; its resources exactly match `fit_resource_check.json`.
- `fit_observer_started.json` records local PID 35624, first delay 24300 seconds, first observation `2026-10-07T07:16:46.522037+08:00` and later interval 240 seconds. The scheduled timestamp equals actual launch plus delay. No live progress or terminal result was queried.
- Doc90 records M0 PASS plus formal-fit LAUNCHED. It does not claim formal completion, new hit counts or three effective modules. Seed 2027, protected 5598/4848, required 5620/4764 on the same 9508 rows, then three direct-ablation-supported effective modules before independent author-initialized Sr3D/Nr3D training remain unchanged. The text grants no additional array/dataset deletion scope.

Publication checks:

- The generator consistently replaces the own-report name, separate FIT source gate, predecessor, fit receipt paths, Doc90 section/receipt, fit-registration prefix and fit observer/state fields. One inherited Doc88/M0-launch docstring was found and corrected in both generator and output; publisher executable body is byte-identical after newline normalization. No execution logic change was required.
- Predecessor is actual `preflight_publication.json`, section 20.376.89. All four local documents have SHA `0b209960abc9ee23d817c34144ca92f1d2cb5d7ca58a6d472743b33017838651`, each contains Doc89 once and no Doc90; three current HEADs equal the predecessor. HEADs were read through local Git metadata without Git commands.
- The publisher checks its own SOURCE_ONLY report and the separate FIT_SOURCE_REVIEW report (PASS/WARN, zero blocks and exact bound-file SHA), actual launch fields, closed M0 receipt, observer schedule, predecessor/state relation, four document bytes, expected HEADs and clean worktrees before any remote write. Missing required runtime files fail normal reads; none is fabricated.
- Payload is exactly 25 named text source/document/JSON files plus `.gitattributes`, total 26, under `refine-logs/pvground_reference_keep_20261006/fit_registration/`. Existing payload text and receipt schemas were inspected for publication scope. No weights, raw NPZ, `.aris`, privacy directory or credential file is selected. Password is read from an environment variable for SSH only and never serialized in the bundle.
- Remote code resolves only `/root/autodl-tmp/mcln_published_evidence_20261005/pvground_reference_keep_20261006/fit_registration` beneath the fixed project. It checks the old document SHA before writes, requires a new destination, checks containment, exclusively creates each payload, verifies byte readback and appends the document. It does not write into the running NN experiment source root. Remote existence/symlink state was not queried by this reviewer.
- `old + section.encode(UTF-8)` preserves the complete old document. Main/PV receive fixed evidence payloads plus their owned manifest entry; all three repositories and Desktop receive the same handoff. Only owned paths may be staged; payload bytes and old committed document prefix are checked. Evidence `.gitattributes` disables text normalization. Non-force main push and readback precede the final equality/receipt checks.
- The sync guard contains the old Doc89 SHA exactly once. Replacement targets that SHA only. The final receipt schema records section 20.376.90, heads, handoff SHA, four-local/remote equality, main head, payload count, `REFERENCE_KEEP_M0_PASSED_FORMAL_FIT_LAUNCHED_NOT_COMPLETE`, controller and false new-formal/raw-NPZ/weight flags. No publication receipt was created by this review.
- Embedded remote Python uses syntax compatible with the existing Python 3.7 environment by inspection; `shlex.join` runs locally. No compilation or package changes were performed.

Only local filesystem reads, hashes and these two report writes were performed. No command, SSH/network, GPU/model operation, progress query, deletion, publication or primary source edit. Old PUBLISH_PREFLIGHT_SOURCE_REVIEW JSON/Markdown bytes remain unchanged. This PASS is a bounded source verdict, not a completed-publication or formal-training-result attestation.

Stable reviewed input identities follow. The JSON additionally binds this Markdown file and deliberately does not self-bind its own JSON.

| Path | Bytes | SHA-256 |
| --- | ---: | --- |
| `C:/Users/gb/.codex/tmp/pvground_reference_keep_20261006/prepare_fit_publication.py` | 8473 | `f1d287b548d41d8a63fe5d3959c525c27e50a3d39b066aa81da97060443e6b22` |
| `C:/Users/gb/.codex/tmp/pvground_reference_keep_20261006/publish_fit_authorized.py` | 13101 | `24a76b3b6a26549e98c54b474d2c62eff1e57b17ca0afedc806aea6461f727ee` |
| `C:/Users/gb/.codex/tmp/pvground_reference_keep_20261006/preflight_wait.json` | 24415 | `1580fcb2aa03d9aca97e850834d6c2b4d6ec0cee121534d2bd7c6078ad6a71a8` |
| `C:/Users/gb/.codex/tmp/pvground_reference_keep_20261006/publish_preflight_authorized.py` | 12382 | `8df5c4dd549a24af6f24a2742ebc1cdfeb0aa9740e3452cf895340f15728b0aa` |
| `C:/Users/gb/.codex/tmp/pvground_reference_keep_20261006/preflight_complete/control/preflight.json` | 7570 | `f82b48cea02bf27df6e6958e49bf57c44d10cf1bd541d1b2f9cf668e710ad8e2` |
| `C:/Users/gb/.codex/tmp/pvground_reference_keep_20261006/preflight_complete/keep/preflight.json` | 7566 | `1e1415ae3d6ea358bf2937db636f27717eb01d4fd762be80db701c2d108c2ff7` |
| `C:/Users/gb/.codex/tmp/pvground_reference_keep_20261006/preflight_publication.json` | 647 | `ec6cbf8180b57033eb9528c5e895d870aa415e3d692dddc9ebe881260c9ecf18` |
| `C:/Users/gb/.codex_mcln_g0_20260905/docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md` | 2351640 | `0b209960abc9ee23d817c34144ca92f1d2cb5d7ca58a6d472743b33017838651` |
| `C:/Users/gb/.codex_pvground_cs_20261002/docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md` | 2351640 | `0b209960abc9ee23d817c34144ca92f1d2cb5d7ca58a6d472743b33017838651` |
| `C:/Users/gb/.codex_mcln_v99_internal_20260928/docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md` | 2351640 | `0b209960abc9ee23d817c34144ca92f1d2cb5d7ca58a6d472743b33017838651` |
| `C:/Users/gb/Desktop/document/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md` | 2351640 | `0b209960abc9ee23d817c34144ca92f1d2cb5d7ca58a6d472743b33017838651` |
| `C:/Users/gb/.codex/tmp/sync_cs_handoff_remote_20260923.py` | 1435 | `4d535a4a285649a18e6e4de58d79c2f87f29458533f11e6d9febf9de64b76b6d` |
| `C:/Users/gb/.codex/tmp/pvground_reference_keep_20261006/fit_launch.json` | 1876 | `4090e1f55a03a26a24e3b48d1a5e61d60344612bed644870f13ce5ee864e064c` |
| `C:/Users/gb/.codex/tmp/pvground_reference_keep_20261006/fit_resource_check.json` | 156 | `75a582198ab61363c198904e9debec107abf014d04fca556fee3eb8e3be46310` |
| `C:/Users/gb/.codex/tmp/pvground_reference_keep_20261006/fit_observer_started.json` | 340 | `551103a0822b737bd6cc51fdf9ee6d436d4a7bf40e04ff2ddc487f7da9218e5b` |
