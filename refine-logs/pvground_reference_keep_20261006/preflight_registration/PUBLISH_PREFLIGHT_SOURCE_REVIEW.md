# Doc89 preflight publication source review

- Reviewed at: 2026-10-07T00:20:33.666+08:00
- Execution scope: `SOURCE_ONLY`
- Verdict: **PASS**; blocking findings: **0**.
- Same established review context and model family; provisional. Requested Astra/max is not backend attestation. No independent-backend or runtime-success attestation.

This bounded administrative review covers `publish_preflight_authorized.py` and `prepare_preflight_publication.py`. It does not reopen the model/loss review. The primary SOURCE_REVIEW gate owns its final source identities at publication time. Primary source/spec/report files, active continuation state and future observations/results are deliberately excluded from this administrative hash gate.

The actual launch receipt records `2026-10-07T00:04:49.814251+08:00`, controller `776037`, `PREFLIGHT_LAUNCHED_NOT_PASSED`, two planned optimizer steps per arm, no saved weights and no accuracy result. Doc89 describes the launch, pending M0 checks and later conditional full fits. It does not report M0 success, new accuracy or completed effective-module evidence. M0 termination is not a prerequisite for this source/launch publication.

The unchanged goal order is ScanRefer on the same complete 9508 rows with at least 5620/4764 hits, three effective modules supported by direct ablations, then independent Sr3D/Nr3D training from the respective author checkpoints. Current 5598/4848 stays protected; seed 2027 only. Control/keep weights are 0/1. Primary source-review metadata reports PASS with zero blocks and the resolved CPU/CUDA matched-index issue; this review does not re-certify its 49 inputs.

Checks completed:

- The generator selects the established Doc87 publisher, replaces the Doc89 section, predecessor, payload prefix, control-spec path and launch/state/receipt labels consistently. No stale Doc86/Doc87 or old cleanup gate remains in the generated publisher. Neither script was executed or compiled in this review.
- The actual predecessor is `pvground_referit_mask_reference_20261006/comparison_publication.json`, section 20.376.88. All four local documents equal its SHA `b6026ddd87bc775aea060c37ffe51dc2561920ea9623a8ed04a6ca448a6735c4`; Doc88 occurs once and Doc89 is absent. Three current HEADs were read through local Git metadata and match the predecessor. No Git command or remote query was used.
- The publisher checks its own source report and the separate primary source report, then launch fields, predecessor/state relation, four document bytes, expected HEADs and clean worktrees before publication. Clean worktree status and active-state relation remain runtime checks; they were not claimed as executed here.
- The fixed payload is exactly 24 named UTF-8 source/document/JSON files plus `.gitattributes` (`** -text\n`). No glob, raw NPZ, weights, `.aris` or credential file is included. The actual existing named files were inspected only as text payload/schema where needed; no literal password assignment or private-key block was detected. Password retrieval uses the environment only for SSH and is not included in the serialized bundle.
- Remote code checks the old document SHA before writing and resolves only `/root/autodl-tmp/mcln_published_evidence_20261005/pvground_reference_keep_20261006/preflight_registration` beneath the fixed canonical project. It requires a new directory, validates each contained destination and uses exclusive file creation plus byte readback. No SSH was performed to attest remote state.
- `old + section.encode(UTF-8)` preserves the prior document bytes. Four local copies, the remote handoff and Git main follow the inherited publication sequence. Only owned document, manifest and fixed payload paths can be staged; staged payload bytes and the committed prior document prefix are checked. `.gitattributes` prevents text normalization of evidence payloads. Push is non-force and main is read back.
- The current sync guard contains the old Doc88 SHA exactly once. Runtime replacement is limited to that exact SHA. The final receipt is written after document equality and Git checks and records section 20.376.89, three heads, handoff SHA, payload count, `REFERENCE_KEEP_M0_LAUNCH_NOT_PASSED`, controller ID and false new-formal/raw-NPZ/weight-publication flags.
- The embedded remote Python contains no post-3.7 syntax; `shlex.join` stays in the local publisher. No source edits or new defensive mechanisms were introduced.

Execution boundary: local filesystem reads, hashes and requested report writes only. No commands, SSH/network, progress query, model construction, GPU work, deletion, publishing, training, new observations or future results. This PASS means the reviewed publication source has no identified blocking defect; it is not a publication receipt or M0 acceptance.

Stable inputs bound below. The JSON additionally binds this Markdown file; it does not self-bind its own JSON.

| Path | Bytes | SHA-256 |
| --- | ---: | --- |
| `C:/Users/gb/.codex/tmp/pvground_reference_keep_20261006/publish_preflight_authorized.py` | 12382 | `8df5c4dd549a24af6f24a2742ebc1cdfeb0aa9740e3452cf895340f15728b0aa` |
| `C:/Users/gb/.codex/tmp/pvground_reference_keep_20261006/prepare_preflight_publication.py` | 8098 | `4d783a4cb54e73f9e3f519f776dbe47fee4749b1f9208d1d5ac4dde916089ba2` |
| `C:/Users/gb/.codex/tmp/pvground_referit_mask_reference_20261006/comparison_publication.json` | 650 | `ec873742e26c88658c070be683d6f721d5cf1773956873728e2943abe84ad1e2` |
| `C:/Users/gb/.codex_mcln_g0_20260905/docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md` | 2348312 | `b6026ddd87bc775aea060c37ffe51dc2561920ea9623a8ed04a6ca448a6735c4` |
| `C:/Users/gb/.codex_pvground_cs_20261002/docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md` | 2348312 | `b6026ddd87bc775aea060c37ffe51dc2561920ea9623a8ed04a6ca448a6735c4` |
| `C:/Users/gb/.codex_mcln_v99_internal_20260928/docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md` | 2348312 | `b6026ddd87bc775aea060c37ffe51dc2561920ea9623a8ed04a6ca448a6735c4` |
| `C:/Users/gb/Desktop/document/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md` | 2348312 | `b6026ddd87bc775aea060c37ffe51dc2561920ea9623a8ed04a6ca448a6735c4` |
| `C:/Users/gb/.codex/tmp/sync_cs_handoff_remote_20260923.py` | 1435 | `fddced921c68e694862ea3fb76ef6e52c4b8ea01d1f86edf9e5c9be73239c045` |
| `C:/Users/gb/.codex/tmp/pvground_referit_mask_reference_20261006/publish_preflight_authorized.py` | 12866 | `18d9b0413064d9deb4a8412e0fe9c78e7ac6532aa513d3add8f0c8532c491078` |
| `C:/Users/gb/.codex/tmp/pvground_reference_keep_20261006/preflight_launch.json` | 891 | `2bd91dd4ca5b49fe84eba2da2d24e0e37e54414b916179ce879a0c367af73f7b` |
| `C:/Users/gb/.codex/tmp/pvground_reference_keep_20261006/resource_check.json` | 111 | `88e72c17f938ac1315dc3e0380687131a34199c25f4f48730f5c77c3cfca45d5` |
