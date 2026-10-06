# Approved archived-array cleanup source review

**SOURCE_ONLY PASS; zero unresolved blocking findings.** Reviewed 2026-10-06T22:34:21.707+08:00. Same established context, same-family/provisional; backend not attested. Requested gpt-6-astra/max routing does not attest the actual backend.

Source: remove_approved_archived_arrays_authorized.py. SHA256 ff9e4f0ebfd2299b36fdf1b68be9d5e59c2c993bf18ac4dbbcb2367ce97d56c4. No code change was needed or made.

The real approval says “允许，仅删除这4756份已保留本地的NPZ” and binds the preview SHA 703bf767856c7d6ddebc365961df923cc19a7a2714f4c66c8095101cf95f89b0. The preview contains exactly 4756 distinct paths and 571823151 bytes. Every entry is a batch NPZ under one of native_reference/fused_mask_reference × initial_formal/formal, and every name/size/SHA matches both the preserved Scan intake and actual terminal-audit digest index. This review checked those existing metadata identities; it did not reread binary arrays or repeat the experiment audit.

The local source checks approval/scope/preview SHA, its own SOURCE_ONLY report and bound input hashes, and then every local archived file's actual size and SHA before making its remote connection. The embedded remote code fixes the root to /root/autodl-tmp/pvground_mask_reference_20261006, requires the old actual exit0/complete markers, and checks the selected best SHA before deletion. All 4756 remote files must pass path resolution, fixed allowed arm/stage, size, SHA and uniqueness checks before the sole unlink loop begins. There is no wildcard or recursive removal, fallback, exception recovery, weight load or training command.

Deletion is confined to the approved remote copies. Best fused initial.pth is outside the targets and is hashed again afterward. Its SHA 2301e90a9391af4e4cd56edae09ce49db1cc8f67463e1c2568352ee2dafaff61 agrees with the actual step0 restoration and best-only weight-retention receipts. Local arrays, text records, official PV, G and V99 do not enter the deletion list. The existing Scan controller/observer are closed in the bound actual receipt; no new remote process query was made by this reviewer.

The output receipt schema distinguishes logical bytes from observed disk free space and records local_files_deleted=0, weights_deleted=0, text_records_deleted=0 and best_weight_preserved=true. Reading or creating that future receipt is not part of this review. A SOURCE_ONLY PASS is not an executed cleanup result.

Only local file reads, JSON parsing, metadata comparisons and SHA checks were performed. No command, SSH/network request, Python execution, model/weight load, GPU work, deletion or publication occurred. Embedded Python uses 3.7-compatible syntax by source inspection; shlex.join remains local. AST/compile execution is not claimed. No mutable primary ReferIt preflight source/spec/review or pending launch/cleanup result is bound in this report. Only the new review files, timestamped copies and local review manifest are written.

Exact input paths, sizes and hashes are in the JSON. That report's own digest is supplied separately; this Markdown is bound in reviewed_files.
