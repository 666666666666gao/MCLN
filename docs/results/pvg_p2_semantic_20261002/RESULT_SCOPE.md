# Result scope and post-review disclosure

The original analysis inputs were preserved unchanged for the fresh audit. This note records its additional disclosure; it is not a new model evaluation.

- Initial REC matches G exactly, row by row, in both bbs and bbf.
- Initial Mask floats do not all match. Semantic-only P2 differs from G at training row IDs2058 and16804. Historical joint P2 differs from G at16804 and18482. At the additional row2058, G/joint Mask IoU is0.9480887055397034 and semantic-only is0.9277961254119873. The cause is unexplained; do not label it rounding error or claim complete output identity.
- The reviewer independently recounts selected-box IoU and all reported paired/coverage/transition aggregates. Full-256 coverage is aggregated from the saved oracle flags and verified executed code; this artifact does not contain every candidate box for a separate reconstruction of all256 IoUs.
- The terminal checkpoint hash agrees with the saved training receipt. Successful formal strict loading is evidenced by the runner, formal exit and result receipt. No local terminal tensor/Adam inspection was performed; load.json was written before the terminal load.
- Audit verdict isWARN, same-family/provisional. The warnings also cover dormant helpers and the one-seed development scope. No new Nr3D/Sr3D result is available.
- Original G5615/4495 remains the strong start. Joint P2 and semantic-only P2 are negative controls under the tested budget, not promoted pretrained starts. P3 remains a design draft, with no implementation, preflight or training yet.

See EXPERIMENT_AUDIT.md and EXPERIMENT_AUDIT.json for exact file:line evidence and all audited hashes. The analysis REPORT.md and SUMMARY.json remain exactly as audited.
