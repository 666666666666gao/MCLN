# MCLN Nr3D 视角增强修复审计 Tracker

Updated: 2026-09-05

执行合同：master §20.30（修订 eval-only 起点；固定 split/门不变）。

| Run | Purpose | Status | Evidence / next action |
|---|---|---|---|
| VA0 | 原实例输入、两源码快照、零更新 smoke | INPUT FOUND / PREPARING SMOKE | SHA 76aa6c...6edba1 精确匹配；没有 optimizer/scheduler，已在新指标前修订为匹配的新 AdamW |
| VA1 | old 完整一轮 fit + heldout | NOT STARTED | 25,768 fit rows / 1611 steps / 7,151 heldout rows |
| VA2 | fixed 同一合同 | NOT STARTED | 串行跟随 VA1，实际 row order 必须完全相同 |
| VA3 | 一次性机械决策 | PENDING | Overall .25 正、.50 非负、view-dependent .25 正 |
| G1 | Candidate-Edge Direct Scorer | WAITING G0 | 未部署或训练；未提交草稿保留在独立旧 worktree |

正式 7,899-row evaluation=0；持久生成模型权重=0；ScanRefer/Sr3D 受保护模型不变。
