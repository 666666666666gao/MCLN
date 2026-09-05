# MCLN Nr3D 视角增强修复审计 Tracker

Updated: 2026-09-05 10:52 CST

执行合同：master §20.30（修订 eval-only 起点；固定 split/门不变）。

| Run | Purpose | Status | Evidence / next action |
|---|---|---|---|
| VA0 | 原实例输入、两源码快照、零更新 smoke | PASS | 两角色零更新 forward/loss/backward PASS；inputs_v3 唯一差异为5213822，EOL纠错/AST证明见master20.32 |
| VA1 | old 完整一轮 fit + heldout | RUNNING: OPTIMIZER UPDATES | 10:18:51 实测266/1611步；fit 25,768 rows，heldout 7,151 rows；11:15附近检查 |
| VA2 | fixed 同一合同 | QUEUED AFTER VA1 | 串行跟随 VA1，实际 row order 必须完全相同 |
| VA3 | 一次性机械决策 | PENDING | Overall .25 正、.50 非负、view-dependent .25 正 |
| G1 | Candidate-Edge Direct Scorer | WAITING G0 | 未部署或训练；未提交草稿保留在独立旧 worktree |

正式 7,899-row evaluation=0；持久生成模型权重=0；ScanRefer/Sr3D 受保护模型不变。

现场：screen `mcln_g0_pair_20260905`，old PID3409；固定输出 `/root/autodl-tmp/mcln_g0_view_pair_20260905/results`。

P1 Query identity trace：commit `0ade888`；screen `mcln_padding_identity_after_g0` PID4216，flock4223等待完整G0配对；四条fit、0更新，v2结果待运行。详情master20.33。

P1 REC/Mask选择路径CPU反例PASS；四条fit完整候选诊断已按commit8664e73排在G0后，screen mcln_candidate_contract_after_g0 PID4706。真实回执待运行，正式指标未更新。P2草稿接口审计见master20.34。
