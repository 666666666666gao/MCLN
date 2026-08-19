# V99：ScanRefer 双阶段 REC@0.25 最优系统封存

本文件封存当前 ScanRefer 双阶段 `REC@0.25` 最好完整系统 V99。封存不复制大权重；四个必需组件继续使用服务器上的只读文件，并由不可变 JSON 清单绑定路径、SHA-256、inode、权限和大小。

## 封存结论

V135 关系反事实困难负样本在 scene-disjoint 128-row holdout 的两个 epoch 都完全回退父模型：REC 为 `124/117`，`fix=break=0/0`，没有部署修复。按预先约定，V135 不进入 9,508-row 正式评估，其全部 smoke 权重已在指标、日志和合同审计后删除。因此停止继续追逐该双阶段方向，并把 V99 作为 `REC@0.25` 与三项 Mask 的正式最好系统封存。

V99 是一个真实存在的单一系统，不拼接 V99、V109、V113 的单项最好。V113 的 `REC@0.50=50.8519%` 仍单独保留，用户指定的 V109 也继续保留。

## 正式指标

评估集为完整 ScanRefer validation，共 9,508 条；V99 使用修复后的官方 mesh-derived superpoints。

| 指标组 | Acc@0.25 | Acc@0.50 | mIoU |
|---|---:|---:|---:|
| REC Overall | **58.6033% (5572/9508)** | 50.4523% (4797/9508) | — |
| REC Unique | 88.8654% (1261/1419) | 80.5497% (1143/1419) | — |
| REC Multiple | 53.2946% (4311/8089) | 45.1725% (3654/8089) | — |
| Mask Overall | **59.8443% (5690/9508)** | **52.3349% (4976/9508)** | **45.9303%** |
| Mask Unique | 90.2044% (1280/1419) | 80.1268% (1137/1419) | — |
| Mask Multiple | 54.5185% (4410/8089) | 47.4595% (3839/8089) | — |

`REC@0.50` 已超过目标 138 hits；`REC@0.25` 距 59% 仍差 38 hits。Mask 三项相对用户给出的 MCLN baseline `58.70/50.70/44.72` 分别提高 `+1.1443/+1.6349/+1.2103pp`。

## 可主张的创新点

1. **Top-16 query-set 上下文层级。** 对冻结 REC 候选集合使用单层、四头、置换等变 Transformer，使每个候选的判断显式依赖同场景其他候选，而不是逐候选独立打分。
2. **query 与 mask variant 的层级选择。** 先对 query 建模，再在每个 query 的有效 mask/geometry variant 内打分；候选 padding 被严格屏蔽，最后选择一个可部署的扁平 query-variant 索引。
3. **双阈值任务头与有界质量目标。** 训练目标固定为 `IoU + 2*hit@0.25 + hit@0.50` 的 soft-listwise 形式，分别预测 `@0.25`、`@0.50` 的命中倾向，而不是只拟合单一连续 IoU。
4. **固定 Pareto 安全部署门。** proposal 相对 parent 的预测 `delta@0.25` 和 `delta@0.50` 必须都严格为正，并满足 `2*delta@0.25 + delta@0.50 > 0.13312220573425293` 才允许切换；该门没有额外验证集阈值搜索。
5. **scene-disjoint OOF 与完整来源绑定。** 五折训练 OOF 的净增益为 `+175/+474 hits`，两个 scene-bootstrap 95% 下界为 `+132/+385`，五折均为正；full-fit artifact 绑定 backbone、parent、geometry、normalization、候选 digest、模型状态和 OOF receipt。
6. **Box/Mask 同 query 的部署。** 最终 REC query 同时驱动 Mask 选择。mesh-derived superpoint 修复属于数据管线纠错，不是对 V99 artifact 使用 validation 重新训练或调阈值。

## 主张边界

- 可以主张 V99 是当前单一完整系统的 `REC@0.25` 和 Mask 三项最好结果。
- 不应把 V99 的 `REC@0.25` 与 V113 的 `REC@0.50` 拼成一个模型结果。
- V99 artifact 在旧 mixed-superpoint official 上相对 frozen geometry parent 只带来 `+10/+24 hits`；最终 meshSP 复核的额外变化同时包含数据管线修复影响。因此论文中要把“模型创新”和“superpoint 纠错”分开叙述。
- 正式运行在 9,508 条预测和全部指标打印后，旧 subgroup export 才返回 code 1；receipt 明确标记为 `post_metric_export` 恢复结果，不能写成 clean exit。

## 只读依赖

| 组件 | SHA-256 |
|---|---|
| epoch71 backbone | `3e44f4bdad3bd66ad82102032e1cb0241de57d147c0aa1d3eff9736926ef2208` |
| parent reranker | `f06f8972fdcfbbdcb799df267864ab2ebc9ca8403ff92576e2bbdb0a8c17269b` |
| geometry reranker | `835c25be4717dfcbb324e0c4c5b9d1d3f3e2b90a4dbcb4d4ebe79f215f263b6f` |
| V99 contextual hierarchy | `9752990c393fa6e45173a9dd129c4de4bb740924094dcbbec2f3121cbf39d1f2` |
| official result receipt | `311097c8a0fc1eceab3c95983937071e67fd8082ac46d1af5d3701ada4eb491c` |

不可变机器清单由 `scripts/archive_v99_rec025_best.py` 生成，固定位置为：

```text
/root/autodl-tmp/DATA_ROOT/output/v99_rec025_best_archive/v99_rec025_best_archive.json
```

清单只记录元数据，`weight_copy_count=0`，不会额外占用约 794MB 的 backbone 空间。
