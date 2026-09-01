# MCLN 同目标多描述路线新颖性核验（2026-09-01）

## 结论

候选路线 **Natural Sibling Disagreement Regularization（NSDR）不能作为下一代 MCLN 的主创新继续开发**。

- C1“同一目标多描述的候选分布一致性”与 2021/2022 年 synonymous referring-expression contrastive learning 的核心思想直接重叠，只是从 2D feature space 换到了 3D candidate-score space。
- C2“把一条 sibling 描述暴露出的错误实例广播给同目标其他描述”保留一个窄差异，但本质接近 group-conditioned online hard-negative reweighting。
- 不生成文本、不使用 parser、单次推理和三数据集统一执行是重要工程合同，不构成独立新颖性。
- 当前总体新颖性评估为 **4/10，CAUTION**；不修改正式训练代码，不启动 GPU。

## 原始候选方法

训练时按 `(scene_id, target_id)` 聚合同目标的普通数据集描述：

1. C1：对齐 sibling descriptions 在同一候选实例集合上的 score distribution；
2. C2：收集各 sibling 的高分错误实例，将只被部分 sibling 暴露的错误实例广播为全组共享 hard negatives；
3. 保留普通每行 grounding loss；
4. 推理仍执行单条描述的一次 MCLN forward，无 gate、threshold、reranker 或 sidecar。

数据可行性并不是问题：Nr3D、Sr3D 训练行的同目标多描述覆盖率为 100%，ScanRefer 为 99.8718%。问题是方法的新颖性和相对于普通 HNM 的实质差异不足。

## Claim 逐项判定

| Claim | 判定 | 最接近工作 | 精确判断 |
|---|---|---|---|
| C1：同目标多描述 Top-K score distribution 对齐 | LOW / overlap | Chen et al., *Understanding Synonymous Referring Expressions via Contrastive Features* | 该工作已把同一物体的不同表达当作 positives，并学习同目标表达在视觉 grounding 空间的一致表示。换成 3D candidate scores 主要是实现位置变化。 |
| C2：partial-sibling 错误实例 union 与 group broadcast | MEDIUM-LOW / narrow partial | 普通 online HNM、same-class HNM、3DVG-Transformer、EG-3DVG | 可辩护差异仅是：错误实例由同目标其他描述暴露，再广播给全组；仍容易被归类为 group-level hard-negative sampler。 |
| C3：无生成文本、无 parser、无推理 sidecar | LOW / system contract | ViewRefer、AugRefer、LEG | 与生成式增强路线形成系统差异，但“没有使用某模块”不是正技术贡献。 |
| C4：同机制覆盖 ScanRefer/Nr3D/Sr3D | LOW / evaluation scope | 多个跨 benchmark 3DVG 方法 | 是泛化验证要求，不是新颖性来源；Sr3D 为模板生成表达，不应全部称为 natural siblings。 |

## 最强拒稿理由

普通 grounding CE 已把非 GT candidates 作为负例；per-row HNM 已选择当前高分错误；same-class HNM 已集中训练同类干扰物。C2 只是把 hard-negative 来源从当前 row 扩展到同一 `(scene,target)` group。若它不能在负例数量、loss weight、训练步数完全匹配的条件下稳定超过 per-row HNM 与 same-class HNM，就只是一个直接的数据采样技巧。

## 唯一可保留的精确定位

若以后只作为小型证伪实验保留 C2，应改称 **Disagreement-conditioned Group Hard-negative Margin（DGHM）**，不再使用 NSDR，也不把 C1 写成创新。

设同目标描述组为 `G=(s,t)`，描述为 `q`，场景实例为 `i`，分数为 `z_q(i)`：

```text
H_q = TopM_{i != t} z_q(i)
rho_G(i) = (1 / |G|) sum_{q in G} 1[i in H_q]
N_G = {i != t : 0 < rho_G(i) < 1}
L_DGHM = sum_{q in G} sum_{i in N_G} max(0, m + z_q(i) - z_q(t))
```

它的唯一技术含义是：某条描述暴露出的、但并非所有描述一致误选的实例，会成为同目标其他描述的训练期共享困难负例。

## 必须先通过的等预算对照

任何未来证伪实验都必须在新的 Nr3D train-only scene-disjoint split 上同时比较：

1. Base MCLN；
2. C1-only；
3. per-row hard-negative margin；
4. same-class hard-negative margin；
5. random sibling broadcast；
6. Chen-style synonymous contrastive adaptation；
7. DGHM/C2-only；
8. C1+C2。

所有分支固定相同 backbone、seed、candidate set、训练步数、负例数量和辅助 loss 总权重；不访问 7,899-row formal Nr3D，不搜 threshold/margin。

## 证伪与停止条件

出现任一情况立即 ABANDON：

- C2-only 没有明确超过 per-row HNM 与 same-class HNM；
- C1+C2 不超过 `max(C1-only, C2-only)`；
- 增益不集中于已诊断的 Multiple/Hard、same-class ranking failures；
- partial-sibling disagreement error 不下降；
- consensus-wrong 增加；
- 改善只体现在 auxiliary loss，而不进入 held-out Top-1 REC@0.25。

即使通过，也只能将其定位为 lightweight training-only regularizer，不能作为新的 contrastive-learning 范式。

## 当前决策

**主路线 NO-GO，窄 C2 仅保留为将来可选的低优先级证伪项。** 当前服务器 GPU 被独立 ScanRefer 消融占用，而且该方向的新颖性不足，不值得现在修改训练主线或挤占正式实验资源。

下一步应继续寻找满足以下条件的结构性方法：

- 直接改变主模型的单次 Top-1 同类消歧能力；
- 监督来源不是模型自预测、自蒸馏或常规 HNM 的重新加权；
- 不依赖 parser、dataset-specific metadata、GT-derived anchor sidecar；
- 推理不使用 post-hoc gate/switch；
- 在 ScanRefer/Nr3D/Sr3D 上保持同一架构与机制。

## 主要近邻工作

- [Understanding Synonymous Referring Expressions via Contrastive Features](https://link.springer.com/article/10.1007/s11263-022-01647-z)
- [3DVG-Transformer](https://github.com/zlccccc/3DVG-Transformer)
- [EG-3DVG](https://openaccess.thecvf.com/content/CVPR2026/html/Park_EG-3DVG_Expression_and_Geometry_Aware_Grounding_Decoder_for_3D_Visual_CVPR_2026_paper.html)
- [ViewRefer](https://arxiv.org/abs/2303.16894)
- [AugRefer](https://arxiv.org/abs/2501.09428)
- [Latent Expression Generation](https://arxiv.org/abs/2508.05123)
- [DDPA-3DVG](https://www.ijcai.org/proceedings/2025/117)

独立 GPT-5.5 xhigh reviewer 的完整 prompt/response 保存在本地 `.aris/traces/novelty-check/2026-09-01_run01/`，按 trace 规则不提交 Git。
