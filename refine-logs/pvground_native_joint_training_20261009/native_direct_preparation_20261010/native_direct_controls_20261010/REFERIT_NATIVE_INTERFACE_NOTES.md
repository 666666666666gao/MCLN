# Nr3D/Sr3D正常训练接口：只读核对

状态：2026-10-10，仅源码契约核对。没有构造Nr/Sr数据集，没有读取新的远端运行状态，也没有开始跨基准训练或评估。本记录用于固定最终模型后接入正常训练，不能当成泛化结果。

## 已读的实际源

- 当前正常训练源：`source/native_model_initialization.py`、`source/pvground_semantic_assignment.py`、`source/selected_query_mask_objective.py`、`source/native_root_bbs.py`、`source/models/losses.py`。
- 实际保留数据接口的本地精确快照：`pvground_extremal_span_evidence_20261009/runner_v1/source_review/snapshots/038_joint_det_dataset.py`。SHA256为`3afebfc69232f9547e6810563ba6dd4f34a0f770f929df2f4203b21493de0e3d`，与正常源发布清单的`src/joint_det_dataset.py`一致；这里不声称重新查询了活动服务器源码。
- 作者源历史副本：PV工作树`refine-logs/pvground_pretrained_resources_20260908_v1/source/src/joint_det_dataset.py`。用于对照原本的root/anchor组织，不代替当前保留接口。

## 已确定的迁移差异

| 项目 | 源码证据 | 需要的实际适配 |
|---|---|---|
| 模型初始化 | 初始化第19–20行明确断言ScanRefer和seed2027；随后加载ScanRefer官方/G/d06/f989链 | Nr/Sr分别绑定对应作者核心权重和明确的新模块初始化；不能加载ScanRefer的1072个G delta覆盖Nr/Sr核心，然后称为仅从对应作者权重开始 |
| G的文本目标 | `pvground_semantic_assignment.py`第26–31行定义0.6/0.2/0.2/0.1，第35行限定ScanRefer；第47行外部系数0.5/7 | Nr保持其原生文本权重，Sr按0.625/0.125/0.125/0.125；外部语言系数Nr/Sr为1，并依据实际decoder层数保留原始归一化 |
| 原生总loss | `models/losses.py`第484–487行区分Sr文本权重；第946–960行ScanRefer的CE/对比系数0.5，其他为1 | 监督公式与本基准原生criterion一致；不把ScanRefer权重写死到迁移模块 |
| C选中候选Mask目标 | `selected_query_mask_objective.py`第13–16行断言ScanRefer、首个有效GT为0、root匹配一次；第18–24行保护所有已匹配候选 | 在真实指代表达上验证目标槽及匹配对应；联合检测行不能按第一个GT自动当root。保留matched-other原职责 |
| 身份与样本来源 | 实际数据快照第1264–1272行：`language_dataset=self.test_dataset`，`sample_dataset=anno['dataset']`；第1402–1403行分别返回 | 用实际注释来源区分真实表达和联合检测；不能因为language_dataset是Nr/Sr就给同batch所有检测样本添加唯一root监督 |
| root与anchor | 实际数据快照第1086–1119行：指代目标先进入槽0；启用detect_intermediate时再追加anchor；第1077–1079行Sr可添加anchor文本槽 | 真实样本检查root-first契约以及非空多GT保护；现有ScanRefer root-only面板不是多GT测试 |
| 推理评分 | 当前`native_root_bbs.py`仅汇总表达token，不做额外质量排名 | 三个基准都保留唯一原生评分和同Query框/Mask；需逐项核对各自实际评估器，不能把ScanRefer函数名称当成已经完成Nr/Sr数值等价性 |

## 初始化与训练历史必须先写清

使用各自作者权重已获用户授权。最终Nr/Sr运行须列出核心来自哪个对应权重，G/A/B的新模块采用重新初始化还是继承ScanRefer模块，以及优化器是否重新初始化。继承ScanRefer模块属于额外训练历史，必须披露；尤其不能把G完整核心delta与“仅对应作者父模型”混称。

架构和机制保持统一，原数据协议及原生损失系数可以不同；模型forward不读取dataset ID决定答案，也不新增测试期GT目标/Anchor。检测候选辅助与GT对象输入分别报告。单seed2027，单A100串行。

下一阶段必须实际检验：对应训练/验证注释数量、root与anchor、联合检测行、作者核心严格加载、新增状态与优化器冷恢复、真实GPU前向及原生最终评分。当前这些检查未完成；不能仅删掉ScanRefer断言就启动训练。

先等待当前ScanRefer正常训练的真实结果并固定最终版本。所有ScanRefer、Nr3D、Sr3D结果最终来自同一完整结构；不把已冻结的ScanRefer强版本和另一套Nr/Sr结构拼成一个方法。
