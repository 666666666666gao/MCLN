# 六面残差尺寸初始化：源码救援审查

结论：**WARN；源码结论 PASS；blocking_findings = []。** 该最小修订消除了新残差层最终尺寸下限与“零残差必须精确保留旧轴向输出”的矛盾。可据此继续既定的串行两步工程复测；本报告不确认修订后的 GPU 检查成功，也不准入完整三轮训练。

`execution_scope = SOURCE_ONLY`；`review_independence = same-family`；`acceptance_status = provisional`；reviewer identity **UNATTESTED**。本审查没有运行模型、PyTorch、数据加载器、梯度、GPU、SSH 或训练。检查了本地文件、哈希、差异及 Python AST，并只读查看已有失败记录和补充 CPU 回执。完整输入身份见 `SOURCE_REVIEW.json` 的 `audited_input_hashes`。

1. **失败与修复对应。** `REVIEW_INPUTS.json` 的 26 个文件全部匹配 SHA-256。旧 `preflight.log` 在 `native_face_preflight.py:175` 的尺寸相等断言退出，退出码为 1；该行之前的中心断言已通过，训练调用在其后，不能将此失败写成完成更新。旧、新残差源码的唯一计算变化是 `raw_size.abs().clamp_min(1e-6)` 改为 `raw_size.abs()`；工厂文件仅同步参数化说明，控制器另记录真实 child 退出状态。

2. **保留旧轴向输出的计算成立。** 新模块 `source/face_residual_span_mixer.py:18–21` 将最后线性层置零；`prior_gate` 已在 `[0,1]`，所以初始 residual、face movement 均为零。`57–63` 直接在旧中心与尺寸上加移动量，正 prior size 的 `abs` 不改变值。旧版多出的最终 floor 可以改变这个正值。保留 `abs` 仍会按中心及绝对尺寸表示独立面移动后发生交叉的两端，不新增 GT、筛选或备用框。

   **实际尺寸幅度仍待证据。** 补读并绑定的 `native_mask_geometry.py:35` 已将 mask size floor 到 `1e-6`，旧 `extremal_span_mixer.py:38` 同样 floor native size；数学凸组合不低于此下限，但 float32 分项乘加可能舍入到略低于它的值。原失败日志没有尺寸数量、最小值或误差，不能据此声称真实数据有 `1e-7` 量级尺寸。新增 `native_face_preflight.py:174–185` 在严格断言前保存这些真实量，且没有放宽相等标准。

3. **补充 CPU 回执的含义受限。** `cpu_neutral_size_regression.py` 与 `CPU_NEUTRAL_SIZE_REGRESSION.json` 的源哈希对应新旧残差源码，回执记录 PyTorch `1.13.1+cpu`、旧版改变 64 个构造尺寸元素、新版改变 0 个、两模式初始输出相等。此测试直接调用 `refine_one`，构造了 `1e-7/3e-7` mask sizes，未经过实际 native mask floor，也未执行完整 prior forward。因此它验证直接接口上的正尺寸初始化不变量，不能证明原 GPU 样本的真实尺寸幅度、实际修订 GPU 成功或精度。本审查没有重新执行该测试。

4. **初始化、两组容量和来源控制正确。** 工厂先按原严格路径加载 1234→1271→1295 个状态，再包装原 prior 并新增 6 个状态至 1301；新增残差参数 23425，保留 prior 29793，合计 53218。预检将旧 prior 前缀映射回去，逐项核对 1295 个 E0 状态，另检查新输出层为零。两组协议仅 `face_residual_mode` 和 init 文件路径不同，init 内容仅模式不同，118 个源文件身份完全一致。关闭组仅将新增残差输入中的极值 token 和成员比例置零；旧 extremal prior、其 gate、Query、几何与有效性仍在，因此只能叫“关闭新增来源证据”的同容量控制，不能叫完全无来源模型。

   两组都继承曾训练的 support/span 权重，C 仅在当前额外 selected-mask 监督上关闭。预检明确承认比较用 E0 parent 的 C 标志原为真；这不是 never-C 初始化，也不是改从旧 C-off 三轮终点继续训练。

5. **原生训练与 loss 路径未被替代。** `train_dist_mod.py:128–129` 只改为 face 工厂；`main_utils.py:428–470` 仍完成原生 forward、完整 criterion、backward、clip、AdamW 与 scheduler 更新。预检调用此 `train_one_epoch` 一次，只供给两份真实增强 B8 batch，计划 2 次更新／16 行，未调用完整 `main()` 或正式验证。真实 loader 要求 36665 train／9508 val、4583 batch、drop_last，协议没有 debug 或 joint-detection 替换。RoBERTa 仍冻结，原 trainable core、backbone、support、prior 和 face 模块进入互不重复的优化器分组。

   实际绑定的 `models/losses.py` 保持 Text Mask Hungarian 匹配，cost 为 class=1、bbox=0、GIoU=2、mask=0.0002；不是另一个 QueryMask 匹配实验。最终中心／尺寸进入已有 GT L1 和 GIoU，G 修正仍启用，C extra mask loss 关闭。GIoU 坐标转换和原生评价的既有 floor 保留，未增加 DFL 或新 loss。梯度检查接受首步零输出层导致隐藏层梯度为零，并要求第二步 face internal 梯度为正，累计 prior/output 梯度为正及分组参数有改变。分组梯度通过不等于每个参数均有效，更不等于各机制有独立精度贡献。

6. **保存恢复与计数一致。** 两步后检查 loss 次数=2，保存 epoch=0 的工程 checkpoint，formal metrics 为 null；要求 1301 个模型状态、826 个 Adam 状态且每个 step=2。826 与原正常回执的 820 加新增 6 个参数张量一致，实际值仍由严格断言确认。恢复用同模式工厂、`initialize_weights=False` 重建后严格加载，逐项核对模型、Adam 值和 param groups、scheduler、Python／NumPy／Torch／CUDA RNG，start_epoch=1。这里是**同一进程内重建并恢复模型和状态**，没有验证另一冷启动进程，也没有声称恢复下一训练 batch 或精确续训轨迹。工程权重不供正式训练使用。

7. **源路径、串行和失败状态有相应约束。** staging 从绑定的旧 116 文件复制到隔离 v2 目录，再覆盖两个入口并增加两个 face 文件；协议及 SOURCE_HASHES 的 118 项一致。入口先切换到该源目录并放到 import 搜索路径首位；控制器检查源哈希，最终回执会记录实际 import origins。启动器要求原 C-off 完成、单张空闲 A100、预留空间和既有 `flock` 锁，只启动一组；第二组需再次满足这些条件。工程 admission 只打开 serial preflight，`full_training_admitted` 始终为 false。child 结束后先写 `failed` 或 `checking_receipt`、退出码和时间，再做成功断言，修复了本次退出后状态仍写 running 的实际问题。观察器首次约 480 秒、随后 240 秒，不自动重启失败任务。

后续只需按现有工程计划获取并审阅：真实 `INITIAL_OUTPUT_COMPARISON.json`、两组各自的两步梯度／状态恢复回执、两组 `input_batch_sha256` 的实际相等性和 import origins。源码采用同 seed、sampler epoch、loader generator 及相同构造路径，并记录完整 batch 指纹，但当前没有两组实际回执可比较；单组控制器也不会自动完成跨组比较。

本次另读的 dataset-source `utils/lr_scheduler.py` 原始哈希及 LF 归一化哈希均不等于部署 pin，故未把它当作实际 scheduler 源码。scheduler 结论限于已绑定调用、保存恢复断言及旧恢复回执，不伪称独立检查了远端全部 118 文件。所有补读文件身份及用途均记入 JSON。语法检查包括 14 个清单 Python 文件、3 段内嵌远端脚本及 7 个补充 Python 文件；没有执行其中任何模型或远端操作代码。

正常三轮训练、正式精度、独立模块收益以及完整研究目标均未由本审查确认。无需为此次修复加入 fallback、额外异常吞没、兼容层或无证据的防御分支。
