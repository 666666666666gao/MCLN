# Face residual CPU 模块源代码审查

结论：**PASS，0 个阻断问题**。本结论只支持按 `CPU_CHECK_PLAN.md` 执行一次限定的合成 Torch CPU 模块检查；本次没有执行待审代码，也没有获得 CPU 运行结果。没有发现需要修改的实际 bug。

审查路由由父代理确认：`gpt-6-astra` / `max` / `fork_turns=none`。无独立模型身份凭据，`reviewer_identity: UNATTESTED`、`review_independence: same-family`、`acceptance_status: provisional`。

## 正确的实现与断言

- **固定旧权重与模块布局。** Launcher 从 `source_conditioned_init.json` 只取 d06 support、f989 span 的路径和 SHA；checker 在加载前核对两份权重，再对 A 的 10 项状态和 B 的 14 项状态执行 `strict=True`。旧 B 的实际结构为 29793 参数，包装器直接注册 `axis_prior`，新增 `332→64→32→1` 为 23425 参数，合计 53218 参数、20 项模块状态，其中新增 6 项。CPU 包装器对已加载 prior 深拷贝，未重新初始化旧 prior；无源模式严格装载可见模式的同一状态。对应 checker 39–61 行、wrapper 11–21 行。
- **两个模式的零输出等价检查成立。** A 用随机 50000 点、8 个 superpoint 生成真实 `member_statistics` 几何，`native_mask_geometry` 保留 native coarse 框并设置 mask 框，随后旧 B 与两个包装器读取相同输入。旧 B 已将轴 gate 限定在 [0,1]；新增输出层的权重和 bias 为零，故新增 gate movement 为零，中心/尺寸直接加到旧输出上。checker 85–86 行对两个模式均使用 `torch.equal` 比较旧中心和尺寸；90 行另检查重复后的轴 gate。该断言覆盖本合成夹具，不能延伸为已验证全部真实数据的 Torch 等价。
- **隐藏列位置正确。** 输入列 0–31 是 face token，32–319 是 query，320–324 是五项几何，其中 324 是 source fraction；325 是 reference validity，326–331 是 face identity。隐藏模式确实将 token 与 fraction 清零，checker 的 pre-hook 直接观测首层输入并检查这两组列。它仍共享已经使用 extremal evidence 的旧轴 prior；文档正确地只把它称为新增面残差输入控制。
- **两步梯度夹具与零初始化相符。** checker 94–122 行另建模块，直接调用 `refine_one`，明确供应 0.5 内部 gate。所选几何使每个轴的 native-minus-mask 两面差分别为 0.05、0.55，损失对 residual 有非零导数。第一步输出层可获得梯度，而零输出权重使首层梯度为零；第一次 AdamW 更新输出层后，第二次反传检查首层非零梯度。代码确实执行两次仅针对新 head 的内存 AdamW 更新，并检查输出有限且尺寸为正。它没有声称加载旧 prior 的实际门控、A/B/PV 主干或 native criterion 已通过梯度检查。
- **供应残差的倒置解码正确。** checker 126–142 行明确替换输出为给定残差，x 轴源中心分别为 4 和 0、单位尺寸，原 gate 为 0.5；两个面改为 gate 1 和 0 后，原始尺寸为 -3，绝对值解码后为 3，中心由实现保持为 2。代码断言原始/最终尺寸及输出有效性，不把该结果说成网络已学会预测这种残差。
- **内存恢复的声明有明确边界。** checker 144–156 行把完成两步更新的独立梯度模块 state 保存到 `BytesIO`，严格加载新建模块，再对相同 `refine_one` 输入比较 evaluation 中心/尺寸。它包含模块状态加载，未保存文件，也未恢复 optimizer。此 roundtrip 的 prior 是梯度夹具中新建的 prior，且恢复后没有调用完整 wrapper forward；因此它不证明 f989 包装器全 forward 恢复或 PV1295/1301 模型恢复。

## 执行边界与来源核对

Launcher 的本地写入限定在新建的 `F/cpu_execution`；远端显式写入限定在不存在的 `/root/autodl-tmp/pvground_face_residual_cpu_20261010`，只上传两个待审模块和 bundle，并写该目录内的 stdout/stderr、结果及执行记录。固定 native source 路径只用于哈希、导入和工作目录，两个 Python 进程都带 `-B`，不会因这些导入写 source bytecode。环境设置空 `CUDA_VISIBLE_DEVICES` 和单线程；所有 Torch 构造及 checkpoint `map_location` 都在 CPU，最终检查当前进程未初始化 CUDA。代码没有活动进程、训练日志、GPU 利用率查询，也没有启动、停止、替换当前正常训练或改变 C-off 顺序的操作。审核过程中没有读取认证文件或执行 SSH。

实际审读的 A/B/geometry 三份本地源 SHA 与 `NATIVE_SOURCE_PORT.json` 所列 SHA 全部一致；A 依赖的 `whole_mask_range.py` 同 SHA 本地副本也已审读，导入及 `member_statistics` 没有文件写入。Port 的 `model_source` 与远端 launcher 断言路径一致。已核对 launcher 的现有环境哈希计算：本地两份同内容 env spec 的 canonical SHA 与 port 的 `966235…` 一致，且具有所用 `PYTHONPATH`；没有把另一个旧 env 副本冒充目标 runtime。远端当前文件和环境仍由未来运行时断言核实，本审查没有远端观测。

7 份 Python 源及 embedded remote launcher 通过纯 AST 解析，参数数量另用整数算式核对。AST/哈希工具过程中的本地错误及处理完整记录在 trace：默认 `python` 缺少 `pyvenv.cfg`；第一次选择的旧 env 副本 canonical SHA 不匹配，随后只读对照找到与 port 一致的现有副本。这两项是审查工具/候选副本选择问题，不是待审 runner 的执行失败；没有安装软件或更改任何待审输入。

## 不支持的结论

本次未执行 Torch CPU runner，未直接读取远端 d06/f989 权重，未核验活动训练状态；没有真实场景、PV1295/1301 工厂、native criterion、完整 optimizer recovery、GPU、REC 或精度增益证据。`CLOSED_ALGEBRA_WITNESS.json` 中 9508 行、31 个倒置轴等数字仅作为来源背景阅读，本次未重算其数据级统计。`METHOD_PREPARATION.md` 所述未来工厂接入亦未在本次审核。所有运行结果须等实际 CPU 执行后再做限定 outcome review；本 PASS 不构成 GPU 或正式训练准入。

`blocking_issues: []`。没有要求新增 fallback、防御分支、哈希方案或无关重构。
