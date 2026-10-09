R8 SOURCE_ONLY 结论：FAIL，唯一阻断 B8.1 是新启动器在 Python 3.7 远端代码中调用 shlex.join。限定的预检梯度/诊断修订本身通过源码核对；第一次 M0 的失败历史、实际更新和完整恢复要求均保留。

这是 /root/pvg_native_joint_source_20261009 原审查任务的续轮，不是新 fresh review。原始 fresh context=true，本轮 fresh_context=false。请求 gpt-6-astra / max；实际模型、推理级别与路由均 UNATTESTED，same-family / provisional。

B8.1：远端标准库 API 与固定暖环境不匹配

preflight_attempt2/launch_authorized.py 第 66、69 行位于发往暖环境 Python 的嵌入 code 内，分别为 flock 命令及 pgrep 的 argv 表达式调用 shlex.join。固定 runtime 仍为 /root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python。

R5 已封存的原生 CPU 导入失败栈明确包含 /root/miniconda3/envs/bdetr/lib/python3.7/importlib/__init__.py；当前 NATIVE_SOURCE_PORT 仍绑定 cpython-37m 的 PointNet extension、同一暖环境 SHA，R5/R7 及第一次 GPU 运行的源身份一致。该环境的 Python 3.7 不能提供 Python 3.8 才增加的 shlex.join。[Python 官方文档](https://docs.python.org/3/library/shlex.html#shlex.join)

源码执行到第 66 行将因缺少该属性而失败，尚未启动 screen/controller；在此之前却已经创建 attempt2 输出目录、复制 helper 并写 admission。这里是从固定解释器与 API 条件推出的源码阻断，没有实际执行 attempt2，也没有声称观察到第二次运行的 AttributeError。

最小修订仅将嵌入远端 code 的两处 shlex.join 改为 ' '.join(shlex.quote(arg) for arg in argv) 的对应列表表达式。第 67 行的 shlex.quote 可保留；本地 Windows Python 执行部分第 82 行的 shlex.join 不属于此阻断，不需要一起改。无需 fallback、解释器升级、安装或兼容层。父任务已同意先封存本轮，再提交 R8.1 限定差异复核。本轮未编辑任何执行文件。

预检修订核对

- 以 native_preflight_failed_attempt1/native_joint_preflight.py 为直接基线，其 SHA 等于 R5 封存 preflight。新文件只新增现有两次训练 forward 的 raw gate 诊断、把训练后的诊断落盘移到验收断言前、把 span_output 从每步非零改为两步梯度范数之和非零，并在成功收据中记录该验收语义。
- backbone、original_core_and_G_reader、support_corrector 仍要求每一步梯度范数非零。现有 hook 的梯度有限性检查、loss 有限性检查、真实两批 batch=8、普通 train_one_epoch 调用和 optimizer/scheduler 路径未变。
- backbone/core/support/span_output 四组真实 changed_elements>0 的原断言完整保留；完整模型、Adam 状态与 param_groups、scheduler、epoch、Python/NumPy/Torch/CUDA RNG 的保存和重建恢复核对完整保留。没有用梯度累计替代参数更新检查或恢复检查。
- NATIVE_M0_TRAINING_DIAGNOSTIC.json 在训练完成、两次 loss 数量核对及实际参数差值统计之后，且在训练后梯度与参数更新验收断言之前写出。它包含每步 gradient_norms、losses、raw gate min/max、raw<0、raw==0、raw>1 的计数、元素数和真实 updates。低端比例可由 (negative_count+zero_count)/elements 得到，零边界与负区间可区分。
- 诊断读取的 span_raw_axis_gate 确由当前 models/pv_ground.py 第 612 行产生；该文件的哈希与 R5/R7 相同。诊断对现成结果 detach 后统计，没有新增 NN forward、训练步、loss 项或参数修改。
- 两步累计非零与有界门的计算结构相符，但不能据此确定第一次失败就是 span。第一份日志只显示原第 161 行 support/span 合并断言失败，没有哪组、哪步为零的数值。第一次失败之后的参数变化与完整恢复检查未执行，不能追认通过。

独立 attempt2 组织与入场

1. 三个 helper 的 SHA 均匹配 SOURCE_MANIFEST；controller 与 protocol 字节仍等于 R5，observer 字节等于 R7 已审阅版本。新 launcher 在任何 SSH 子进程前逐一校验三个 helper 的 manifest SHA、R8 审查输入 SHA，以及自身的 R8 审查输入 SHA。本轮 JSON 对所有六个 attempt2 文件使用 Path.resolve() 后实际 D: 路径与裸 SHA256，满足该消费者契约。
2. 远端目标固定为 /root/autodl-tmp/pvground_native_joint_training_20261009/preflight_attempt2，要求精确路径、父目录无符号链接重定向且目标不存在。它只向新目录写三个扁平 helper；旧 preflight 和本地失败快照均仅作为读入前提，无覆盖或删除语句。
3. 入场要求本地失败 INTAKE 的 controller_exit=1、采集新增 neural calls=0、observer closed 且旧控制器不活跃；远端还要求原 preflight/controller.exit=1 且原 PID 不存在。当前 NATIVE_PREFLIGHT_WAIT 确为 exit1；其中 status 仍为 running 是失败前的状态，不能当作成功 complete。
4. 远端代码核对暖环境 canonical SHA、116 文件清单内全部远端文件 SHA、两种模式的 official/G/support 三份父权重字节 SHA。116 文件清单自身与 R5/R7 完全相同；本地 14 个计算源码逐一哈希未变。本轮没有重新语义审查其余源文件，也未执行远端 116 文件验证。
5. 创建新目录前要求同一文件系统可用空间大于 2605151860 字节、无 GPU compute PID、恰好一张 index 0 A100、总显存 40000–45000 MiB、已用显存低于 500 MiB。此空间数值与 R5 审阅预算相同；真实当前空间、实际序列化大小和第二次入场仍 pending。
6. 声明中的原 span 已闭合沿用既有封存历史；第二次入口新做的退出/PID条件是针对第一次 M0，不能表述成它重新查询过原 span。预定 flock 使用现有暖环境 GPU lock，位于独立 screen 内包住 controller；B8.1 阻止当前代码到达该步骤。
7. controller 顺序执行两种模式的两步工程预检，任何非零返回即停止；不调用正常训练 launcher。observer 绑定 attempt2 的 launch 文件、PID 与 argv，首次计划 1500 秒，之后每 240 秒观察，证据写在 attempt2。后续新成功收据、收集/退役/正常训练路径绑定文件不在 R8 范围，未生成占位收据或 SHA。

保留的证据与边界

31 个文件输入保留完整字节快照和 SHA；其中 14 个计算源码仅进行身份哈希核对，新增诊断字段的生产位置做了必要定点读取。R5 的 69 个和 R7 的 42 个封存 artifact 全部无漂移。旧失败 INTAKE、源码、whole_support.log 和 exit1 wait 收据均纳入本轮，未改写历史。INTAKE 的 new_neural_calls=0 描述的是采集过程，没有否认第一次 GPU 运行。

原始执行工具回执、差异、提取的远端代码、结构检查和完整报告分别落盘，RAW_REVIEW_R8.md 与 RAW_RESPONSE_R8.md 保留同一报告全文。一次 reviewer 读取脚本因 R5 seal 使用 artifact_hashes 而非 artifacts 报 KeyError，随后按真实 schema 读取；失败回执保留。Python 3.8 历史文档 URL 一次不可访问，随后官方当前文档的版本标注可核对；外部文档只保存明确标注的短证据摘录，不伪装成完整网页原文。

本轮未执行启动器、SSH、GPU 命令、Torch/模型构造、checkpoint 加载或训练；未读取 AUTH、凭据值、私人 askpass、known-host 文件、全局 MEMORY，也未调用 get_goal。脚本中出现的传输文件路径只作为源码文本审阅，未打开这些私人文件。首次真实 M0 失败属于既有输入证据；R8 不提供任何新的实际运行通过结论。

待完成：先修正 B8.1 并封存 R8.1；之后的真实 attempt2 M0、两臂诊断数值、实际参数更新和完整恢复、真实空间/锁/源与父权重入场、成功收据的实际审查以及后续正常训练路径绑定均须独立完成。

