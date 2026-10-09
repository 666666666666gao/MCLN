R8.1 SOURCE_ONLY 结论：WARN；源码修订 PASS，B8.1=CLOSED_SOURCE_ONLY，blocking_findings=[]。实际第二次 M0 尚未由本轮执行或验证。

直接对照 R8 封存启动器，当前 launch_authorized.py 只有三处替换：

- 第 66 行远端 flock 命令改为对对应 argv 逐项 shlex.quote 后用空格连接。
- 第 69 行远端 pgrep 匹配字符串采用相同显式拼接。
- 第 18 行读取 NATIVE_M0_SOURCE_REVIEW_R8_1.json，保留 R8 FAIL 文件。

嵌入远端代码不再调用 shlex.join；本地第 82 行调用保持原样。两处替换保留原 token 列表、quote 语义、flock 包装与 pgrep 表达式，没有 fallback、兼容层、环境变更或其他分支。结合 R8 已封存的 Python 3.7 暖环境证据，本轮关闭原标准库 API 缺口；没有运行第二次启动器来复现或证明远端成功。

其余五个 attempt2 文件逐一与 R8 相同，三个 helper 的 manifest 和审查 SHA、自身 launcher SHA 的入口核对均保留。所有六个实际 Path.resolve() 路径以 D:\Program Files\UserCache\gb\codex\tmp\pvground_native_joint_training_20261009\preflight_attempt2\ 开头；本轮 audited_input_hashes 使用这些绝对路径及裸 SHA256。

R8 的 49 个封存 artifact 均无漂移，原 FAIL、错误启动器和全部输入快照保留。本轮只复核这三处替换及六文件身份；预检真实诊断、Span 两步累计梯度、原每步 core/backbone/support、四组参数更新、完整恢复、独立目录、源/权重/存储/GPU 锁入场等其余结论复用 R8 封存，不重新审查 116 个模型源文件或声称重新验证远端状态。

这是原 /root/pvg_native_joint_source_20261009 审查任务续轮，original_review_fresh_context=true、fresh_context=false。请求 gpt-6-astra/max；实际模型、推理级别与路由均 UNATTESTED，same-family/provisional。

10 个本轮文件输入、完整差异和原生工具回执已保存。RAW_REVIEW_R8_1.md 与 RAW_RESPONSE_R8_1.md 保留完整报告。未执行 SSH、GPU、Torch/模型、checkpoint、任何启动器或训练，未读取 AUTH、凭据、私人 askpass/known-host、全局 MEMORY 或 get_goal，未修改执行文件。

WARN 只表示实际 M0、真实两组数值/恢复、当前远端入场和后续成功收据审查仍 pending；不保留新的源码阻断。R9 的成功后路径绑定属于单独审查范围。

