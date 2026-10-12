# Launcher 与训练责任计数追加审查

**结论：源码准备 PASS；综合 WARN；`scope=SOURCE_ONLY`；`source_preparation_pass=true`；`scoped_blockers=[]`。** 本轮没有发现仍待修复的范围内逻辑、接口或启动边界错误。WARN 保留运行门禁和审查归属边界，不代表模型失败。

Requested **gpt-6-astra / max**；实际身份和推理档位 **UNATTESTED**；**same-family / provisional**。本轮复用已闭合 R2，是同一委派审查者的追加审查，不宣称新 fresh reviewer 或实际运行验收。

**范围与原证据保留。** 审查六份新增执行源码、`prepare_launcher_sources.py`、其保留失败版本与两个 preparation/syntax JSON。R1/R2 报告四份文件的 SHA-256 均与原值相同；R2 五个 overlay 同字节保留。已存 `SOURCE_STAGE_RECEIPT.json` 记载 **2026-10-12 18:00:08 +08:00、117 源文件、0 GPU/NN/progress query**，其 review SHA 对应原 R2 JSON；该回执及全部内容未改。本轮只读取这个存档，没有再次连接远端认证其当前状态。根目录为 `C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012`。

**确定性 CPU 结果。** 使用且仅使用 `E:/anaconda/python.exe -B`（Python 3.7.6），未导入模型或 Torch。六份实际保存执行源码与生成器均直接 AST 通过，CRCRLF=0；三个嵌入 remote 片段均可解析；三个完整 SSH argv 经实际 shlex split 往返一致。仅执行字符串生成和 argv 构造，没有运行任何 wrapper/controller 入口。CPU 工具回执 `f307d5` exit0，30 项绑定输入前后相同。

保留的 `prepare_launcher_sources_r1_failed.py` 在只执行文本生成部分时复现 **generated controller line46: EOL while scanning string literal**。`prepare_launcher_sources.py:13` 的 raw 三重字符串修复使实际保存内容正确；另一个差异是锁内成功检查后记录 `GPU_device_checked=True`。当前生成器精确复现实际 `normal_run_controller_v3.py` 与 `source_overlay_v3/main_utils.py`，旧失败文件不覆盖。完整 diff 与异常已入 trace。

| 检查 | 源码结论 |
|---|---|
| 责任计数只读原记录 | `source_overlay_v3/main_utils.py:460–462` 初始化策略组计数；`:484–491` 在原 `_compute_loss` 返回后累加已有 record；`:529–535` 于整轮完成后校验并写 JSONL。控制组不读该字段。 |
| 无训练方法漂移 | 删除上述 **三个新增 If 分支** 后，完整 `main_utils.py` AST 与 R2 **完全相等**。loss、indices、targets、优化器、梯度步骤、seed/sampler、E0、8 轮/strict decline2、best/latest 政策及推理均未改。 |
| 单卡锁使用一致 | 两个 launcher 的嵌入命令都为 `flock --no-fork -n environment['resource_limits']['gpu_lock'] runtime -B -u controller`，读取同一个既有 env_spec。与旧 MCLN launcher 的锁设置相同。锁在控制器等待子进程期间持有。 |
| 工程两组 gate | `launch_gpu_preflight_pair_authorized.py:17–26,37–40` 要求原 normal-training 真实 complete/exit0 intake、原 PID 271902/271903 不存在和新 source stage。控制器 `:18–33` 在锁内再次核对 terminal、环境、helper/source、空闲单张 A100 和空间，然后才启动工程子进程。 |
| 工程顺序与责任验证 | `gpu_preflight_pair_controller.py:46–82` 顺序 control→strategy，并等待每组结束；每组要求 2 更新、真实两批/16 输入、36665/9508 dataset、1295 state、同进程恢复且不带工程权重进入正式训练。策略另需实际 extra_rows 和 direct-gradient 总量为正；两组输入指纹须相等。结束只记录 pair complete / formal_training_admitted=false，不自动开正式训练。 |
| 正式准入 | `launch_normal_arm_authorized.py:22–37` 要求实际 pair complete/4 更新，并且独立 `actual_pair_review/EXPERIMENT_AUDIT.json` 的 execution_scope 必须是 ACTUAL pair、无 blocker、所审 intake 哈希匹配。本 SOURCE_ONLY 报告不能替代该 actual audit。 |
| 正式组顺序与 fresh 起点 | remote 段 `:48–59` 重核 MCLN/pair 终态、PID、真实 receipts；`:61–64` 限定 selected_geometry 在 native_control 完整结束且其训练 child 已退出后开始。协议禁止 checkpoint/frozen，仍使用原 official/G/A/B init spec。工程权重路径没有进入正式命令。 |
| 锁内正式设备准入 | `normal_run_controller_v3.py:25–47` 在持锁控制器内再次核对环境、MCLN/pair、GPU 空闲/单卡、磁盘和 117 源文件，成功后才置 `GPU_device_checked`。随后 `:58–69` 唯一训练入口仍是原生 `train_dist_mod.py` 并等待退出。 |
| staging 写入边界 | `stage_launcher_sources_authorized.py:26–30` 要求既有 manifest 一致、尚无 preflight/normal 目录，先归档 R2 manifest 和原 main bytes。`:31–45` 只写两个明确 helper 和 `PV-Ground/main_utils.py`，更新其条目、保持 117 源文件并校验全部记录。嵌入 stager 没有 MCLN/GPU/进程进度查询或模型导入。 |
| SSH transport | 固定 OpenSSH argv，已有 askpass 与 known_hosts、StrictHostKeyChecking=yes、显式引用的 remote Python 参数；凭据来自环境。本轮未读取或输出凭据，也未执行 transport。 |

`extra_loss_sum` 的语义是**已有、已按 batch 归一化的每批新增 loss 的和**，不是合格 Query 的平均损失，也不是精度。roles、rows、extra_rows 来自原 objective 的同一 record；整轮检查 roles 总数等于 rows，extra_rows 不超过 unmatched。该日志只在策略组完整轮后产生，适用于现有 WORLD_SIZE=1，不能用未完整轮的输出冒充完整统计。旧 selected-mask 两组关闭、原匹配和新增责任边界继续沿用 R2。

**启动回执不能升级为运行通过。** launcher 的 `Popen` 与 `poll() is None` 只说明刚发出的控制器进程尚存活；未证明它已拿锁、通过 device admission 或完成模型预检。pair 的预先 admission 记录也不能单独当作 GPU 成功证据，应以锁内检查后实际状态、日志和 receipts 判定。正常控制器的完成检查仍只核对指标/停止记录与 best/latest 文件存在，checkpoint 内容和真实保留最优轮仍需 actual result audit。没有要求为未发生的错误增加 fallback、try/except 或新的 hash 框架。

**仍待真实执行。**

1. 执行尚未运行的本实验隔离 R3 staging，并核对实际回执。
2. 从既有观察器取得 MCLN 的真实 normal 终态 intake；在持锁控制器内通过当时的 device/storage/source 检查。本报告没有认证当前 MCLN、PID 或 GPU 可用性。
3. 真正完成两组 native 工程预检与精确 intake 的 fresh actual integrity 审查；此时才验证 factory、full get_loss、实际 B final 几何梯度、原参数组更新与同进程恢复。
4. 分别 fresh 开始 native_control 与 selected_geometry；验证同源同初始化、完整 9508 E0、normal 更新、责任计数、best/latest 与结束政策，之后再作真实配对效果审计。
5. **normal cold resume 继续不受支持。** 同进程工程恢复、保存 history/optimizer/RNG 和正常冷启动续训是不同验收项，不能互相替代。

预检估计 1300 秒、首次观察 1120 秒，后续 240 秒；normal 首次观察暂 1500 秒。这些只是未来调度参数，不是本轮运行耗时测量，也没有创建观察器。启动后应依据真实吞吐修正；本轮没有查询或修改活动 MCLN、观察器 37188 或原 19:05 安排。

R2 的证据边界完整保留：旧 CPU probe 是同 8 个增强输入两次观察共 16 行，只存部分 mask slices，saved Mask endpoint 不是未缓存 B final。新 launcher 和计数代码没有消除 G/no-object 责任关系，也不构成新有效研究模块。任务仍是同一模型 ScanRefer **60/52**、三项有效贡献和 **Nr3D/Sr3D**；CPU/source PASS 不满足这些目标。

**追踪。** `SOURCE_REVIEW_LAUNCHERS.json` 提供 launcher 要求的 SOURCE_ONLY/pass/no-blocker 字段，真实运行各 gate 明确保持 pending。run03 保存完整收到的请求、30 项输入路径/哈希及编号快照、当前源码 diff、失败生成器复现、CPU 原始回执、完整报告、routing/UNATTESTED 元数据和 FINAL_RESPONSE。审查者只写 `source_review`；0 SSH、0 GPU query、0 NN forward、0 训练。
