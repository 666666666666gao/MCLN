源码准备通过；综合结论 **WARN / SOURCE_ONLY**。当前 `source_preparation_pass=true`，`scoped_blockers=[]`。复用已闭合 R2/R3，本次只审查新增评估诊断、观察收集和 R4 入口；没有新的未解决源码阻断。此结论不能作为设备准入、实际 GPU 预检、完整训练或精度通过。

请求路由为 gpt-6-astra/max；实际模型和 reasoning 身份均为 **UNATTESTED**，同 family、provisional。本次仍是同一个被委派 reviewer 的限定追加复核，不冒充新一轮独立 fresh reviewer。全程 0 SSH、0 GPU 查询、0 神经 forward、0 optimizer update、0 活动 MCLN 进度查询。未读取私密 MEMORY 或凭据；只在 source_review 内写审查记录。

发现的一项证据问题已闭合：先前成功 CPU 检查器读取 dataset_source 的 evaluator（SHA b77376d1…），而 R3 真正暂存的是 model_source 的 PV evaluator（SHA **39c8de921aff9fea6e2a9a39f68e5634bdd765b1bfd065f0ee3c6c9dac931677**）。两者常规 bbs 数学相同不能替代来源绑定。父任务保留旧脚本/旧 receipt，当前 [check_eval_diagnostics_cpu.py](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/check_eval_diagnostics_cpu.py:38) 直接绑定实际 PV 文件，执行其自己的两个原生方法，移除了 dataset evaluator 专用 stubs。审查者又独立 CPU 确认。先前 GBK 读取 UTF-8 的原始失败也保留；它不是模型失败。

- [helper](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/selected_geometry_eval_diagnostics.py:28) 使用原 native_root_bbs 的 root/modify/pron/relation/other 公式和原降序 argsort，全 256 候选中只取真实 top1；框 IoU 采用原 kernel 和严格 (>.25/ .5)。完整 256 框覆盖、合格数量和首个合格排名只用于日志，不加入二次评分、裁剪或 GT 空间门控。
- 实际 PV 的 own Mask 是 [256,S]，text Mask 是 [1,256,S]，alpha 为标量；helper 用同一 query 和 superpoint→point 映射，记录自身与 alpha×text+(1−alpha)×own 的整数交并，并以 **2I>U** 表示严格 Mask IoU>.5。GT 取有效槽 0，native coarse、A 后 Mask 端点及当前 B final 框分别读取原输出。当前 GT 槽容量 132，语义和 query 轴为 256；并未以 128 替代完整候选轴。
- [eval hook](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/source_overlay_v4/train_dist_mod.py:217) 在原 evaluator 的 last_ 调用后每个 batch 追加一次。删除新增诊断语句后，完整 train_dist_mod AST 与原入口相等。两组共用诊断；网络/loss/匹配/资格/score/seed/RNG 和 R3 的 E0-inclusive 连续两次 strict 下降、reset、max8、best 排序规则保持。
- [finish](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/selected_geometry_eval_diagnostics.py:80) 要求9508行且逐行 top1 命中与原 last/bbs 总数一致。E1+ 对 E0 核对行号、场景、目标、表达哈希和 GT 框；只有点云与 root GT Mask 两个哈希均相同才计入 exact-input 修复/破坏。该标签不意味着所有模型输入和状态都已做完整字节指纹。日志不保存256原始数组或NPZ，GT诊断不回流到预测、训练或选模。
- [R4 stager](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/stage_eval_diagnostics_authorized.py:21) 只在尚无 preflight/normal 的本实验固定目录归档 R3 manifest/原入口，替换入口并增加一个 helper，将117项变为118项。v4 两个 launcher 与已审 R3 相比只替换 receipt/review 文件名；实际 pair 审查、fresh 初始化、单卡锁/设备准入和控制结束后才启动策略的条件保持。
- [observer](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/observe_study_authorized.py:21) 按已保存 launch 的固定 root 与实际 argv 核对进程，未来到期只读一次。显式 --collect-rows 且 status complete、controller/child 均消失才下载完整逐行文件。传输/收集失败不会生成成功 INTAKE 或自动重启；本轮仅静态审查，没有执行该观察。
- [新增终态权重元数据分支](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/observe_study_authorized.py:72) 仅在 preflight 完成且进程消失后，对每组唯一 best.pth 的大小/SHA 与实际 receipt 字段核对，返回 unscored engineering 标记；不会下载、加载、改写或删除权重，运行时也不哈希这些大文件。这不是权重晋级或精度证明。

实际审查 CPU 回执为 **eb8c36 / exit0**，使用 E:/anaconda/python.exe -B（Python3.7.6 / Torch1.13.1+cpu）。8个实际保存文件与4个嵌入片段解析及 argv quote-roundtrip 通过；重放检查器的写入只重定向至 review trace。独立纯方法样例使用256语义/query轴、132 GT槽，选中 query 5/200，并检查同分排序、native融合Mask整数、输入/RNG不变。两轮9508行、16,380,022字节仅为两个样例复制形成的**合成聚合**；改变点云的那项合成 repair 正确未计入 exact-input repair。完整工厂、神经网络、GPU、完整 evaluator/9508真实数据、optimizer 恢复和精度均未在本次执行。

CPU 后的最后一处 helper 更新也已绑定：只从 mapping.shape 断言删除 `and gt_mask.any()`，保留尺寸一致检查。原接口没有为此日志声明 GT 点级 Mask 必须非空；日志只记整数交并，不做除法，删去额外约束不需 fallback 或新分支。审查工具 **1675c0 / exit0** 核对两份 helper 的精确差异、Py3.7语法及 manifest SHA；最终 helper SHA 为 **44ca7699b1dd641c8a349f39b4be639962282ec0ae1857cc3fce5cb24ade258a**。先前完整 CPU 绑定的 c00ed422… 字节、工具回执及报告草稿都保留，未把这次静态最小更新冒充新一轮模型运行。

R1 FAIL、R2、R3 报告原字节及此前失败证据保持；初始 CPU 对43个绑定输入核对不变；之后仅两份 helper 的非空GT断言删除及 preparation SHA 更新共3项变化，其余40项保持，最终43项另存 INPUT_HASHES_FINAL.json。本地 R3 stage receipt 的时间为2026-10-12 19:30:41+08，记录117文件；这里仅读取保存回执，未复查远端。新增 R4 stage 和未来运行尚未由本审查执行。完整输入快照、收到的请求/补充、实际工具回执、报告和 FINAL_RESPONSE 位于 [run04 trace](C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012/source_review/.aris/traces/experiment-bridge/2026-10-12_run04)。

仍待实际验收：授权 R4 staging 的118文件回执；MCLN 终态与唯一 A100 锁内设备释放；两组真实原生两步/同输入/梯度责任/save+同进程restore及其 fresh 实际完整性审查；随后两组 fresh 正常训练、E0与每轮完整9508诊断对账、best/latest与停止证据。冷启动恢复仍 unsupported；本接口限定单 rank、ScanRefer、root槽0、256查询，不把 subset、Nr/Sr 或128查询当作已通过。

任务目标仍是同一模型 ScanRefer5705/4945（60/52）、超过同预算控制与保留5677/4920的证据、三项有效贡献和 Nr3D/Sr3D。CPU PASS 不缩小目标。新增诊断不证明所有任务冲突已消失；所选几何训练责任与原 G/no-object 语义职责的关系仍需实际结果，若只是小幅训练差异就相应收缩 novelty。

