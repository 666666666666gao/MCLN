# R2 源码复核

**结论：WARN；源码准备 PASS；`scope=SOURCE_ONLY`；`scoped_blockers=[]`。** R1 的实际语法阻断已经闭合，本轮未发现仍待修复的范围内源码错误。WARN 保留真实运行门禁与审查身份边界，不把尚未执行的准备步骤称为模型失败。部署与精度验收均未通过，也未执行。

Requested **gpt-6-astra / max**；实际后端和 reasoning effort **UNATTESTED**；**same-family / provisional**。这是 R1 委派审查者的 R2 续审，不冒充新一轮全新上下文或跨模型家族审查。

**范围与保留。** 根目录为 `C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012`。复核五个 `source_overlay_v2` 实际文件、两个 overlay 生成器、common-loop CPU 证据；按追加请求静态审查 `native_selected_geometry_preflight.py`、`normal_run_controller.py`、两个 protocol、`prepare_execution_sources.py` 和 `stage_source_authorized.py`。R1 报告 `SOURCE_REVIEW.md/json`、原 `source_overlay` 失败字节和 run01 trace 均保留；没有覆盖生产源、活动 MCLN 或旧实验数字。本审查不认证活动 MCLN 当前状态，未轮询它。

**已闭合的具体问题。**

- **SRC-01 / R1：** `prepare_source_overlay_v2.py:17` 先将原 CRLF 规范为 LF，`:24` 使用 `write_bytes`。五个实际保存文件均被 Python 3.7.6 直接解析，全部 CRCRLF=0。旧 `pv_ground.py` 和 `losses.py` 的 continuation 语法失败没有被覆盖。三个旧 hook 文件的逻辑文本与 R1 相同；objective 完全同字节。两个生成器按其实际替换内容在内存中复现，结果与保存文件一致。
- **R2-TRANSPORT-01 / 已闭合：** 初版 `stage_source_authorized.py:87` 使用 `shlex.join`，指定 CPU 解释器 3.7.6 实测该 API 不存在。主任务说明真正 transport 原计划使用既有 Python 3.11 wrapper，因此本报告不宣称它在 3.11 失败。该处仍已直接改成 `' '.join(shlex.quote(part) ...)`，无 fallback/兼容分支。复核当前字节与完整远端 argv 往返均通过。修复前完整文件、差异和实测回执已存 run02。

**实际字节与关键接口。**

| 检查 | 结论与证据 |
|---|---|
| 五个 R2 overlay | `main_utils.py` 24881 B；`models/losses.py` 42048 B；`models/pv_ground.py` 28937 B；`native_model_initialization.py` 4712 B；objective 4410 B。全部直接 Py3.7 AST 通过；完整 SHA-256 见输入清单和 CPU JSON。 |
| native bbox / GIoU 与真实 targets | objective `:14,62–64` 使用当前 `last_center/last_pred_size` 和原生 `criterion.loss_boxes`。原 `models/losses.py:518–544` 的 center L1、0.5 size L1 与 diagonal GIoU 沿用，系数 (10,2)/7。新增项分母是实际 `len(indices)`；原 loss 分母保持。 |
| 当前末层索引与 GT | `models/losses.py:872,919,975–976` 连接本次真实 compact targets 与末层匹配；没有 frozen 旧匹配或预测伪 GT。objective `:20–30` 要求有效首槽 root=0，保护所有已匹配 Query，包括 root 和 other。 |
| 唯一 native 选择 | objective `:12–13` 对原生 bbs 排序每表达式只取一个 Query；`native_root_bbs.py:4–9` 沿用 softmax 后 root/modify/pron/rel-other。已有匹配不再接受新增目标，未匹配项只指向 compact root0，无重复扩充原 indices。 |
| Mask 支持资格 | objective `:32–48` 在点域计算 own 与 fused Mask；两者均以整数 `2*intersection > union` 严格判断，IoU=0.5 不过门。资格与选择停止梯度；Box IoU 不参与资格。 |
| factory → forward → get_loss | `native_model_initialization.py:66,75`，`models/pv_ground.py:283,616,619`，`models/losses.py:959,971–980` 已静态实接新旗标；eval 的 `self.training=False` 阻止执行新增项。没有新 GT 几何/Mask 推理支路、裁剪候选或第二评分。 |
| 参数与两组控制 | 两份 initialization 只在 geometry 旗标上不同；G 开启、旧 selected-mask 额外项两组均关闭。原初始化与 trainability 路径、1295 state 项设计保留，没有新参数模块。这里没有构造模型或认证真实 state 相等。 |

这些 objective/hook 结论承接 R1 的源码与 CPU 内核审查，本轮通过实际字节等价性确认适用；没有把 R1 的有限 CPU 探针升级为完整 native 工厂/网络执行。

**common loop 已真实写入共享入口。** 两个协议都指向同一 `PV-Ground/train_dist_mod.py`，该原生入口继承共享 `main_utils.BaseTrainTester`，没有第二套训练循环。

- `source_overlay_v2/main_utils.py:374–382` 限定 fresh start、每轮完整评估、max8，并先记录 E0、初始化 history、保存 E0 best。
- `:172–178` 从 history 末尾检查相邻 strict hit 的严格下降；持平或回升立即重置。E0 在 history 内，E0→E1→E2 两次下降即可触发，不是相对历史 best 的 patience。
- `:383–394` 保留真实 sampler epoch、loader generator seed、原生 train_one_epoch/evaluate_one_epoch，并先写指标和 history。
- `:167–169,395–398` best tuple 为 `(hits025>=5705 and hits050>=4945, hits050, hits025)`。先同一模型同时达 60/52，再 strict、再 loose；严格 `>` 保留平分的早期版本。每轮 latest 保存发生在 early-stop 之前。
- `:399–407` 在第二次连续下降后 break，或最多到第 8 轮；记录实际 history、保留指标和 reason。目标仍未宣布完成。
- `:151–157` 会将 history 通过 config 与 model/optimizer/scheduler/RNG 一起保存；`:133–145` 不恢复该 history，且 `:375` 明确拒绝正常训练 checkpoint resume。因此 **fresh-run loop 准备通过；正常冷启动续训不受支持**。没有据保存字段存在而认证恢复。

已复算真实存档 `EPOCH3_ANALYSIS.json` 的 strict 序列 **4920→4599→4625→4476**，末尾连续下降数应为 1，回升处确实重置。这是旧结果的只读政策例子，不能移作本实验结果。

新增 CPU 检查提取了上述真实 main 尾部 `:374–407`、实际 policy/record 函数；train、evaluate、checkpoint 明确使用桩。五类确定性 fixture 验证 E0 两次下降在 E2 停止、持平重置、回升重置、max8、目标优先/strict/loose 以及平分保留早者；同时核对每次 checkpoint 调用前 history 已更新、latest 在 break 前、日志和 stop history 相同。**这证明已抽出的控制流，不是完整 normal loop、网络、optimizer 或 checkpoint 状态执行。**

**追加执行入口只做静态审查。**

- `prepare_execution_sources.py:32–55` 由现有 native preflight 作最小文本适配；复现出的文字与实际保存 preflight 同字节。两协议 common arguments 与旧配对协议仅 max_epoch 3→8 的改变一致；arm、init path、geometry flag 是两协议唯三差别。提取真实 `parse_option` 解析成功：ScanRefer-only、B8、seed2027、Q256、6 decoder、core/backbone LR1e-6、A/B LR1e-5、WD5e-4、clip0.1、每轮验证、max8；不 frozen、不 checkpoint、不 joint_det。按数据契约每轮 floor(36665/8)=4583 次更新，8 轮上限 36664 次；这些是配置上限，不是已完成更新。
- `native_selected_geometry_preflight.py:95–104,153–184,195–200` 接原生 get_criterion/get_model/_compute_loss/train_one_epoch，检查新增损失对真实 last center/size 的直接梯度不越出合格位置；eval 初始几何、旧 mask 关闭、原参数组梯度/更新检查均为待执行源码。`:232–274` 保存后在同一进程重新构造并核对 model/Adam/scheduler/global RNG。该入口尚未调用，其未来结果也只可称同进程恢复，不能认证 normal cold resume。
- `normal_run_controller.py:12–21` 要求两组真实预检、fresh 初始化与 MCLN 已终态/释放设备的 admission；`:33–46` 将原生 train_dist_mod.py 作为子进程；`:47–65` 检查 E0 到末轮连续、9508 rows、stop history 与理由、best/latest 文件存在。它没有检查 checkpoint 内容或保留最优轮的 tensor 身份，并明确 `formal_result_audit_pending`。`admission.json` 的实际生成、串行 A100 锁与启动器仍未实现；本文件自身不取得 GPU 锁。
- `stage_source_authorized.py:13–16` 与本 JSON 的 source-only/pass/no-blocker 字段吻合。该门只够批准源码准备这一范围，不能作为 GPU 开跑条件。`:17–33` 读取既有 116 文件记录、5 个 overlay 与 6 个明确资产；集合合并为 117 个源文件。四项关键未覆盖本地源（train_dist_mod、native_root_bbs、旧 selected-mask、G semantic assignment）与原记录哈希一致。其余远端实际字节尚未读取。
- 嵌入远端段 `:34–76` 使用固定新目录，先断言不存在，再复制记录源并写 overlay/资产；源写入路径有目录包含断言；现有 manifest 的全部 116 路径都属于其声明原 model_source。只读既有环境 JSON 和源文件，解析 Python AST，写新目录清单；没有模型导入、训练/进程状态查询或 GPU 调用。
- transport `:77–91` 使用固定 OpenSSH argv、已有 askpass/known_hosts、StrictHostKeyChecking=yes、固定目的地与 shlex.quote 的远端 Python argv，凭据由环境继承，无凭据文本写入命令/报告。当前本地和嵌入远端 AST 及整条命令 quoting roundtrip 均通过。未执行 SSH，未认证远端现在的可用性或 host/device 状态。

**证据边界与未完成条件。** R1 的 objective CPU 证据仍是同一组 **8 个增强输入、两次观察共 16 条记录**，只存部分 cached mask slices，几何是 **saved Mask endpoint，非未缓存的 B final**。不能把它称为完整 Q256 mask/score forward、factory、全 get_loss 或精度验证。历史负实验叙述不代替任何新运行证据。

真正运行仍需：隔离 staging 与实际 import 验证；活动 MCLN 真实终态/设备释放和唯一 A100 admission/launcher；两组真实 native GPU 两步预检与同进程恢复；fresh 同源同起点完整 9508 E0；共同 normal loop 和 checkpoint/最优轮结果审计。需要 normal cold resume 时须另行真实实现与验证，当前不宣称支持。上述是**部署待执行条件**，不加入已闭合的源码 blocker 列表。

G 原有 semantic reassignment 采用它自己的 Box IoU>0.5 责任判断；新增项按 point Mask 支持资格可能监督未匹配但低 BoxIoU 的 Query，其几何责任仍可与 no-object/not-mentioned 和 contrastive 责任并存。不能宣称全任务冲突消失。当前至多是有明确假设的训练责任改变；若只产生小幅策略差异，应收缩 novelty，不能直接凑成第三个有效模块。最终任务仍是同模型 ScanRefer **60/52**、三项有效贡献和 **Nr3D/Sr3D**，并需相对共同 control 与保留强点的真实证据。

**实测与追踪。** 只使用 `E:/anaconda/python.exe -B`，Python 3.7.6 / 已装 Torch 1.13.1+cpu；未安装任何包。主 R2 验证 tool chunk `527f51` exit0；追加静态/argv 验证 `5b2e04` exit0。原始 shlex 检查 `fed8bd` 与审查脚本自身的路径失败/修正回执也完整保留，后者不计产品缺陷。0 SSH、0 GPU、0 NN forward、0 optimizer 更新。

`SOURCE_REVIEW_R2.json`、`CPU_VERIFICATION_R2.json` 和 `CPU_VERIFICATION_R2_ADDENDUM.json` 给出结构化结论。run02 保存完整收到的各次请求、全部 review 输出、输入路径/哈希、CPU 工具回执、失败前 staging 字节、当前追加源快照、路由/身份元数据及 FINAL_RESPONSE；run01 原样保留。输入清单的 D:/Program Files/UserCache 路径是 C:/Users/gb/.codex 的解析别名。
