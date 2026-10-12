# MCLN 正常训练源码审查：实际容量修订

**结论：PASS；blocking_findings=[]。最终容量门槛为 2,664,907,480 B。** 这是根据真实CPU测量和当前保存顺序得到的保守预算，覆盖尚未创建的Adam状态，不是为启动随意降低门槛。

时间：2026-10-12T14:28:41.760938+08:00。同一reviewer继续审查；same-family / provisional；实际模型、reasoning effort、backend均UNATTESTED。最初的fresh源码审查及第一次容量拒绝保留，本轮不冒充新的独立agent。

## 真实测量与容量计算

执行者保存的CPU-only收据显示：755个optimizer参数ID，731个已有state，24个缺失state；四组amsgrad均False。完整model张量598,498,856 B，其中冻结text_encoder为498,586,640 B，全部非RoBERTa张量99,912,216 B。已初始化的两份moments各97,089,656 B。测量脚本只CPU加载已完成工程checkpoint并在BytesIO内序列化，无模型构造、forward/backward/step或新权重文件。

全部非RoBERTa张量包括buffers，因此按其两倍199,824,432 B预算Adam moments是保守上界，覆盖当前尚缺的24个state；相对已有moments最多再增加5,645,120 B。未来不会解冻RoBERTa或更换优化器组、模型shape/dtype。

| 项目 | 字节 |
|---|---:|
| latest上界：model + 全部潜在双moments + 64MiB元数据 | 865,432,152 |
| best上界：model + 64MiB元数据 | 665,607,720 |
| latest原子替换峰值：2×latest + best | 2,396,472,024 |
| 额外日志预算：256MiB | 268,435,456 |
| 最终required_free_bytes | **2,664,907,480** |

替换best的另一峰值latest+2×best更小。64MiB/文件明确包括序列化、config/history、scheduler和RNG等非moment内容；实测完整工程文件的非model/moments开销900,866 B，model-only序列化开销433,514 B，因而该余量充足。正式最多8轮、两份保留权重和一个原子替换临时文件，没有逐轮累计权重。

清理收据记录两份已归档旧native/control重复best共释放1,230,047,504 B；历史free_after=2,391,838,720 B，加待删除的本次工程文件793,579,034 B，预计3,185,417,754 B，相对门槛尚余520,510,274 B。此为历史收据算术，实际准入仍现场读取disk.free。受保护的当前best和official/E71/G/A/B不在此次容量调整的删除路径中。

## 修订范围与失败行为

launcher只新增预算payload、用预算required_free_bytes替换固定4GiB，以及在admission记录预算。环境、源文件、工程实际两步PASS、GPU空闲检查和准确工程文件path/size检查保留；空间判断仍先于mkdir/write/unlink。第一次失败的FIRST_CAPACITY_REJECTION_LAUNCH_STDERR/STDOUT已逐字保留，原终态证据和清理收据已复制到inputs_capacity_r2。旧的更保守门槛本身没有破坏训练证据；本次是有实际容量数据支持的预算修订。

## 最终代码的既有结论

**严格初始化与工程隔离 — PASS。** train:46-50,61,115-119; initializer:11-47. E71 payload is checked, exactly nine obsolete SourceChoice tensors are removed, the remaining 1135 keys match the native factory exactly and load strict. A and B load their exact bound states, B verifies its A parent, total model states=1159. AdamW is new; no engineering optimizer/model is read by formal training.

**原生数据与实际对象接口 — PASS。** train:51-84,109-114; native TrainTester.get_datasets:2609-2613,2892-2970; dataset:101-119,240-258,585-597,1324-1372. split=train augmentation and joint ScanNet detection are native. Saved actual check receipt establishes 48655 rows and joint_det/detect_intermediate/augment_det/butd=True. butd_gt/butd_cls=False keeps detected boxes rather than GT proposals. Validation is split=val and never passed to optimizer.

**完整原生训练而非两步/三轮替代 — PASS。** train:70-76,172-193 invokes train_one_epoch for all 6081 native batch-8 batches when dataset length remains 48655, with max_train_batches=0 and accumulation=1. main_utils:7518-7851 runs forward, native loss, backward, finite checks, clipping, AdamW and scheduler. The original train loader drops the final 7 rows, as native drop_last=True specifies; this is a full native loader epoch, not a claim that all 48655 rows are processed each epoch. max=8; no fixed 3-epoch exit.

**训练图与原生损失 — PASS。** mcln:3167-3170 invokes portable geometry inside forward. portable_geometry:19-34 replaces actual final query mask and boxes. losses:3194-3238 consumes those final predictions in native Hungarian boxes/labels/masks/alignment loss. A has differentiable corrected mask logits; B has differentiable box mixing. native_mask_geometry:13-21 deliberately detaches hard membership. No new score head, G label target, six-source G core, V99 restore, or new candidate ranking is introduced.

**优化器、冻结与DDP — PASS。** train:118-135; main_utils:5777-5803; mcln:1364-1366. Native trainable core and exactly 57634 A/B parameters enter AdamW once; A/B are moved to LR1e-5, core/backbone LR1e-6, WD5e-4, clip0.1. Equality to the full requires_grad parameter set is asserted. RoBERTa parameter freeze is asserted. Single-rank NCCL/DDP uses find_unused_parameters=True and broadcast_buffers=False. Native model.train is used, preserving baseline dropout/BN semantics.

**真实E0与每轮9508评价 — PASS。** train:89-105,167-190; TrainTester.evaluate_one_epoch:3490-3638; main_utils._main_eval_branch:7854-7914. Full native evaluator executes, synchronizes counts, and the wrapper reads actual dets/gts at (last_, .25/.5, 1, bbs). gts must equal 9508 for both thresholds. evaluator:1739-1754 compares predicted boxes with dataset center_label/size_gts from :1992-2007 using native strict IoU>threshold. Counters are computed, not copied from prior 5473/4777.

**停止与同检查点选优 — PASS。** train:147-164,167-198 and native_decline_policy:6-17. E0 enters history. Acc@.5 must decrease in each of two adjacent transitions; ties and increases reset the streak, regardless of Acc@.25. Seven pure-policy CPU cases passed. Best is the lexicographic (hits050,hits025) of a single evaluated model, with earliest exact tie retained. latest is atomically written before best selection; the just-trained state is also saved before evaluation.

**保存状态与恢复边界 — PASS_WITH_EXPLICIT_LIMIT。** train:149-159,178-183 stores full model, optimizer, scheduler, epoch/update count, args/spec and Python/NumPy/CPU-Torch/all-CUDA plus train/test loader-generator RNG. Completed latest additionally stores history. Before-evaluation latest marks validation_pending; preceding completed metrics remain in native_metrics.jsonl. Best is a model/metrics artifact. The entrypoint has no cold-resume path, and no formal restore was executed; saved-state availability is not cold-resume verification.

**真实工程前置门禁 — PASS。** launcher:25-30 requires complete/exit0 and the exact two-update same-process-restore PASS status. check source:118-147 makes this status conditional on finite gradients, actual changed A/B parameters and nonzero core gradients. Saved status/result receipts match: 16 rows, 2 updates, 48655-row native dataset, positive internal/output A/B gradients and changed counts27265/29793. Same-process restore=true; cold_restart_verified=false. This review did not produce those GPU results.

**容量、权重退休与失败行为 — PASS。** Final launcher:18-19 reads the reviewed STORAGE_BUDGET into the existing payload; :42-43 checks current disk.free plus only the exact completed engineering file against 2664907480 bytes before mkdir/unlink; :52-54 preserves the budget in admission. Actual CPU receipt has model598498856B, frozen RoBERTa498586640B, other model tensors99912216B; all755 optimizer parameters are bounded rather than only731 initialized states. AdamW amsgrad=False requires at most2 moments per potential trainable tensor. L=865432152B and B=665607720B include64MiB metadata each; max(2L+B,L+2B)=2396472024B, plus256MiB logs yields2664907480B. No retry/fallback/other-weight deletion was added.

**原源码完整性与运行边界 — PASS。** All 736 original source-manifest entries match their recorded bytes/SHA. Reviewer changed no author source. New wrapper/controller/policy live only in normal_training; child uses -B and the declared env with source/pointnet2 prepended. Existing SHA review gate is reused without a new scheme. Actual local Python3.7.6 parsed/compiled 25 files plus embedded remote body; explicit local3.11.14 has shlex.join. Remote Python3.7/Torch1.10.2 is the existing declared environment, not re-probed here. Capacity R2 additionally parsed/compiled the revised launcher, local budget calculator, CPU measurement source and both embedded remote bodies with actual local Python3.7.6. Only launcher changed among32 previous audited inputs; budget and calculator add two static bindings. No source file was edited or imported by this reviewer.

## 复核与边界

本地真实Python3.7.6通过3个相关脚本和2段内嵌remote body的AST/compile；独立算术重算与STORAGE_BUDGET一致。此前32个静态输入仅launcher变化；最终门禁扩为34个输入。原训练代码、停止策略和checkpoint保存代码没有变化。一次本地复核因reviewer写短了归档文件名而失败，已记录CAPACITY_LOCAL_CHECK_ATTEMPT01.json；纠正文件名后实际复核exit0。这不是训练或容量计算失败。

- 本结论是最终静态源码准入 PASS，same-family/provisional；不是 GPU 正式训练、完整epoch、正式E0、性能提升或冷恢复通过。
- 本轮未调用SSH、nvidia-smi、Torch/native factory、模型forward/backward或训练入口。只读取已落盘的工程运行收据。
- 正式脚本没有冷启动/续训入口；进程失败后的目录和日志应先诊断，不能直接重跑并宣称恢复。工程同进程恢复不验证四worker/DDP完整续训。
- 完成的latest包含history；validation_pending latest没有history字段，既往指标须从同目录native_metrics.jsonl读取。若以后实现续训，应按这些实际保存字段验证，不应在本轮增加未经要求的恢复框架。
- 原生训练loader drop_last=True：48655行对应6081个batch、48648行/epoch；验证9508行不drop。
- A/B接入后的一次续训不能分离模块贡献与续训收益；PLAN已要求有保留价值后再做同起点同预算原生控制。PV 60/52、三项有效模块及Nr/Sr完整目标仍未达成。
- 容量为固定FP32模型/AdamW组、冻结RoBERTa及当前保存顺序下的保守规划上界，包含每份权重64MiB元数据和总256MiB日志预算；它不是已执行正式保存峰值，也不保证之后其他进程不占磁盘。launcher必须再次读取当前空间。
- 两步工程文件793579034B并非未来完整Adam状态上界：实际仍缺24个state。本修订已覆盖缺失状态；旧报告的四份工程文件预算数字仅是历史预算参照，不再作为最终门槛。

本review累计新SSH/GPU/NN/优化更新均为0；CPU checkpoint测量由执行者完成，reviewer只读落盘结果和测量源码。正式训练、实际正式保存峰值、冷恢复和性能提升都未由本轮证明。
