# PV-Ground隔离环境与完整预训练前向：工程通过

2026-09-08 07:01:38 CST，官方ScanRefer checkpoint完成严格加载及四条真实训练输入的完整前向，终态退出0。此项是运行能力检查：优化步数0、正式评估行数0、没有新增训练权重，不能作为精度增益或正式模型晋级。

## 环境与实际修复

隔离目录为`/root/autodl-tmp/mcln_pvground_runtime_20260908_v1`，最终spec SHA `966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c`。沿用bdetr Python3.7/Torch1.10.2+cu111及已有独立spconv2.3.6/cumm0.4.11；没有升级受保护环境。完整PV-Ground/OpenPCDet源码提交分别为`262e2592589baec7bb83a0d46aae6542d4ccedfb`和`233f849829b6ac19afb8af8837a0246890908755`。

首次构建因新venv的setuptools47.1.0内嵌packaging缺少Version.major，把CUDA11.6/11.1的小版本差异走成大版本错误。只将新venv的构建工具对齐到原bdetr58.0.4，未修改PyTorch检查，保留其原始小版本警告。两项OpenPCDet扩展及PV-Ground pointnet2扩展均实际编译，原bdetr包清单前后完全相同。该检查逻辑见[PyTorch v1.10.2官方源码](https://github.com/pytorch/pytorch/blob/v1.10.2/torch/utils/cpp_extension.py#L769)。

随后固定witness中的两个体素边界点暴露了参考计算dtype错误：float64参考与float32算子相差一格。修正参考为float32后保留精确坐标断言，完整witness通过：FPS一致、两处空邻域归零、分组最大误差0、特征梯度精确一致、RoI成员判断和原始点顺序一致。没有改算子或调宽容差。

独立新上下文按文档逐字运行一次也退出0，witness用时15.204秒。环境执行PASS，总体保留WARN：本地`--out`以.json为扩展名，内容实际是两行sentinel日志，不能直接当JSON解析。原始文本、有效的独立结构化报告、远端有效`agent_kernel_receipt.json`均分别保留。审核为gpt-6-astra/max、same-family、provisional；不冒称跨模型独立验收。

源码只删除了一条没有使用的BasicBlock2D导入，避免导入OpenPCDet整套无关检测器；参数与算子源码不改。记录的完整模型加载移植另见下文。

## 严格权重加载

沿用唯一已下载权重，830036622字节，SHA `6f24c67cc3409f44befdef188f14383497a824d8ab309b8fd572a999ae8bdec3`，epoch81。实际构造PVGround；checkpoint的旧`config.model=MCLN`字段没有被当作当前MCLN核心类名执行。

第一次严格加载只有`text_encoder.embeddings.position_ids`缺失。现场旧Transformers4.17会保存该固定整数buffer，而[作者版本对应的Transformers4.40实现](https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/roberta/modeling_roberta.py)将其注册为`persistent=False`。v2先验证其精确等于0—513整数序列，再通过公开register_buffer接口匹配保存方式，原值不改。

之后1234个state tensor严格加载，missing/unexpected keys均为空；实际模型参数总数152605243。没有用strict=False，没有随机补学习权重，没有改checkpoint。两批前向后所有已加载state与该固定位置buffer逐项未变。

## 真实输入与前向结果

输入来自已有29778-row fit列表，依序选择四个不同物理场景。保留全部36665条annotation供原生干扰物和unique计数，仅解析这四条实际使用的表达。原首次导出将已有superpoint Tensor误作numpy处理而失败，未产生任何fixture；类型修正后的v2在48.229秒完成。没有读取6887-row holdout或9508-row正式验证。

使用正确mesh-derived superpoint目录、原50000个XYZ/RGB点及BUTD预测检测框。GT目标ID仅在来源记录中，GT框/Mask/Anchor/token标签不进入模型。现场固定源与官方源的`_get_pc`、`_get_detected_objects`、`_get_scene_objects`函数语法树完全相同；这只是三个函数的对照，不声称整个项目源码相同。

| 训练行 | 场景 | 检测框数 | 体素数 | superpoint数 | 三维尺寸全正的原始Query框 |
|---|---|---:|---:|---:|---:|
| 0 | scene0000_00 | 54 | 49515 | 667 | 215/256 |
| 173 | scene0001_00 | 22 | 42430 | 710 | 198/256 |
| 237 | scene0002_00 | 47 | 45464 | 2329 | 216/256 |
| 455 | scene0004_00 | 13 | 41822 | 1441 | 198/256 |

四行分两批、每批两行，seed2027；官方eval仍含Gumbel采样，本次没有改其随机机制。各行输出256个原始框、1024个seed、语义/对比特征及逐Query/Text Mask，所检查张量均有限，体素化前后原始点顺序完全保持。两批GPU前向分别0.698秒和0.236秒，最大allocated显存1269432320字节；这些小样本数字不是完整系统性能基准。

本地另用标准库解码四个NPY、复算1024框有限性与尺寸计数，并核对源码、fixture、环境spec、严格加载及包清单，全部通过。827/1024个框三维尺寸全正，197个含非正尺寸；没有对它们补框、取绝对值或调新阈值。后续必须明确原生最终分数与正式evaluator合法候选规则，不能把256个原始Query都称为有效候选。

## 资源与下一步

07:08:18 CST复核：E71、V99和官方PV-Ground权重SHA不变，GPU已释放至1MiB；服务器剩余6285197312字节。没有新增训练权重，也未重新下载上轮已清理的830MB本地权重中转副本。

本项证明完整预训练组件可在真实输入上运行；尚未验证完整训练反向、原生最终评分/候选过滤或任何正式REC成绩。下一步先核对这两条训练与决策接口，再定义ScanRefer从现成权重起步的短周期对照。不会把下载作者模型包装为原创，也不把旧V99成绩贴到它名下。

受保护ScanRefer仍58.6033/50.4523，Mask59.8443/52.3349/45.9303；Nr3D与Sr3D正式结果未变化。Scan正式达到同一V99 REC保护线及原MCLN Mask底线后尽快转Nr/Sr REC，仍不等待59/51伸展目标全部达到。

证据为`refine-logs/pvground_runtime_20260908_v1/`、`pvground_train_fixtures_20260908_v1/v2`、`pvground_pretrained_forward_20260908_v1/v2`。原失败源、日志与退出码保留；v2前向receipt的终态为PASS，旧v1退出1不被覆盖。
