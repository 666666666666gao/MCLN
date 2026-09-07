# 官方ScanRefer预训练组件的完整前向核验

此项只回答运行与输入是否真实可用，不训练、不使用正式验证集，也不报告REC精度。ScanRefer先行、V99保护线、Scan Mask底线、Nr/Sr REC要求均不变。

1. 使用已经校验的PV-Ground ScanRefer完整checkpoint：830036622字节，SHA `6f24c67cc3409f44befdef188f14383497a824d8ab309b8fd572a999ae8bdec3`。不重新下载、不部分加载、不用随机参数补齐不匹配层。
2. 完整构造官方`PVGround`，模型参数来自其checkpoint的Namespace：RGB点输入、BUTD预测检测框、256 Query、6层Decoder、对比对齐。保存的`config.model=MCLN`是该权重的旧类名记录，不据此误用当前MCLN核心。1234个state tensor必须strict load成功。首次严格加载实际只缺`text_encoder.embeddings.position_ids`：这是本机Transformers4.17会保存、作者4.40不保存的固定整数buffer。v2核对它逐项等于`arange(max_position_embeddings)`后，按[Transformers4.40官方实现](https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/roberta/modeling_roberta.py)注册为`persistent=False`，保留原值和严格加载；不填补任何学习权重，也不改checkpoint。
3. 真实输入沿用当前固定MCLN数据加载器和正确的`DATA_ROOT_mcln_meshsp/`，从已有29778-row fit列表依序选前四个不同物理场景。保留全36665条annotation供原生干扰物/unique计数，仅对实际读取的四句执行原解析器；关闭增强。6887-row模块holdout和9508-row正式验证不参与。首次导出因将已有Tensor误当numpy数组而失败，四行输出均未生成；v2修正类型处理，并免去已实测耗时数分钟的未使用句子解析。
4. 导出模型输入仅含原始50000个XYZ/RGB点、完整文本、预测对象框/类别/有效位和superpoint。目标ID只写到外部来源记录，GT框、GT Mask、GT Anchor与token标签不进入模型。
5. 使用作者DataProcessor与其原始float数组转CUDA方式，核对体素化后原始点的数量和顺序。固定seed2027；官方Gumbel采样即使eval也含随机性，此核验不更改该机制。四行分为两个batch，每batch两行。
6. 核对256个中心/尺寸、1024个seed、语义响应、对比特征与逐候选Mask的形状和有限性。记录正尺寸框数量、耗时、显存及全部候选原框，结束后核对所有权重和buffer未变。四行输出不用于选择方法、超参数或决定性能晋级。

环境需先有实际算子PASS及按文档独立运行的receipt。`scripts/export_pvground_train_fixtures.py`生成输入；`scripts/verify_pvground_pretrained_forward.py`执行严格加载及两个无更新前向。若失败，保留原日志并修复具体接口错误，不用此工程检查宣称网络有效。
