# V2终态实物校验准备

V2已在首批断言处退出，未产生终态权重；以下命令保留为未执行的准备记录。
参数顺序依据模型注册顺序核对：权重平均文件的键经过排序，不能用parent字典顺序
推定AdamW参数索引。此独立校验器目前只通过原Python3.7编译。

独立目录入口已在原Python3.7编译，0GPU/0优化器；V2终态权重尚未产生，尚未运行实物校验。
脚本检查原生16张量相对parent的实际变化、新17张量形状/有限性及输出投影非零、两组
AdamW参数顺序/形状/6687步/LR/动量。新非输出参数相对随机初始化的变化，只由运行内
初始快照核验和终态回执证明，此独立脚本不夸大为已经重新构建初始随机值。

终态controller0和receipt齐备后执行：

```text
CUDA_VISIBLE_DEVICES= /root/autodl-tmp/mcln_sparse_runtime_20260906_v3/venv/bin/python /root/autodl-tmp/mcln_sparse_artifact_verifier_20260906_v2/verify_nr3d_sparse_point_artifacts.py /root/autodl-tmp/mcln_sparse_point_pair_20260906_v2 /root/autodl-tmp/mcln_sparse_point_pair_20260906_v2/artifact_verification.json
```

质量门仍由独立逐行summarizer判断；该脚本不授权正式推广。
