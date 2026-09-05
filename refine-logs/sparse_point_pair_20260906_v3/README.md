# Sparse point memory V3：运行与证据入口

完整配对计划见 EXPERIMENT_PLAN.md；当前科学状态由实际controller/receipt决定。
启动07:17:29 CST，PID28416。完整首批与6172行起点复核通过后才执行每臂6687次更新。
不要将启动、梯度正常或模块留出结果写成正式泛化提升。

source_snapshot.zip保留input_manifest.json列出的全部11份源码/计划/测试原始字节，
逐项解包SHA复核通过。归档SHA：
`5a4c3b96430d4f35aec4b81d8f26c42a45b2b20563e2a09c536b6ef5de41823f`。
普通Git路径中runner、summarizer和summary测试三份文件发生CRLF/LF转换；换行归一化后
内容相同。要核对本次运行的原始哈希，使用此归档及published_source_identity.json。
它是源码快照，数据、受保护权重和稀疏环境仍由manifest记录的独立路径及哈希约束。

终态controller0与receipt齐备后，使用已冻结在运行目录中的两个独立入口：

```text
CUDA_VISIBLE_DEVICES= /root/autodl-tmp/mcln_sparse_runtime_20260906_v3/venv/bin/python /root/autodl-tmp/mcln_sparse_point_pair_20260906_v3/scripts/verify_nr3d_sparse_point_artifacts.py /root/autodl-tmp/mcln_sparse_point_pair_20260906_v3 /root/autodl-tmp/mcln_sparse_point_pair_20260906_v3/artifact_verification.json
CUDA_VISIBLE_DEVICES= /root/autodl-tmp/mcln_sparse_runtime_20260906_v3/venv/bin/python /root/autodl-tmp/mcln_sparse_point_pair_20260906_v3/scripts/summarize_nr3d_sparse_point_pair.py /root/autodl-tmp/mcln_sparse_point_pair_20260906_v3 /root/autodl-tmp/mcln_sparse_point_pair_20260906_v3/summary.json
```

两入口目前只完成原Python3.7编译/相应CPU检查，终态实物校验尚未执行。
质量必须同时胜过原生终点和受保护起点：mIoU至少+0.2个百分点、两个Mask命中阈值不下降。
通过该筛选也不自动推广正式输出；失败保留完整负结果，不改门槛或挑选中间最佳。
