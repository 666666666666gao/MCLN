# 原生框头终点到 Nr3D/Sr3D 的初始化准备

当前 ScanRefer 两臂仍在固定训练。本页记录格式接续准备，不改变正式晋级规则，也不代表 Nr3D/Sr3D 已开始训练或取得新成绩。

本轮终点仅保存16项框回归参数与优化器；原生 `main_utils.load_checkpoint` 读取完整 `checkpoint['model']`，键使用 `module.` 前缀。原有 `--model_only_initialization` 已支持跨数据集初始化，不需要修改加载器。

`scripts/export_native_box_transfer_initialization.py` 读取通过正式ScanRefer审计的固定候选，以受保护E71补全完整模型，仅覆盖16项中心/尺寸头参数。头内BatchNorm运行均值、方差和计数器仍来自E71；输出不包含旧优化器或调度器。工具重新核对既有9508行、正式指标、输入及权重SHA后才写入文件，不启动新评估，也不允许将GT-only控制臂代替候选。

在正式结果通过后，于远端项目根目录执行：

```bash
CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/bdetr/bin/python \
  -m scripts.export_native_box_transfer_initialization \
  --formal-directory /root/autodl-tmp/mcln_scanrefer_native_box_transfer_official_20260907_v1 \
  --output /root/autodl-tmp/mcln_scanrefer_native_box_transfer_official_20260907_v1/native_model_initialization.pth
```

新数据集原生训练入口使用此完整模型，并显式指定：

```text
--model_only_initialization --checkpoint_start_epoch 1 --start_epoch 1
```

这三个选项是初始化接口，不是完整训练命令。Nr3D/Sr3D的数据协议、REC监督、更新范围和固定预算须在真实ScanRefer终点通过后绑定，并做实际训练数据的梯度检查。ScanRefer完整V99过线并不证明导出的原生网络单独达到V99，因此主表中原生网络与完整系统仍分开报告。

## 已完成检查

- 原Python3.7/Torch1.10.2环境8项单元测试通过：正确覆盖、保留冻结参数与BN缓冲、清除旧优化器，以及错误来源、缺失/额外参数、形状/类型不匹配、控制臂拒绝。
- 使用真实E71和明确标记的合成框头差值，实际调用Nr3D/Sr3D原生模型工厂与checkpoint加载器；每个入口1144项完整state逐值一致，16项改变正确装入，其余1128项保持E71值，优化器为空，调度器不变，起始epoch为1。
- 合成格式测试实际训练步数0、GPU前向0、正式样本0。临时599063649-byte完整模型在核对SHA与路径后删除，没有持久化测试权重。
- 测试脚本曾在加载模型前遇到暂存`scripts`包遮蔽和切换cwd后相对路径失效；仅修正检查脚本的导入范围和绝对目录，原错误日志保留。未修改训练源码或环境。

真实训练终点尚未产生；正式通过后的导出CLI、实际终点在两个数据集上的加载及GPU训练检查尚未执行。不能把本页的格式准备称为跨数据集效果验证。
