# 细尺度稀疏局部记忆运行环境：准备及GPU算子检查PASS

本次完成运行条件验证，尚未实现或训练体素局部记忆模型。0数据集行、0原生MCLN
forward、0优化器更新；不代表REC、Mask或跨数据集指标提高。

使用独立venv：/root/autodl-tmp/mcln_sparse_runtime_20260906_v3/venv。
保留Python3.7.11、原Torch1.10.2+cu111、NumPy1.21.5、Transformers4.17.0；
增加固定spconv-cu114 2.3.6/cumm-cu114 0.4.11及其缺失依赖。两个wheel按PyPI SHA256
核验。v3安装前完整原环境包清单已落盘，安装后逐项相同；原Torch从bdetr包路径导入。
实际新增依赖版本在runtime receipt完整记录；未安装OpenPCDet或替换现有主干。

准备失败均保留：v1下载前遇到原Python无有效默认CA路径；指定并核对系统CA后TLS
证书/主机名校验通过。v2安装成功，验证脚本把绝对sys.prefix与相对venv比较而失败；
独立导入诊断通过，v3修正根目录解析后完整PASS。未禁用TLS校验，也未增加自动重试
或运行时替代分支；v1/v2的controller1不改写为0。

在Nr3D对象外观配对完整结束、独立复算和实物核验通过后，取得原GPU锁，执行固定
合成算子检查。A100上两批80个活跃体素，8输入/16输出通道，关闭TF32，比较相同
权重SubMConv3d与稠密Conv3d的有效位置输出和梯度。预设atol1e-5/rtol1e-4未改变。

| 数值比较 | 最大绝对误差 |
|---|---:|
| 输出 | 3.57627868652e-07 |
| 输入梯度 | 1.78813934326e-06 |
| 权重梯度 | 7.62939453125e-06 |
| bias梯度 | 1.07288360596e-06 |

SubM索引保持原样；stride2稀疏卷积配合同indice_key的逆卷积恢复原索引和空间shape。
输入、下采样及逆卷积权重梯度均有限且非零。总计3次稀疏算子forward、1次稠密参考
forward和3次backward，无参数更新。记录算子段耗时6.629189秒，不含完整进程初始化；
Torch allocator峰值214528 bytes，不含库自行分配的全部显存，不作为真实场景效率结果。
05:25:47确认controller0、进程结束、GPU1MiB。首次本地observer因收集脚本文件名错误
未能查询，修正后直接读取同一次GPU结果，没有重启或重复GPU检查。

运行环境manifest SHA：`c648922c6d07414f556701f64955aca5e400f0359040898fd6b41fe1ad81866d`。
运行环境receipt SHA：`6f048d383b59e3625d8cde4f408f4913ead6dbafd5ca2dc3be99643cf7a11484`。
GPU检查manifest SHA：`6a93dcf47928b235a3cd919cc653143de412d896c8598228a63c47a58c07aee0`。
GPU检查receipt SHA：`b3d59154c4b7bd52b4f59c61ae028d825cca06f6360190ef15d04b6df0d0ab2b`。

入口scripts/check_sparse_runtime.py；依据与失败修正见SPARSE_LOCAL_MEMORY_RUNTIME_PLAN_2026-09-06.md。
证据refine-logs/sparse_environment_20260906、sparse_runtime_20260906_v1/v2/v3及sparse_kernel_check_20260906_v1。

下一步才是固定真实训练输入的点—体素逆映射、空间占用与边界保持检查，再确定细尺度
局部特征到超点/对象读取的最小接口。不能把已有SA1插值B实验或本次算子检查写成体素
主干验证。O1、B、C1、L1失败结果保持封存；三数据集正式结果不变，完整目标仍active。
