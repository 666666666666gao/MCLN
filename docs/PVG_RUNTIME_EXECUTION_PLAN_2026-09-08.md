# PV-Ground 隔离运行环境与算子核验

## 状态与来源

2026-09-08，本计划仅准备官方ScanRefer预训练组件的运行条件，0训练步、0正式评估行。上一轮CPU权重读取已完成，但还没有完整模型前向结果。

当前环境spec：`configs/pvground_runtime_env_20260908.json`，按sorted-key compact JSON计算SHA256：`966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c`。PV-Ground提交`262e2592589baec7bb83a0d46aae6542d4ccedfb`，OpenPCDet提交`233f849829b6ac19afb8af8837a0246890908755`。源码由固定Git tree逐文件验证Git blob SHA；依赖由PyPI元数据SHA验证。

### env: pvg-runtime@c73eed09

- how：SSH服务器的`/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv`。
- base：bdetr Python3.7.11/Torch1.10.2+cu111；通过明确的`.pth`复用已有`mcln_sparse_runtime_20260906_v3/venv/lib/python3.7/site-packages`内spconv2.3.6/cumm0.4.11。
- tier：A100 40GB一张，CUDA11.6编译，SM80；最多4个编译任务、数学库单线程，使用既有`mcln_v99_backbone_gpu0.lock`。
- 新包：networkx2.6.3、imageio2.31.2、tifffile2021.11.2、PyWavelets1.3.0、scikit-image0.19.3、SharedArray3.2.4；离线、无依赖重解、只安装到新venv。
- CUDA扩展：未改动的OpenPCDet pointnet2_stack和roiaware_pool3d源码；未改动的PV-Ground pointnet2扩展。原生bdetr与旧稀疏venv不安装或升级包。
- weights：复用已核验的`/root/autodl-tmp/mcln_pvground_scan_checkpoint_inspection_20260908_v1/PV-Ground_ScanRefer.pth`，SHA `6f24c67cc3409f44befdef188f14383497a824d8ab309b8fd572a999ae8bdec3`。
- validated：失败。06:31构建退出1，OpenPCDet尚未完成编译；该spec及原始日志保留。

### env: pvg-runtime@5e83ce8f

复用以上源码、权重、硬件及已经安装的运行依赖，只将新隔离环境的setuptools从ensurepip自带47.1.0对齐为原bdetr已使用的58.0.4。这个构建依赖写入spec首个pip phase；原bdetr不修改。

实测47.1.0内嵌的packaging Version没有major属性，触发Torch1.10.2源码的NaN比较，使CUDA11.6/11.1的小版本差异被误判成大版本差异。原bdetr58.0.4的同一属性存在。修复不修改或跳过Torch版本检查，不伪造nvcc，保留上游小版本警告，仍须通过实际CUDA核验。依据为[PyTorch v1.10.2原始实现](https://github.com/pytorch/pytorch/blob/v1.10.2/torch/utils/cpp_extension.py#L769)及两个解释器的现场probe。

此次接续仅重建未完成的CUDA扩展；不重装完成的运行依赖、不重新下载830MB权重。setuptools官方wheel为816539字节，SHA `69cc739bc2662098a68a9bc575cd974a57969e70c1d58ade89d104ab73d79770`。独立日志为`build_repair_v2/run.log`和`controller.exit`。

- how：同一专用runtime目录的显式构建修复，接续脚本`resume_compile_v2.py`。
- validated：三项CUDA扩展均完成；采样、分组、梯度和RoI核对通过，体素参考断言失败，构建控制器退出1。

### env: pvg-runtime@966235b2

二次probe保存了全部8个合成点：其中2个边界点在float64参考与float32算子间相差一格，float32逐项参考与算子8/8完全相同。witness仅修正参考计算的dtype，保留原点、精确整数坐标断言、全部其他检查及容差。spec新增witness源码SHA和参考精度，三项已编译二进制及所有运行包复用，不重编译、不修改算子。

- how：`finalize_runtime_v3.py`核对原包清单和三项扩展，重跑完整固定witness；记录原v1/v2退出1的历史。
- validated：06:50:23 CST完整witness通过，分组最大误差0、梯度精确匹配、体素坐标与原点顺序通过，原bdetr包清单不变；仍待下述文档独立执行。它只证明运行环境，不证明完整预训练模型前向或任何REC指标。

## 有证据的最小移植范围

仅删除PV-Ground的`pv_utils.py`中未被使用的`BasicBlock2D`导入。该导入会执行OpenPCDet整个models注册，拉入17种无关检测器；全文查找确认没有第二处引用。运行目录保留原文件SHA、修改后SHA与删除行。没有改attention、采样、回归、Mask或权重。

PV-Ground自身`utils/scatter_util.py`是纯PyTorch实现，不需安装torch_scatter。OpenPCDet common_utils直接导入SharedArray；prepare_data直接导入skimage，所以上述依赖有实际代码依据。OpenPCDet的版本文件由构建脚本写入源码提交号；算子单独编译避免注册无关检测模型。

## 固定检查

`scripts/verify_pvground_runtime.py`检查两批合成点：FPS对照CPU最远点采样，stack半径分组对照逐点结果及精确特征梯度，空邻域归零，RoI点内判断对照轴对齐几何，体素ZYX坐标及batch合并后原始点顺序。该检查还实际导入完整PVGround类，但不实例化或前向模型。原包清单构建前后必须逐项相同。

构建使用screen后台持久日志`run.log/controller.exit`，估计5—10分钟；异常时保留实际日志，修复具体原因，不因一次观察超时重启。构建结束后才判断CUDA witness，随后按技能要求进行独立的文档执行。

## 按文档独立执行命令

只有原构建已成功、`build_receipt.json.status=pass`时执行。以下为Windows PowerShell中的完整命令；内部SSH凭据从当前已授权Goal只读取得，不打印或保存凭据：

```powershell
uv run --no-project --with paramiko python -u C:/Users/gb/.codex/tmp/run_mcln_authorized_20260908.py C:/Users/gb/.codex/tmp/mcln_remote_script_20260907.py C:/Users/gb/.codex/tmp/run_pvg_runtime_documented_witness_20260908.py --out C:/Users/gb/.codex/tmp/pvg_runtime_agent_witness_20260908.json
```

命令只在已有环境重跑同一个固定算子witness，写入独立`agent_kernel_receipt.json`，不修复、不安装、不训练。期望退出0，包含`PVG_RUNTIME_WITNESS`，并由包装脚本核对receipt。新上下文审核属于同模型家族的文档执行检查，不声称跨模型独立验收。

运行环境检查通过后，下一步才是严格加载完整权重、核对真实ScanRefer训练输入并执行无更新前向。随后才决定预训练组件的训练对照；完整三数据集目标和Scan先行规则保持不变。
