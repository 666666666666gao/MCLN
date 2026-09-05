# 细尺度稀疏局部记忆：运行环境准备与验证边界

05:01原服务器只读检查：bdetr为Python3.7.11、Torch1.10.2+cu111、NumPy1.21.5、
Transformers4.17.0；没有spconv、OpenPCDet、MinkowskiEngine或torchsparse模块。
驱动550.90.07，A10040GB；当前PATH找不到nvcc，不等价于全机器不存在编译器。
证据refine-logs/sparse_environment_20260906/receipt.json。检查0安装、0GPU、0训练。

PV-Ground已核对的官方提交262e2592589baec7bb83a0d46aae6542d4ccedfb要求的示例环境
是Python3.12和spconv-cu124，并另外编译OpenPCDet。不能直接把整套安装命令用于
受保护模型当前依赖。它的体素主干最终仍返回关键点，Mask仍读取radius.2/nsample2，
因此整体移植也不能代替“细尺度特征直接到达Mask”的机制验证。
来源：[PV-Ground官方代码](https://github.com/AaNnWwTt/PV-Ground/tree/262e2592589baec7bb83a0d46aae6542d4ccedfb)。

存在更小的环境准备方案：spconv官方说明2.x不依赖Torch二进制，CUDA11小版本可混用，
提供Python3.7与SM80预编译支持。实际PyPI元数据确认spconv-cu114==2.3.6与
cumm-cu114==0.4.11各有CPython3.7/Linux x86_64 wheel，后者满足前者>=0.4.5,<0.5.0
依赖；版本、URL、SHA256及依赖列表保存在wheel_metadata.json。
来源：[spconv官方安装说明](https://github.com/traveller59/spconv#install)、
[spconv2.3.6](https://pypi.org/project/spconv-cu114/2.3.6/)、
[cumm0.4.11](https://pypi.org/project/cumm-cu114/0.4.11/)。
元数据存在和官方支持说明不是本机算子通过的证明。

第一步仅建立独立venv，继承原bdetr的已安装包，安装上述两个固定wheel及其缺失依赖。
下载两个wheel时按已读取的PyPI SHA256核验。原bdetr包清单前后必须相同，新环境的
Torch/NumPy/Transformers版本保持原值，并在CUDA_VISIBLE_DEVICES为空的进程中导入
spconv.pytorch。实际解析的附加依赖版本单独记录。此步0GPU、0原生模型forward、0
优化器更新；不安装OpenPCDet，不替换PointNet++或任何保护权重。

后续GPU算子检查需等当前Nr3D配对终态及完整性核验完成，取得原GPU锁后再单独执行。
至少检查稀疏坐标与特征对应、与相同权重稠密卷积在有效位置上的数值/梯度一致，以及
下采样—逆卷积坐标恢复。只有算子检查通过后，才设计真实输入细尺度局部记忆接入。
本文件不注册新结构训练，不改变O1终态门槛或复活B/C1失败设定。

算子检查入口scripts/check_sparse_runtime.py：固定seed41、两个合成batch、5×4×3
网格中去除坐标和模3为0的位置，采用8输入/16输出通道。固定相同KRSC转换权重，比较
SubMConv3d和稠密Conv3d在有效位置的输出、输入/权重/bias梯度；预设atol1e-5、rtol1e-4，
关闭TF32。再验证stride2稀疏卷积及同indice_key逆卷积恢复原索引/空间shape和有限非零
梯度。3次稀疏算子forward、1次稠密参考forward、3次backward，0数据集行/0原生模型/
0优化器更新。失败保留差异证据，不按结果放宽数值容差。

机制范围仍遵循用户方案：保留全局PointNet++，细尺度空间特征通过点—体素逆映射直接
提供给点/超点读取。尚未选择体素尺寸、网络深度、训练预算或正式部署方式；这些需
结合实际空间占用、边界保持与算力检查确定，不能由安装成功推导结构有效。

准备勘误，05:11：v1在首个wheel下载前因CERTIFICATE_VERIFY_FAILED停止，未创建venv，
0安装、0GPU。实查原Python默认CA文件/目录均为null，OpenSSL编译前缀仍指向不存在的
临时构建路径。显式使用/etc/ssl/certs/ca-certificates.crt后，对同一HTTPS下载URL的
证书和主机名校验通过，HEAD200。系统CA SHA
8e9482d461319198d2c5758d8ad29a1fb9dc0bc6850a24c57ffc83f6a8082cab。
保留v1原失败；v2独立目录仅为准备进程设置SSL_CERT_FILE和PIP_CERT到该已验证CA，
校验CA哈希，保持TLS证书验证与wheel SHA校验，不修改系统/原训练环境证书配置。

05:19补充：v2两个wheel下载/哈希核验及venv安装已完成，实际无GPU导入诊断确认
spconv2.3.6/cumm0.4.11可导入，Torch/NumPy/Transformers保持原版本。自动准备脚本
把sys.prefix绝对路径与传入的相对venv路径比较，导致最后验证报错；v2原controller1
仍保留。v3将运行根目录解析为绝对路径，直接复用v2已验证wheel，并将原环境完整包
清单在安装前落盘、子进程stdout/stderr在检查退出码前输出。没有改动库版本或关闭
校验；安装完成并不替代后续GPU数值检查。
