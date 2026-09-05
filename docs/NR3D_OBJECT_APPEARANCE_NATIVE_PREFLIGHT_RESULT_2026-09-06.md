# 对象外观原生预检PASS，尚无训练后质量结论

03:38:58从原受保护Nr3D parent启动，03:41:59实查controller.exit=0、进程退出、GPU1MiB。
固定16条fit/16场景、4批、12次原生forward全部完成；0参数更新、0checkpoint写入、
0模块留出或正式验证行。新增模块41472参数/5张量，当前显式边界版本。

零输出矩阵精确复现六层Query、采样索引、原Box及最终分数、两种原始Mask和alpha。
固定输出扰动后，前五层Query、seed、Text Mask和alpha逐元素相同；末层Query、
soft-token logits、contrastive Query投影、Box/最终分数与raw Query Mask均有变化。
这证明真实点外观可通过原生Decoder进入最终输出，不等于已经改善实例选择。

原生REC主项对零起点输出矩阵的梯度范数为0.020566–0.027852；完整loss对应范数
0.034911–0.159161。固定输出扰动后，4批中全部5张量的REC/完整loss梯度有限且非零。
这说明梯度路径连接正确，不是任务梯度冲突或泛化改善证据。
全部parent参数/buffer、源文件、数据、点Tensor及新增模块状态恢复检查PASS。

耗时31.563秒不含数据集初始化，峰值allocated1714117120 bytes。记录的native前向
为no_grad、addon前向保留autograd，不能把两者计时差直接当作公平推理延迟比较。

receipt SHA：`807105f3d77b4776c1f4d92093498b2d48591eb0cac2d0acc242ad27448d65ba`。
输入清单SHA：`7451d43c0644a03888f73c919ceea0cc4bb11570bfd69215bec32c48e3b377e9`。
完整证据在`refine-logs/object_appearance_native_preflight_20260906_v1/`。

下一步的两臂学习计划已固定于`docs/NR3D_OBJECT_APPEARANCE_PAIR_PLAN_2026-09-06.md`：
两臂共同有限更新最后Decoder的对象cross_d/norm_d；appearance另外学习真实点外观。
同2048fit、各1024步、6172模块留出，原生损失与候选协议不变；采用双参考REC门槛。
当前仅完成计划，配对runner及原环境检查尚待完成，没有启动学习或正式验证。
不加载C1/B/L1/P2/R1结果；C1已封存FAIL，三数据集保护结果与完整目标状态不变。
