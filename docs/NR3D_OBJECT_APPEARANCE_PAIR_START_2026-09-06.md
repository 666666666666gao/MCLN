# 对象外观原生配对学习启动，首批检查PASS

运行v2于04:00:37启动，screen=mcln_object_appearance_pair，PID24914。
04:03:39实查进程存活，GPU10363MiB，尚无controller.exit，已通过首批检查并进入
完整6172行起点评估。当前没有完整起点identity或训练后留出质量结论。

两臂实际共享6张量/333504个参数，即最后Decoder的cross_d和norm_d；appearance
额外学习5张量/41472个真实点外观参数，总374976。原手算曾多计576，导致staging v1
在CPU计数检查停止：0GPU、0更新，pytest尚未执行。真实parent张量核对后仅修正
数量、断言和计划勘误；可训练层、结构、数据、预算、质量门槛均不变。v1原文件保留。

v2原Py3.7编译、7项终态评估测试、612源码/724数据文件和parent/预检证据检查PASS。
实际首批两臂loss均16.18368148803711；6个共享参数张量梯度逐元素相同，新增输出
矩阵梯度范数0.1120738313，其余4个新增张量梯度在零输出起点为零，符合链式求导。
首批0优化器更新；两臂输出、冻结状态一致。此前固定扰动已验证5个新增张量均连通。

预算按已固定计划：同2048fit/262场景、两轮B4、各1024更新；原生loss不变，
fresh AdamW lr1e-5、wd.0005、clip.1、无增强、同实际batch。6172条/98个模块留出
场景被主干见过，不能当作正式或新场景泛化。末层Query/Box/分数允许变化，前五层、
采样、Text Mask及alpha逐batch检查固定；保留原合法性与REC/Mask选择协议。

appearance终态相对native终态和保护起点均需REC@.25净增至少10，且REC@.50、两个
Mask命中阈值及mIoU不下降。没有中间质量选权重、自动延训或正式升级。
本次修改增加真实点外观证据，不是复活R1的框/类别重评分，也未整体替换主干。

Manifest SHA：`5f467d81155c03fa939f898a8d15a5263a98b407c31c342a72223b931b5a844f`。
远端目录`/root/autodl-tmp/mcln_object_appearance_pair_20260906_v2`。
入口`scripts/run_nr3d_object_appearance_pair.py`，终态独立复算
`scripts/summarize_nr3d_object_appearance_pair.py`。原生预检与计划已在3abf270，
配对代码和CPU证据在9911008；三数据集保护结果未更新，完整目标继续active。

启动检查安排于+180秒，已完成；后续observer首次04:20:37，间隔240秒，按本次
实际阶段速度更新ETA。参考同环境前次预算，完整终态初估05:10–05:30；这不是实测
终态承诺。不得把首批梯度通过写成训练有效。
