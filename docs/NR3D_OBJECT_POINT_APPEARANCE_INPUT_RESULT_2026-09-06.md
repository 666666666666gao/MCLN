# 对象点外观输入检查：16行通过，完整输入发现一例数值边界问题

原型41,472参数/5张量的4个CPU合成测试PASS。随后固定16条fit、16场景、683有效槽，
进行了32次真实输入appearance CPU forward（零输出及固定扰动）：输入点hash、框、
原始裁剪点数均与既有审计相同；输出有限，padding槽为零，最终参数恢复。
0次MCLN主模型forward、0GPU、0optimizer，不构成原生接入或质量证据。
实际输入检查计时26.662秒，不含数据集初始化。

完整Nr3D训练输入检查覆盖32919表达涉及的511个场景、16,181有效对象槽。
当前无增强路径的每条表达都复制缓存orig_pc并拼接原生RGB，没有重新采样；
逐场景执行时确认实际XYZ/RGB与该缓存表达一致，因此无需重复相同场景数万次。
源码/data hash通过，0正式验证行、0模型forward/更新。审计完成，当前非空合同FAIL：
1个空裁剪、0个非正尺寸槽。

唯一空槽为scene0054_00的对象79，输入框尺寸约0.002496/0.044967/0.002518米。
进一步读取同一缓存确认它有2个采样实例点（成员仅用于诊断）。
原float32 abs(point-center)<=half_size得到0点；显式lower/upper边界得到1点；
转为float64做原abs比较仍0点。1 ULP扩框诊断得到2点，但不据此采用扩框。
采样点16592在显式边界内，却在abs形式的z轴超出半尺寸约5.96e-8米。
说明数值舍入使数学等价的裁剪表达不同，不是原始场景完全没有观测。

下一步仅将外观原型改为显式AABB边界比较，加入这个实测案例的回归测试，并重新审计
全部511场景。保持输入框、点云与GT协议，不引入容差、扩框或替代点。
此处记录的是修改前v1证据；原型尚未用于原生MCLN GPU前向或训练。

证据：refine-logs/object_point_appearance_cpu_20260906_v1/、
refine-logs/object_point_appearance_inputs_20260906_v1/和
refine-logs/all_object_crop_inputs_20260906_v1/。

完整审计receipt SHA：`057e5a30bcf71a4e62201abaa4be7c508e34917d88ff37a4a2404d2cff6a0511`。
原型v1 SHA：`2d605badf1d8cccf31aa5a0f457fdd7fde4a1ffce0f3f86b4b13861db58b5ed4`。
