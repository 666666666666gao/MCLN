# 显式AABB边界修复的完整输入复核

v1全511场景/16181有效槽审计发现scene0054_00对象79为空。数值诊断确认：
abs(point-center)<=half_size的float32舍入排除了点16592，而该点位于显式lower/upper
边界内。改用显式AABB比较可取到1个已有点，不需要增加容差、扩大框或借用其他点。

外观模块新增单一box_crop_mask函数，直接比较xyz与center±half_size；模块与本审计
调用同一个函数。其余编码、归一、参数、框和有效性掩码不变。实测案例加入CPU回归
测试；原服务器环境5个测试PASS。保持非空合同，不加入空裁剪fallback。

复核同511个场景的全部有效槽，逐场景点hash、框hash、对象索引必须与v1相同，
逐对象重新计算旧abs裁剪点数并匹配v1。新Torch predicate还须与NumPy显式边界逐点
相同。记录新增/移除的点成员次数及受影响对象数，检查新空裁剪和非正轴数。
这个对照不根据GT实例成员挑选点，也不调整任何几何阈值。

脚本仍为scripts/audit_nr3d_all_object_crop_inputs.py，v1运行目录和Git版本保留。
v2使用新的独立目录、输入清单和receipt；0原生模型/GPU forward、0参数更新、
0正式验证行。输入合同通过之后，才准备原生Decoder GPU预检及学习对照。
结果仅适用于已锁定的无增强Nr3D训练输入，不外推其他数据集或增强后的对象框。
