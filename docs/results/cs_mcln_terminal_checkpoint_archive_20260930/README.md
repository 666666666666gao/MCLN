# CS/native终点恢复权重的本地归档

两份已结束的epoch21模型与优化器恢复状态于 **2026-09-30T11:14:53.711866+08:00** 完成本地归档，总量 **1605956944字节**；2026-09-30 11:18 CST再次独立读取本地全部字节并核对大小和SHA，均与传输前远端源校验一致。

| 实验 | 文件字节数 | SHA256 |
|---|---:|---|
| CS epoch21 latest | 813106668 | e6875c46cd64caeec518067214bc9d209e20021f3eb41dd9d89fce31cd9292f1 |
| native epoch21 latest | 792850276 | c054eeedf5072c944c7fd57dc94eb0fb8b3eca7687f28148d3f4903e6f99ae58 |

本地目录：`C:\Users\gb\.codex_mcln_checkpoint_archive_20260930`。模型元数据为epoch21、seed2027、batch12；CS有1175个状态条目和3个优化器参数组，native有1135个状态条目和2组。目录包含两个原始`.pth`和manifest；不是重新序列化的近似复制。

**远端latest尚未删除，当前没有因此释放远端磁盘空间。** CS/native best、E71和V99保护链没有改动。GitHub、交接文档和桌面results只同步小型manifest与核验记录，不提交权重。

这份归档使完成实验的终点可恢复，后续新训练仍需依据真实预检的best/latest文件大小及原子替换峰值核算空间。归档完成不代表新结构预检或性能通过；batch12完整诊断复核仍按11:45节点观察。
