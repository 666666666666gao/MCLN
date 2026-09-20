# EG-3DVG Nr3D预训练迁移结果

## 20.248 EG Nr3D零更新迁移正式完成；适配预检接口错误修正后独立重启（2026-09-20 18:28 CST）

作者ScanRefer epoch69权重在Nr3D7899条的完整评估于18:19:01完成，1579.74秒；CPU独立复算18:19:18通过，preflight/formal/audit/controller均exit0，所有1276模型状态保持不变，0优化步。规定主输出last/bbs，不按结果切换诊断头。

| 输出 | @0.25命中 | @0.50命中 | Acc@0.25 | Acc@0.50 |
|---|---:|---:|---:|---:|
| bbs | 3508/7899 | 2909/7899 | 44.410685% | 36.827447% |
| bbf | 3492/7899 | 2897/7899 | 44.208128% | 36.675529% |
| bbs_unfiltered | 3240/7899 | 2778/7899 | 41.017850% | 35.169009% |
| bbf_unfiltered | 3210/7899 | 2766/7899 | 40.638055% | 35.017091% |

这是ScanRefer预训练权重的跨数据集零更新起点，不是Nr3D专用权重复现。bbs相对当前受保护4475/3759少967/850个命中，尚未达到4726/4059目标。输入沿作者Nr协议，包含场景GT实例框及预测类别；主分数保留作者对象支持过滤。未过滤两项仍使用同样对象输入，只作诊断，不称单阶段或GT对象无关推理。

rows SHA 1278b8188111c2fe02d6f3509e3bf6f4105093d525d8f38a7ae200696b5cf97b；candidates SHA 217bedf262c9e9d389652f0157ccc551706937a0c8958e107bc0326ffd887728；独立重算最大IoU差5.26387219e-06，所有选中候选及四模式命中一致。完整文件保留远端，轻量回执、审计、原始日志保存于refine-logs/eg3dvg_nr3d_transfer_20260920_v1。

Nr适配v1实际训练预检在Dataset读取第一批时失败：`_augment_nr3d() takes 1 positional argument but 2 were given`。Python3.7 AST源码替换遗漏两个staticmethod声明，先前类级调用检查没有覆盖实例绑定；preflight/updates.jsonl为空，0优化步、没有fit目录或终点。失败源码、spec和日志均保留，不能把§247的类级检查写成完整Dataset训练调用已经通过。

仅在独立v2训练副本补回两个staticmethod声明。对全部32919条表达同时检查类级与实例调用、增强与view判断相反，全部一致；325条方向词修正的语义不变，0条反向。源码SHA由a425943b8abdb82c0b8aae6cd8c20868b9596ff6045e8c410c472f87640578d9变为4996ac80f2707851520e2c728f8be743df4f9476be4e333df439aa5b72a62167。原作者权重、数据顺序、学习率、5614更新、44909行和终点规则完全不变。

新root /root/autodl-tmp/mcln_eg3dvg_nr3d_adapt_20260920_v2；controller5242于18:26:18启动；spec0761746371fa1fe60ff2e1512cd0c0df5444a25a5cf765f2d74b1187b98b6569；恢复文件目录/root/mcln_eg3dvg_nr3d_adapt_states_20260920_v2。CPU配对分析5243每300秒等待完整fit+终点评估+独立audit，之后严格核对起终点输入身份、点SHA、GT/对象框，分别输出修复/破坏；脚本scripts/analyze_eg3dvg_nr3d_adaptation.py。该分析不增加模型forward、不据结果改推理。

ScanRefer完整作者权重5542/4952（58.2878/52.0825%）和V99保护5572/4797保持独立。Sr3D尚无EG模型结果；三数据集目标未完成。Mask不参与晋级，但作者用于REC平均框的预测Mask路径及原生训练损失保留。


18:30续记：v2预检完成模型正/负表达前向后，在原生losses.py:909读取end_points['super_xyz_list']时KeyError，仍为0优化步，updates.jsonl为空、无fit目录。上游eg.py已在原生前向计算super_xyz_list，却未写入end_points。v3仅在原计算后补一行`end_points['super_xyz_list'] = super_xyz_list`，不修改张量计算、任何loss公式或权重。模型源码SHA bb6ff04fd904f269a01ecb8dce33d00a372f45ca50915ba884f824f352e4407e→b65e77b9e3dd095591e37cbd88f8a6c3179b3499f4ed027569f9a33870438ee1。

当前独立root /root/autodl-tmp/mcln_eg3dvg_nr3d_adapt_20260920_v3，状态目录/root/mcln_eg3dvg_nr3d_adapt_states_20260920_v3；controller5317和终态配对分析等待器5318于18:29:25启动。spec22458ac02fbed7ba7430c87e799e94c7d168eae52ca1c1eb8fefc2ca18b94c2d。v1/v2失败源、日志、空更新记录保留；作者权重、单seed、44909行顺序、5614步、LR、原生损失和终点评估规则不变。验证仍使用完成零更新评估的原源，其不调用训练loss；旧正式结果不因此重算或替换。启动不等于训练通过，随后以完整两步backward/AdamW回执确认。


2026-09-20T18:36:07.906487+08:00实际状态：v3两批16条训练预检于2026-09-20T18:31:02.036710+08:00完成，2次完整原生loss/backward/AdamW、199项冻结参数不变，869项允许训练，峰值分配显存27.313GiB。预检权重已丢弃，fit从作者epoch69重新严格加载。最新固定训练日志64/5614步、512/44909行、197.91秒，loss与梯度有限，controller5317和实际GPU训练进程存活。按已完成训练平均速度加约30分钟终点评估，预计剩余约5.3小时；这是耗时估计，不能作为完成承诺或精度证据。未据loss更改任何设置。完整逐步日志、每512步恢复状态和固定终点保持预定规则；结束后自动完成7899正式评估、CPU复算及起终点配对计数。当前仍无Nr适配终态成绩，Sr没有EG成绩。
