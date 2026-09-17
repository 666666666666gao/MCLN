# PV-Ground ScanRefer固定终点的CPU复核接续

本接续读取已启动的`mcln_pvground_scanrefer_finetune_20260908_v1`产物，不修改其训练源码、Loss、预算、随机种子或运行进程。复核在bdetr的CPU NumPy环境执行，不重新运行模型，不占用GPU。

## 实际复核范围

先等待完整`initial/receipt.json`，校验6887个固定留出行及boxes/scores/rows文件SHA。由全部原框沿作者尺寸截断规则重算轴对齐框IoU；核对两模式的选中Query确为最高分，其框与原始候选对应一致。允许float32 GPU与float64 CPU计算产生小于1e-5的IoU数值误差，实际误差记录到结果；两阈值计数始终按原Evaluator已经确认的逐行IoU重计，不通过容差改变命中。

Mask部分重计已导出的逐行Mask IoU和阈值；未保存二值Mask，因此此CPU复核不宣称重新验证每个原始点的分割预测。完整作者Evaluator的Box计数和Mask IoU总和已由运行脚本同步核对。

原训练controller退出后，只有退出0及完整终态receipt成立才执行全审计。独立核对3723步、29778个fit行恰好一次、末批2、无holdout行进入更新、全部记录损失有限、终点权重及不可变spec/source/plan SHA；再重计initial/terminal两模式完整指标、REC修复/破坏和三个IoU区间转移。

复核通过表示记录及计数一致，不表示方法通过。主bbs REC两个阈值均不低于同次原生起点，才标记可进入固定9508正式评估；不能依据bbf、Mask或中间权重替换主模式。6887个训练场景留出仍不属于正式新场景结果。

## 运行与输出

新CPU队列位于`/root/autodl-tmp/mcln_pvground_scanrefer_endpoint_audit_20260908_v1`，绑定训练spec和审计源码SHA。首次检查接近起点评估预计完成时间，之后每300秒查看依赖状态。原训练出错时保存事实并退出，不自动重启或重新选择终点。

预期输出为独立`initial_audit.json`、固定终点`audit.json`和queue退出记录。只复制小型代码、日志和回执到仓库；原始boxes/scores留在远端实验目录。该队列不启动正式评估或Nr/Sr训练，不产生新的精度主张。

正式评估后续沿用正确mesh目录中312份验证superpoint的既定SHA清单、全部9508条ScanRefer验证表达、BUTD预测检测框、root GT及原生bbs完整Box/Mask输出。最终比较同一V99的5572/4797和Scan Mask58.70/50.70/44.72底线。预训练原生对照与V99完整系统分列，Nr/Sr REC仍需Scan正式过线后接续。
