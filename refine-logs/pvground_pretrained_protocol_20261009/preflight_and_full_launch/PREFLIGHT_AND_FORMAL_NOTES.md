### 20.376.109 官方父权重batch24实际预检与完整协议对照启动（2026-10-09T01:27:57.579396+08:00）

上一节之后实际执行两次失败预检，均保留源码、日志、退出码和原始回执。首轮在导入时缺少原入口的pvground_observation_query路径，尚无PV前向；第二轮在VSA中缺少QueryAndGroup.observation，尚未完成前向。最小修正只补原正式入口三份辅助模块路径和十个原有无参数ObservedQueryAndGroup包装；不安装G读取器，不新增参数、不改环境。第三轮原进程完成真实24条batch24前向、原生两路评估计数及保存状态不变检查：1234个官方状态、10个无参数包装、全部256候选，allocator分配峰值8561415680B、保留峰值15271460864B。该24条是工程预检，不是正式精度结果。

当前已实际启动同一官方epoch81权重的两个完整9508条只读遍历，batch8与作者测试batch24；原进程958304，screen pvg_official_parent_batch_protocol_20261009，启动2026-10-09T01:25:59.886451+08:00。无优化器更新、无新权重、无G读取器；原生bbs/bbf单独记录用于基线复现，不切换主方法评分、不拼接不同分支两列。依据历史官方父权重batch8的完整耗时893.5654秒估计本轮两遍约1967.1秒，首次结果观察2026-10-09T01:55:47.017305+08:00、估计结束2026-10-09T01:58:47.017305+08:00；实际未结束则每240秒检查。未到时间不读NN进度。

新核对到两个真实协议事实：官方与当前GumbelSampling.forward、PVGround._generate_queries的AST完全一致，推理仍无条件调用F.gumbel_softmax，model.eval不会关闭这种抽样；tokenizer按batch最长文本padding，原bbf对token相似度softmax时未排除pad列。作者公开入口seed_all调用为注释，rng_seed参数存在不等于已执行种子设置。本轮固定seed2027，不做多seed或挑seed。batch对照包含随机抽样形状和padding变化，不能单独将差距归给其中任何一项；作者发布日志的完整RNG状态未取得。

主线最佳仍5599/4859（58.8873%/51.1043%），同一完整模型目标严格>59.5%/>51.0%，至少5658/4850，宽松尚差59个净命中。三项有效贡献及固定完整模型的Nr/Sr独立训练仍未完成。已完成的压缩支持训练、审查、权重归档和自动清理不重复；本项不产生新权重。

核查来源：
- [作者batch24测试脚本](https://github.com/AaNnWwTt/PV-Ground/blob/262e2592589baec7bb83a0d46aae6542d4ccedfb/scripts/test_scanrefer.sh)
- [作者Gumbel采样代码](https://github.com/AaNnWwTt/PV-Ground/blob/262e2592589baec7bb83a0d46aae6542d4ccedfb/models/pv_ground.py)
- [作者评估入口](https://github.com/AaNnWwTt/PV-Ground/blob/262e2592589baec7bb83a0d46aae6542d4ccedfb/train_dist_mod.py)

新增源码、原失败证据、成功预检及正式启动回执位于refine-logs/pvground_pretrained_protocol_20261009/preflight_and_full_launch/。完整batch结果仍未观察，不提前宣称复现59.87或主方法新增能力。
