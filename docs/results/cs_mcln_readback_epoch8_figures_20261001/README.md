# R第0—8轮对照曲线

这两张图使用已经保存的完整ScanRefer验证计数，每个点均为9508条表达、原生`last/bbs`输出。E0为共同E71零更新起点；E1—E8为完整训练、评估及保存后的结果。图中没有E9及以后的推测点，也没有将不同轮次的单项最好合并。

| 图 | 数据 | 矢量文件 |
|---|---|---|
| Acc@0.25，第0—8轮 | `epoch0_8_counts.csv`的`hits025 / samples` | `r_epoch8_acc025.svg`、`r_epoch8_acc025.pdf` |
| Acc@0.50，第0—8轮 | `epoch0_8_counts.csv`的`hits050 / samples` | `r_epoch8_acc050.svg`、`r_epoch8_acc050.pdf` |

CS+R、历史CS和历史native均为E71初始化、seed2027、batch12。R同时采用了M1确定性成员聚合修正，因此历史曲线仅供描述性参照，不能据此分离R的独立因果增量。相同数值协议的CS控制尚未训练。这些点是同一单seed轨迹上的相关快照，不是独立重复实验；图中不报告跨seed误差线或显著性。

纵轴经过截断，以显示早期变化；轴标签保留实际百分比。E71水平线来自CSV中的E0计数。图中未绘制V99完整系统，因为它使用不同的输出路径；本图不能用于声称已经继承V99精度。标题及解释置于本文和LaTeX caption中，图内只有坐标轴与图例。

`data_sources.json`记录三份原始CSV的路径和SHA256，绘图脚本读取本目录中的固定计数快照。使用Matplotlib 3.11.2：

```powershell
uv run --offline --no-project --with matplotlib==3.11.2 python gen_acc025.py
uv run --offline --no-project --with matplotlib==3.11.2 python gen_acc050.py
```

SVG与PDF为矢量输出，PNG用于渲染检查；生成脚本和LaTeX引用片段一并保留。它们是同一批真实计数的可视化，不构成新增模型实验或精度结果。

LaTeX示例按双栏文档使用`figure*`和`0.8\textwidth`，保持约5.5英寸的展示宽度。排版时应核对最终字体大小，避免将这份图直接缩为半页宽而使图例和刻度难以阅读。
