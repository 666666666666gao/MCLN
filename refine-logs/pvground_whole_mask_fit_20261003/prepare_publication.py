"""Adapt the completed engineering publisher for the next actual reviewed launch."""
import ast
from pathlib import Path

root = Path(__file__).parent
source = root.parent/'pvground_whole_mask_integration_20261003/publish_preflight_completed.py'
text = source.read_text().replace('\r\n', '\n')

def replace(before, after):
    global text
    assert text.count(before) == 1, before
    text = text.replace(before, after)

replace('"""Publish actual engineering completion, with no formal-run launch."""',
        '"""Publish only the actual reviewed formal source-control launch."""')
replace("assert not (local / 'completed_preflight_publication.json').exists()", "assert not (local / 'launch_publication.json').exists()")
start = text.index('previous = json.loads(')
end = text.index('repos = ', start)
text = text[:start] + (
    "previous = json.loads((local.parent / 'pvground_whole_mask_integration_20261003/completed_preflight_publication.json').read_bytes())\n"
    "launch = json.loads((local/'launch.json').read_bytes())\n"
    "review = json.loads((local/'EXPERIMENT_CODE_REVIEW.json').read_bytes())\n"
    "assert launch['formal_training_started'] and not launch['formal_result_available']\n"
    "assert review['verdict'] in ('PASS','WARN') and not review['blocking_findings']\n"
) + text[end:]
replace("b'## 20.376.36 '", "b'## 20.376.37 '")
start = text.index("table = ['|")
end = text.index('new = old + addition.encode', start)
text = text[:start] + '''addition = ('\\n\\n## 20.376.37 完整Mask范围正式来源对照启动（' + now + '）\\n\\n'
    '两组实际PV工程预检完成以后，正式runner复用成功工厂及先前已实际执行的原生评估／fit循环，'
    '经新鲜native Codex源码审查后于'+launch['time_cst']+'实际串行启动。运行目录'+launch['root']+'，'
    'controller进程'+launch['process'].split()[0]+'；状态为已启动、尚无完整正式新精度，不将工程PASS冒充方法增益。\\n\\n'
    '比较local_range／whole_range；两组重新加载同一保护原G5615／4495和官方PV父权重，fresh AdamW，'
    '不继承预检更新或失败P2/P3终点。相同400614参数头、1302输入、局部原生融合Mask支撑和6维粗框条件，'
    '仅109完整范围摘要置零或读取实际预测值。全部256候选、原生全场景／文本／Mask及唯一bbs输出保留；'
    '无P2、教师、质量头、候选对比扩展／归一化、双源排名和数据集推理门控。\\n\\n'
    '每组29778 fit条各一次／3723更新，seed2027、batch8、核心及backboneLR1e-5、WD5e-4、clip0.1。'
    '各记录6887初始／终点模块留出和9508正式开发验证；6887被作者预训练见过，9508长期用于开发，'
    '不据此宣称独立未使用测试或零样本跨域泛化。最终框在原生Mask后精修，实际criterion／evaluator读取同一框。'
    '尺寸沿用加性修正与既有评估clamp，不声明自动正尺寸。\\n\\n'
    '审查为新鲜上下文的native gpt-6-astra/max请求、same-family／provisional，实际backend SKU未独立证明。'
    '结论'+review['verdict']+'、blocking='+str(len(review['blocking_findings']))+'；是SOURCE_ONLY审查，'
    '无新增GPU或精度证据。已有真实两步预检的十份模型依赖原样复用，没有重复预检来制造通过记录。'
    '本地Py3.7语法和真实历史6887初始行的比较器检查通过，不等于正式全网络已逐位配对。\\n\\n'
    'local_range先完成train／formal，whole_range核对其实际初始输入和fit顺序。跨进程连续输出／选择／阈值差异如实保存，'
    '只断言同次前向零输出和实际输入身份，不虚构跨进程逐位一致。正式恢复额外核对当前范围模型元数据。\\n\\n'
    '保存控制器已落实用户持续授权：每组完成9508原生验证、记录SHA、所选框/root的CPU阈值重算及终点SHA之后，'
    '及时删除不优于当前bbs Acc@0.50最佳的本轮终点，相等保留已有最佳；如后来更好，也只在后来核验后替换本轮较差最佳。'
    '仅删除本轮两个目录中的确切terminal.pth；原G／必要父权重／V99与日志、源码、逐行结果不删除，'
    '不再留失败本地权重归档。活动latest原子替换、最终rename为terminal，只有一个活动恢复点。'
    '异常核验停止并保留待核验文件，不执行未核验清理；CPU核验不等于独立重放全部模型／优化器状态。\\n\\n'
    '启动实际空闲数据盘'+str(launch['resources']['free_bytes'])+'字节，检查所需原子保存余量'
    +str(launch['resources']['required_bytes'])+'字节，GPU空闲，warm环境复用未重建。'
    '两组暂估6～8小时，不是新吞吐实测；唯一只读观察器首查40分钟，随后按已完成更新／完整eval估计结束前2～3分钟查看，'
    '未完成再240秒查，不启动第二份训练或观察器。\\n\\n'
    '主目标仍须比较本轮来源控制和原G4495严格命中；只优于续训退化不晋级。'
    '至少保持原G5615／4495，再争取同一bbs检查点5615／4754；Acc@0.25、Mask、bbf、修复／破坏同步报告，'
    '不拼不同模式／轮次最好列。当前无新正式结果、无方法晋级，Nr3D／Sr3D尚未训练，总目标未完成。\\n')
''' + text[end:]
start = text.index("prefix = 'refine-logs/")
end = text.index('client = paramiko.SSHClient()', start)
text = text[:start] + (
    "prefix = 'refine-logs/pvground_whole_mask_fit_20261003/'\n"
    "names = [path.name for path in local.iterdir() if path.is_file() and path.suffix in ('.py','.json','.md') and path.name != 'wait.json'] + ['.gitattributes']\n"
    "payloads = {prefix+name: local/name for name in names}\n"
    "replaced = set()\n"
) + text[end:]
replace("real native factory preflights complete; four engineering updates, no formal result or disk weight.",
        "reviewed fixed-budget whole-Mask range source control launched; no formal result; automatic verified nonbest retention.")
replace('Record real whole-Mask range preflights and formal source-control plan',
        'Launch reviewed whole-Mask range source control with best-only weight retention')
start = text.index('record = dict(time_cst=')
end = text.index('print(json.dumps(record)', start)
text = text[:start] + (
    "record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), section='20.376.37',\n"
    "    heads=heads, github_main=heads[0], handoff_sha256=digest, handoff_bytes=len(new),\n"
    "    four_local_and_remote_equal=True, exact_committed_payloads=True, payload_count=len(payloads),\n"
    "    actual_launch_cst=launch['time_cst'], formal_training_started=True, formal_result_available=False,\n"
    "    review_verdict=review['verdict'], best_retained='original_g_5615_4495',\n"
    "    controller_process=launch['process'], method_promoted=False, goal_achieved=False)\n"
    "(local/'launch_publication.json').write_text(json.dumps(record,indent=2)+'\\n',encoding='utf-8')\n"
) + text[end:]
ast.parse(text, feature_version=(3, 7))
(root/'publish_launch.py').write_text(text, encoding='utf-8', newline='\n')
(root/'.gitattributes').write_bytes(b'* -text\n')
print('FORMAL_LAUNCH_PUBLISHER_PREPARED_NOT_EXECUTED')
