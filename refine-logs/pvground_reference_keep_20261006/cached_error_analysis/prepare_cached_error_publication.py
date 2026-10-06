"""Append the new cached-data diagnosis without querying the active training."""
from pathlib import Path

root = Path(__file__).resolve().parent
source = (root / 'publish_fit_authorized.py').read_text(encoding='utf-8')
source = source.replace('Append actual M0 closure and formal ScanRefer reference-keep launch to Doc89.', 'Append a cached selected-Query geometry diagnosis to Doc90; no training query.')
source = source.replace('PUBLISH_FIT_SOURCE_REVIEW', 'PUBLISH_CACHED_ERROR_SOURCE_REVIEW')
source = source.replace("root/'FIT_SOURCE_REVIEW.json'", "root/'CACHED_REFERENCE_ERROR_REVIEW.json'")
source = source.replace("source_review['execution_scope']=='SOURCE_ONLY'", "source_review['execution_scope']=='SOURCE_AND_CACHED_EVIDENCE'")
source = source.replace("assert not (root/'fit_publication.json').exists()", "assert not (root/'cached_error_publication.json').exists()")
source = source.replace("previous_path=root/'preflight_publication.json'", "previous_path=root/'fit_publication.json'")
source = source.replace("previous['section']=='20.376.89'", "previous['section']=='20.376.90'")
source = source.replace('fit_registration', 'cached_error_analysis')
begin = source.index("section=f'''")
end = source.index('client=paramiko.SSHClient()', begin)
source = source[:begin] + '''summary=json.loads((root/'cached_reference_errors/SUMMARY.json').read_bytes())
assert summary['source_model_hits']==[5598,4848]
assert summary['new_model_evaluations']==summary['new_optimizer_steps']==summary['remote_queries']==0
damage=summary['cohorts']['0.25_damaged']
assert damage['rows']==191 and damage['gt_covered95_and_reference_volume_over4']==128 and damage['mask_good50']==52
assert summary['cohorts']['0.25_repaired']['rows']==174
assert summary['cohorts']['0.5_repaired']['rows']==724 and summary['cohorts']['0.5_damaged']['rows']==371
section=f\'\'\'

## 20.376.91 当前Mask参考的Acc@0.25损失：只读缓存范围诊断（{stamp}）

本次仅分析已完成的5598/4848最好模型的9508逐行缓存，没有模型前向、优化器更新、远端训练进度查询或活动代码变更。当前参考保持λ0/1对照仍按上一节运行；首次观察时间仍为{observer['first_observation_cst']}，此前不轮询。用户Scan5620/4764及三个有效模块门槛不变，之后完整模型从作者对应权重分别训练Sr/Nr，单seed2027。

固定该模型已经选中的同一个Query，对比其原生粗框先验与中性融合Mask参考：Acc@0.25修复174、破坏191，净−17；Acc@0.5修复724、破坏371，净+353。由此原粗框先验为5615/4495，参考为5598/4848。这是同一Query的内部几何比较，不能替换历史4511整模型到4848的−18/+337，也不能拼接不同模型成绩。预测评分和所选Query在本次比较中不变。

191条宽松阈值破坏中，128条参考框覆盖至少95%的GT体积，同时参考体积超过GT四倍；63条GT体积覆盖低于95%。这两个组描述实际几何重叠：多数该类损失表现为覆盖目标但范围过大。191条的参考/GT体积比中位数4.8803916，最大单面绝对误差中位数0.503725米；其中52条所选融合Mask自身IoU>0.5，说明即便分割IoU合格，包围范围仍可能过大。所有191条参考有效，没有触发空支撑原框保留规则。

两阈值破坏集合共有118条，另有73条仅宽松破坏和253条仅严格破坏；严格修复724条中655条融合Mask IoU>0.5。上述数值来自同一9508缓存，不是本轮训练成绩。体积、覆盖及Mask重叠不能证明真实物理实例身份，也没有仅凭缓存证明过大范围由哪些远端误分点造成。GT只用于离线描述，不形成推理门控、硬退回原框或候选过滤规则。

这份新证据把下一项边界结构的观察重点放在“已覆盖目标但范围过大”的参考上；仍先等当前参考保持正式对照结束，再决定边界读取改法，不中途叠加采样/注意力/教师/质量排序。不能把128条全部计作可恢复收益。CPU源与独立缓存核验报告随{prefix}发布，原始候选NPZ、权重和其他私有文件不发布。
\'\'\'
new=old+section.encode('utf-8');assert new.startswith(old) and new.count(b'## 20.376.91 ')==1
names=['postrun/analyze_cached_reference_errors.py','cached_reference_errors/SUMMARY.json','cached_reference_errors/EXECUTION.json',
    'CACHED_REFERENCE_ERROR_REVIEW.json','CACHED_REFERENCE_ERROR_REVIEW.md',
    'prepare_cached_error_publication.py','publish_cached_error_authorized.py',
    'PUBLISH_CACHED_ERROR_SOURCE_REVIEW.json','PUBLISH_CACHED_ERROR_SOURCE_REVIEW.md']
payloads={prefix+name:(root/name).read_bytes() for name in names};payloads[prefix+'.gitattributes']=b'** -text\\n'
assert all(not name.endswith(('.pt','.pth','.npz')) and '.aris' not in name for name in payloads)
''' + source[end:]
source = source.replace('Reference-preservation real M0 PASS and same-budget formal fit launched; no new formal metrics.', 'Cached same-Query reference geometry diagnosis; no training or model replay.')
source = source.replace('Record reference-preservation M0 PASS and formal ScanRefer contrast launch', 'Record cached Mask-reference overextension diagnosis')
source = source.replace("section='20.376.90'", "section='20.376.91'")
source = source.replace("execution_scope='REFERENCE_KEEP_M0_PASSED_FORMAL_FIT_LAUNCHED_NOT_COMPLETE'", "execution_scope='CACHED_REFERENCE_DIAGNOSIS_NO_NEW_MODEL_RESULT'")
source = source.replace("(root/'fit_publication.json').write_text", "(root/'cached_error_publication.json').write_text")
source = source.replace("latest_publication=str(root/'fit_publication.json')", "latest_publication=str(root/'cached_error_publication.json')")
source = source.replace('Doc90 actualM0bothPASS and formal controller', 'Doc91 cached reference geometry diagnosis; original formal controller')
source = source.replace("+' launched, MAIN'", "+' unchanged, MAIN'")
compile(source, 'publish_cached_error_authorized.py', 'exec')
(root / 'publish_cached_error_authorized.py').write_text(source, encoding='utf-8')
print('CACHED_ERROR_DOC91_PREPARED_NOT_EXECUTED')
