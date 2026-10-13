"""Record the bound source branch; no runtime/model or experiment change."""
import datetime
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
stage=json.loads((root/'SOURCE_STAGE_R4_RECEIPT.json').read_bytes())
sources={
    'models/pv_ground.py':root/'source_overlay_v2/models/pv_ground.py',
    'models/losses.py':root/'source_overlay_v2/models/losses.py',
    'main_utils.py':root/'source_overlay_v3/main_utils.py',
    'models/modules.py':Path('C:/Users/gb/.codex/tmp/pvground_mask_support_correction_20261008_v2/complete_fit/PV-Ground/models/modules.py')}
identities={}
for name,path in sources.items():
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest==stage['source_files'][name]['sha256']
    identities[name]=dict(local_path=str(path),sha256=digest,remote_path=stage['source_files'][name]['path'])
model=sources['models/pv_ground.py'].read_text(encoding='utf-8')
loss=sources['models/losses.py'].read_text(encoding='utf-8')
assert 'prediction_masks.append(pred_masks.expand(1,256,pred_masks.shape[-1]))' in model
assert 'proposal_pred_masks' not in model
assert 'pred_masks' not in sources['models/modules.py'].read_text(encoding='utf-8')
assert 'output["pred_masks"] = end_points["last_pred_masks"]' in loss
assert 'output["sp_pred_masks"]=end_points["sp_last_pred_masks"]' in loss
assert 'self.cost_masks = 0.0002' in loss
record=dict(scope='CURRENT_R4_PV_LAST_LAYER_MATCHER_MASK_BRANCH_SOURCE_ONLY',
    recorded_cst=datetime.datetime.now().astimezone().isoformat(),sources=identities,
    text_mask_is_expanded_across256_queries=True,own_query_mask_passed_as_separate_sp_pred_masks=True,
    matcher_reads_pred_masks_not_sp_pred_masks=True,
    matching_mask_cost='0.0002 * binary foreground point Hamming/L1 sum',
    matching_constructor=dict(cost_class=1,cost_bbox=0,cost_giou=2),
    algebraic_scope='For a fixed scene/GT column, repeated Text masks produce identical mask costs across candidate rows; this term cannot distinguish candidate Queries.',
    runtime_cost_matrices_measured=False,performance_effect_of_own_mask_matching_unknown=True,
    inference_inputs_or_scores_changed=False,active_training_source_changed=False,
    new_model_forwards=0,remote_queries=0,new_experiment_launched=False,
    not_a_claim_baseline_has_no_mask_loss_or_zero_query_gradients=True,
    not_a_unique_failure_cause_or_novelty_claim=True)
out=root/'MATCHER_MASK_BRANCH_SOURCE_NOTE_20261013.json';assert not out.exists()
out.write_text(json.dumps(record,indent=2)+'\n')
text='''# 当前PV末层匹配的Mask来源：源码范围澄清

本记录只读取与实际R4清单SHA一致的源码，未读远端进度、运行新网络、改活动实验或统计真实匹配成本矩阵。

`models/pv_ground.py:549`从选定文本token生成一个Text Mask，550将其expand到256个候选；560/561分别保存自身Query Mask与广播Text Mask。570–574的A更正写回自身Query Mask。

`models/losses.py:854`区分多Mask路径；在当前PV末层路径，911把Text结果传为pred_masks，912把自身Query结果传为sp_pred_masks。`HungarianMatcher.forward`的328–342只读前者，按logit>0变成点级前景，再用L1/Hamming和GT比较，系数294为0.0002。自身Query Mask仍接受分割损失，不能写成它完全没有训练。

对某场景的一个GT列，广播Text Mask在256个Query行相同，所以该项成本是列常数，不能区分这些候选。当前constructor在`main_utils.py:291`为class1、bbox0、GIoU2；原生匹配并非整体常量，其他成本仍区分候选。

实际PV的proposal/decoder预测头源码也已绑定同一R4 SHA；该头不产生proposal_pred_masks。本分析不推广到其他MCLN多Mask分支或所有原作者模型版本。

这指出的是自身支撑与当前匹配接口的具体差异，不证明更换来源一定涨点、不是所有退化的唯一原因，也不能将普通Mask匹配声称为首次提出。当前selected_geometry训练保持原规则；若后续需要调整责任分配，应在真实同起点控制中检验具体来源与原生损失尺度，而不是同时改推理排名、匹配和多个监督项。
'''
(root/'MATCHER_MASK_BRANCH_SOURCE_NOTE_20261013.md').write_text(text,encoding='utf-8')
print(json.dumps(dict(recorded=True,scope=record['scope'],source_files=len(sources),active_training_changed=False)))
