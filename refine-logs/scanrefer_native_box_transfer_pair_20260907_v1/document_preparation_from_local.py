import datetime,hashlib,json,os
from pathlib import Path
import paramiko

repo=Path('C:/Users/gb/.codex_mcln_g0_20260905')
master=repo/'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop=Path('C:/Users/gb/Desktop/document')/master.name
runtime='/home/gb/new butd/butd_detr-main/MCLN-main/'
archive=repo/'refine-logs/scanrefer_native_box_transfer_pair_20260907_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=master.read_bytes();assert hashlib.sha256(old).hexdigest()=='d55ad70b13f64bd13693abc1ce6f50c38cc1920e712001544a7f0b16220a68a1'
assert desktop.read_bytes()==old
cleanup=json.loads((repo/'refine-logs/weight_cleanup_20260907_v2/receipt.json').read_bytes())
mesh=repo/'refine-logs/scanrefer_mesh_teacher_transfer_20260907_v1'
probe=repo/'refine-logs/scanrefer_native_box_transfer_probe_20260907_v1'
p=json.loads((probe/'receipt.json').read_bytes());assert p['status']=='pass'
assert (probe/'controller.exit').read_text().strip()=='0'
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M CST')
addition='''

### 20.119 用户授权的失败终点清理：保护权重逐份复核（2026-09-07 12:31 CST）

冻结读出两臂已自然结束，终态6887行、独立审计及未晋级决定均保留。核对实际进程和依赖后，仅删除本轮两份失败终点native_only/frozen_gt，各618602215 bytes，分别对应§20.116已登记SHA。释放allocated1237213184 bytes（约1.152GiB），剩余9332264960 bytes（约8.691GiB）。E71、Parent、Geometry、V99、Nr平均E57和Nr续训E57六份保护SHA在删除前后逐项一致。日志、行级结果、manifest和审计均未删除；Sr历史权重缺失不由本次清理造成。本轮清理receipt保留在`refine-logs/weight_cleanup_20260907_v2/receipt.json`。

### 20.120 正确mesh上的教师几何转移可行性复核（2026-09-07 12:41 CST）

发现旧512条teacher-transfer审计继承mixed DATA_ROOT。本次固定原512条fit身份，只纠正到mesh目录，并核对1201份训练superpoint、614份源码和四份保护artifact。无优化、无新权重、无正式评估。为检查对应关系，保留原生256框、Top16×7几何集合、最终部署教师选择和当次GT root Hungarian索引。教师forward没有GT目标；GT仅用于训练可行性诊断。

| 同一512条fit表达 | REC@0.25/@0.50 hits |
|---|---:|
| 原生Default | 491/461 |
| 正确mesh V99教师 | 497/481 |
| 原生GT Hungarian root | 511/499 |
| 原生Full256 oracle | 511/501 |
| 教师变体所属Query的原框 | 495/457 |
| 与教师框几何最接近的原生Query框 | 497/477 |

独立NumPy重算所有保存框及选择有效性通过，512个原生/匹配记录相对旧审计在1e-6容差内一致，但149行教师Query或variant选择改变。正确mesh教师相对旧mixed教师494/471的差异是数据变化，不是网络提升。教师相对原生修复/破坏为7/1和23/3；仍低于GT Hungarian的511/499，不能替代GT。342行教师IoU>0.25且高于当前Hungarian root框，正IoU差均值0.137298。425个教师答案来自非原框变体，363行几何最近Query与变体来源编号不同；不应按历史Query编号永久绑定监督。

这些仅是主干已见fit场景上的教师目标可行性，不是学生精度。原始rows27.38MB保留在远端和本地证据目录，Git记录路径与SHA，避免把大候选缓存加入源码仓库。
'''
addition+='\nmesh receipt SHA `'+sha(mesh/'receipt.json')+'`；rows SHA `'+sha(mesh/'rows.json')+'`；独立核验 SHA `'+sha(mesh/'independent_check.json')+'`。\n'
addition+='\n### 20.121 原生框转移实际梯度检查通过；固定配对准备完成（'+now+'）\n\n'
addition+='16条预先固定fit输入、两批12/4；14条有正教师收益。只允许最后center_residual_head和size_pred_head共16参数张量、335814参数更新。原生GT梯度范数0.680516/0.812899，辅助梯度7.484442/5.418529，均有限；两批点积为正，但不据16例推断总体梯度一致。gt_only与gt_teacher_box各在末批做两次一次性更新，16项参数改变，其余完整模型及教师/旧读出不变；last_sem_cls_scores、last_proj_queries、proj_tokens、last_pred_masks、sp_last_pred_masks逐值不变。没有保存probe权重，没有正式成绩。probe完成时间12:48:28，GPU峰值4317.93MiB。\n\n'
addition+='下一项是几何能力衔接实验，不是新MoE或完整DETRDistill复现：两臂均原生GT训练，候选臂增加冻结V99实际最终框的辅助回归；当次学生GT Hungarian绑定root，仅当教师IoU>0.25且优于当前匹配框才有正权重。辅助项以正IoU差加权，沿用5×(中心L1+0.2×尺寸L1)+GIoU，权重1；不复制教师分数、不固定Query编号。完整自然语言消歧仍待后续验证，本检查不替代三数据集目标。\n\n'
addition+='固定设置：正确mesh、相同E71、model.eval、lr1e-6、AdamW wd0.0005、clip0.1、B12、每臂一遍29778行即2482更新；两臂共用相同批次，6887模块留出仅做前后完整评估，主干曾见这些场景。原生REC与完整V99 REC分别记录；固定候选gt_teacher_box须在系统REC双阈值相对起点及gt_only都不退化，才做固定正式Scan验证，Scan通过现行底线后尽快转Nr/Sr。不等待59/51，不恢复Nr/Sr Mask门槛，不扫LR/权重/epoch。\n\n'
addition+='为节约磁盘，只保存两份16参数终点及optimizer，不重复保存冻结E71；训练脚本在终态前从磁盘终点+保护E71重建完整state并逐值核对，再执行终态forward。CPU独立审计接续检查2482更新、样本互斥、教师GT支持权重、16份optimizer状态、逐行指标及晋级决定。两份终点预计合计约8MB，实际以保存大小为准。原环境5项loss测试通过、全部新Python在3.7语法检查通过；当前训练仅staged，尚未启动，不能将准备状态称为新训练结果。\n\n'
addition+='probe receipt SHA `'+sha(probe/'receipt.json')+'`；training manifest SHA `'+sha(archive/'input_manifest.json')+'`。当前架构仍是原E71及已有任务头；推理可单独读取更新后的原生框，但尚未证明能达到V99保护线，不能宣称已去除后处理且性能保持。完整计划见`docs/SCANREFER_NATIVE_BOX_TRANSFER_PLAN_2026-09-07.md`。\n'
new=old+addition.encode()
tracker=repo/'refine-logs/EXPERIMENT_TRACKER.md'
lines=tracker.read_text(encoding='utf-8').splitlines()
lines[2]='Updated: '+now+'. Sections20.119-121: failed frozen-readout endpoints safely removed;correct-mesh512 teacher audit and16-row native-box probe PASS;fixed GT-only/teacher-box pair staged,not launched;no new formal.'
lines[6:6]=['| Native teacher-box transfer | Actual16fit probe PASS;14eligible;only16box tensors/335814params update | 5CPUtests PASS;fixed2482/arm pair staged;GT-only versus GT+teacher geometry;no new formal or Sr/Nr training |',
'| Correct-mesh teacher transfer | Same512fit rows;native491/461,teacher497/481;342GT-supported better teacher boxes | Input-version correction,not model gain;rawcandidate rows retained off Git with SHA;GT primary |']
tracker_raw=('\n'.join(lines)+'\n').encode()
plan=repo/'docs/SCANREFER_NATIVE_BOX_TRANSFER_PLAN_2026-09-07.md'
plan_raw=plan.read_bytes()+('''

## 2026-09-07 固定配对设置

实际16行probe已通过（14行有效教师监督），5项原环境CPU测试通过。下一项固定候选gt_teacher_box、控制gt_only；正确mesh，E71预训练，原生GT监督不变，仅最后中心/尺寸头16项参数更新。两臂model.eval，AdamW lr1e-6、wd0.0005、clip0.1，teacher loss权重1，B12，一遍29778fit行=2482更新。6887模块holdout按相同输入做完整前后评估，不能当作新场景正式集。

单独报告原生与完整V99。候选系统REC两阈值均不低于零更新及gt_only才进入固定正式Scan验证；原生结果未达到V99前不声称去后处理。正式Scan达到现行REC/Mask底线即接Nr/Sr的REC训练。失败则封存该配置，不扫权重、LR或epoch。

输出为两份16参数+优化器小终点，绑定E71 SHA；终态前实际重建完整模型并逐值核对。独立CPU审计自动接续。记录全部fit身份、点SHA、教师框和root，核验GT支持权重仅进入训练目标；不把GT筛选用于推理。当前是staged，启动证据另记。
''').encode()
raw_record={'path_remote':'/root/autodl-tmp/mcln_scanrefer_mesh_teacher_transfer_20260907_v1/rows.json','path_local':str(mesh/'rows.json'),
 'bytes':(mesh/'rows.json').stat().st_size,'sha256':sha(mesh/'rows.json'),'included_in_git':False,'contents':'512 fit rows, native256 boxes and16x7 variants; train diagnostic only'}
(mesh/'raw_rows_location.json').write_bytes((json.dumps(raw_record,indent=2)+'\n').encode())
attrs=repo/'.gitattributes'
with attrs.open('a',encoding='utf-8',newline='\n') as stream:
 for name in ['weight_cleanup_20260907_v2','scanrefer_mesh_teacher_transfer_20260907_v1','scanrefer_native_box_transfer_probe_20260907_v1','scanrefer_native_box_transfer_pair_20260907_v1']:
  stream.write('refine-logs/'+name+'/** -text\n')
with (repo/'.gitignore').open('a',encoding='utf-8',newline='\n') as stream:
 stream.write('\n/refine-logs/scanrefer_mesh_teacher_transfer_20260907_v1/rows.json\n')
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
s=c.open_sftp()
with s.open(runtime+'docs/'+master.name,'rb') as f:
 f.prefetch(file_size=len(old));assert f.read()==old
for rel,raw in [('docs/'+master.name,new),('refine-logs/EXPERIMENT_TRACKER.md',tracker_raw),('docs/'+plan.name,plan_raw)]:
 with s.open(runtime+rel,'wb') as f:f.set_pipelined(True);f.write(raw)
 with s.open(runtime+rel,'rb') as f:f.prefetch(file_size=len(raw));assert f.read()==raw
master.write_bytes(new);desktop.write_bytes(new);tracker.write_bytes(tracker_raw);plan.write_bytes(plan_raw)
for name in ['independent_check.json','raw_rows_location.json']:
 s.put(str(mesh/name),'/root/autodl-tmp/mcln_scanrefer_mesh_teacher_transfer_20260907_v1/'+name)
s.put(str(repo/'scripts/check_mesh_teacher_transfer_rows.py'),'/root/autodl-tmp/mcln_scanrefer_mesh_teacher_transfer_20260907_v1/check_mesh_teacher_transfer_rows.py')
s.close();c.close()
proof={'time_cst':now,'section':'20.121','three_master_copies_equal':True,'master_sha256':hashlib.sha256(new).hexdigest(),'master_bytes':len(new),'training_started':False,'goal_complete':False}
(archive/'handoff_sync.json').write_bytes((json.dumps(proof,indent=2)+'\n').encode())
(archive/'document_preparation_from_local.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(proof),flush=True)
