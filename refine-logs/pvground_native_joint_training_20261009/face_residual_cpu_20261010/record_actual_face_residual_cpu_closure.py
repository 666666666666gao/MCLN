"""Close the actual bounded face-module CPU review; no GPU admission."""
import datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
f = root/'face_residual_preparation_20261010'
audit_path = f/'actual_CPU_review/EXPERIMENT_AUDIT.json'
seal_path = f/'actual_CPU_review/OUTPUT_SHA256.json'
audit = json.loads(audit_path.read_bytes())
seal = json.loads(seal_path.read_bytes())
assert audit['verdict'] in ('PASS','WARN') and audit['blocking_issue_count'] == 0
assert audit['bounded_check_verdict'] == 'PASS'
for name,digest in seal['files'].items():
    assert hashlib.sha256((seal_path.parent/name).read_bytes()).hexdigest() == digest
result = json.loads((f/'cpu_execution_attempt2/CPU_MODULE_RESULT.json').read_bytes())
receipt = json.loads((f/'cpu_execution_attempt2/CPU_EXECUTION.json').read_bytes())
assert receipt['exit_code'] == 0 and result['status'] == 'ACTUAL_FACE_RESIDUAL_SYNTHETIC_CPU_MODULE_PASS'
assert result['zero_output_boxes_bitwise_equal_loaded_prior_both_modes'] is True
assert result['CUDA_initialized'] is False and result['GPU_training_admission'] is False
assert result['formal_accuracy'] is None and result['new_weight_files'] == 0
record = dict(status='ACTUAL_FACE_RESIDUAL_CPU_MODULE_EVIDENCE_REVIEW_CLOSED',
    time_cst=datetime.datetime.now().astimezone().isoformat(),audit_verdict=audit['verdict'],
    bounded_check_verdict='PASS',blocking_issue_count=0,
    audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
    seal_sha256=hashlib.sha256(seal_path.read_bytes()).hexdigest(),
    actual_CPU_outcome_review_pending=False,GPU_or_control_training_admission=False,
    reviewer_identity='UNATTESTED',review_independence='same-family',acceptance_status='provisional',
    supported_scope='Synthetic Torch module construction, loaded-prior zero-output equivalence, new-head gradient fixture, supplied-residual decoding, module-only in-memory state roundtrip',
    full_PV_factory_verified=False,native_criterion_verified=False,real_loader_verified=False,
    full_optimizer_recovery_verified=False,normal_precision_result=None,
    first_normal_observation_cst='2026-10-10T08:19:04.678479+08:00',
    active_normal_training_source_changed=False,current_normal_training_queries=0,
    new_weight_files=0,full_goal_complete=False)
target = root/'ACTUAL_FACE_RESIDUAL_CPU_AUDIT_CLOSURE.json'
assert not target.exists()
target.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
note = f'''## 20.376.134 面残差候选的实际CPU模块检查闭环

沿已保存的同轴双面耦合分析，既有面残差候选完成了一次限定的真实Torch CPU模块检查；本节没有查询当前正常训练，不改当前网络、训练配置或C-off优先的控制顺序。

实际child执行时间为{receipt['started_cst']}至{receipt['finished_cst']}，exit0、Torch{result['torch_version']}、CUDA未初始化。共同先验29793参数，新面头23425参数，合计53218参数/20模块状态，其中新增6状态。加载受保护的d06/f989模块后，两种新增证据可见模式在零输出时均与原轴先验的中心/尺寸通过torch.equal检验相同；隐藏模式的新增源token及占比列实际为零。独立内部0.5门控的合成夹具进行了两次内存AdamW更新，第一步输出梯度非零而内部编码器梯度为零，输出更新后内部编码器梯度非零。供应面残差夹具验证了真实Torch解码中的面交叉排序；模块级内存保存/重载恢复了refine_one输出。未写入权重文件。相等检查使用torch.equal，未额外比较张量内存字节。

范围必须保留：上述输入是合成工程夹具；供应残差不是网络学出的预测，内部0.5门控梯度夹具不证明已加载先验在其实际饱和门控上的梯度行为。没有1301完整PV工厂、native criterion、真实loader、完整包装器forward恢复、完整优化器恢复、GPU或REC新成绩。它只排除了候选实现的这些工程疑问，不能算有效创新或精度提升；是否进入后续训练仍依据当前主线与直接控制结果决定。

第一次SSH在主机密钥校验阶段失败（exit255、stdout0）；原脚本、R1及错误证据保留。恢复此前已实际成功的C:/正斜杠SSH路径后，一次静态读取确认专用CPU目录不存在，再经fresh R2源码复核执行唯一attempt2。StrictHostKeyChecking始终保留；没有盲目重启当前训练。确切OpenSSH内部解析根因尚未证明，私有SSH stderr不发布。

实际结果复核为{audit['verdict']}，限定检查PASS、0阻断；Astra/max为请求路由，实际身份UNATTESTED，same-family/provisional，不冒充跨模型独立接受。总目标仍为同一完整ScanRefer模型Acc@0.25>59.5%、Acc@0.5>51%、三项有直接证据的有效机制，之后同一最终结构分别训练Nr3D/Sr3D。当前正常训练原定08:19首次读取不变。
'''
(root/'HANDOFF_FACE_RESIDUAL_CPU_20261010.md').write_text(note,encoding='utf-8')
print(json.dumps(record))
