"""Publish completed residual-arm receipts while the distribution arm remains in progress."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

import paramiko

local = Path(__file__).resolve().parent
assert not (local/'residual_phase_publication.json').exists()
previous = json.loads((local/'volume_analysis_publication.json').read_bytes())
intake = json.loads((local/'residual_phase_complete/INTAKE.json').read_bytes())
fit = json.loads((local/'residual_phase_complete/receipt.json').read_bytes())
formal = json.loads((local/'residual_phase_complete/formal/receipt.json').read_bytes())
retention = json.loads((local/'residual_phase_complete/weight_retention.json').read_bytes())
assert fit['training_steps'] == 3723 and fit['fit_rows'] == 29778
assert fit['head_only'] and fit['original_g_state_unchanged'] and fit['upstream_running_state_eval']
assert formal['status'] == 'pass' and formal['rows'] == 9508
assert not intake['fresh_independent_row_recount']
assert intake['stage'] == 'distribution/train' and len(intake['completed']) == 2
metrics = formal['metrics']; bbs = metrics['bbs']; bbf = metrics['bbf']
deleted_bytes = sum(item['bytes'] for item in retention['deleted'])
workspace = Path('C:/Users/gb')
repos = [workspace/'.codex_mcln_g0_20260905', workspace/'.codex_pvground_cs_20261002',
    workspace/'.codex_mcln_v99_internal_20260928']
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies = [repo/doc for repo in repos] + [workspace/'Desktop/document'/Path(doc).name]
old = copies[0].read_bytes()
assert hashlib.sha256(old).hexdigest() == previous['handoff_sha256']
assert all(path.read_bytes() == old for path in copies)
for repo, head in zip(repos, previous['heads']):
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip() == head
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
guard = workspace/'.codex/tmp/sync_cs_handoff_remote_20260923.py'
guard_raw = guard.read_bytes()
assert guard_raw.count(previous['handoff_sha256'].encode()) == 1
stamp = datetime.datetime.now().astimezone().isoformat()
section = f'''

## 20.376.44 固定原G的边界对照：普通残差完成，六面分布仍活动（{stamp}）

这是本轮第一组完成后的及时记录，不是普通残差／六面分布配对终态。residual训练含初始、终态6887留出评估于{intake['residual_train_finished_cst']}结束；9508条正式开发验证于{intake['residual_formal_finished_cst']}结束。实际恢复10项head delta及原G保护父模型后，原生last/bbs命中{bbs['rec_hits25']}／{bbs['rec_hits50']}，即{100*bbs['rec_hits25']/9508:.4f}%／{100*bbs['rec_hits50']/9508:.4f}%；独立bbf模式为{bbf['rec_hits25']}／{bbf['rec_hits50']}，不拼接两模式单项最好值。

相对原G5615／4495，bbs变化{bbs['rec_hits25']-5615:+d}／{bbs['rec_hits50']-4495:+d}，没有超过原G。bbs Mask命中{bbs['mask_hits25']}／{bbs['mask_hits50']}，mIoU{bbs['mask_miou']:.8f}%。初始6887 bbs为{fit['initial']['bbs']['rec_hits25']}／{fit['initial']['bbs']['rec_hits50']}，终态为{fit['terminal']['bbs']['rec_hits25']}／{fit['terminal']['bbs']['rec_hits50']}。6887来自作者预训练见过场景的模块留出；9508为正式开发验证，不混算两者。

本组实际29778条fit各一次、3723更新、有效batch8（尾batch2）、seed2027、freshAdamW、LR1e-5、WD5e-4、clip0.1；只训练10项400614参数头，原G全部参数及持久buffer保持，原模块eval。输入为同一1302维完整范围／局部支撑；完整109维统计可见。普通中心／尺寸残差使用共同1e-6参考、最终尺寸下限，不把零头的一致性说成对原始负尺寸的逐位保持。原生最终BBox／GIoU监督部署框；语义和Mask路径冻结，最终bbs评分不变，全部256候选保留。没有六面分布监督、教师、P2、新对比目标或质量回写。

实际收取10份已完成配置、加载、初始／终态／正式、退出及清理元数据；正式rows SHA256为{intake['formal_rows_sha256']}。收取0次模型前向、0次优化更新、0份权重下载。controller的CPU所选框两阈值重算与正式bbs／bbf计数均0差异；此时尚未收取全量rows，也未重新计算原始Mask或Full256框，新上下文全配对核对仍待两组闭合。原G历史计数不冒充本轮独立重新评估。

依据用户既有清理授权，自有非最佳residual终点已核对恢复、完整9508评估、CPU计数及SHA后删除{deleted_bytes}字节；不创建失败权重的本地归档。原G父模型SHA仍为{retention['parent_sha256_after']}，此快照的指标最佳为{intake['retained_best']['name']}。收取时系统盘可用{intake['system_disk_free_bytes']}字节、数据盘{intake['data_disk_free_bytes']}字节、实验目录权重{len(intake['owned_weights'])}份；这些是收取快照。

controller继续distribution/train。保持同一完整范围／局部输入、原G固定协议及样本／更新预算，仅在另一组使用456102参数六面分布头和独立DFL/7。表示、监督及输出参数量共同改变，不宣称纯DFL效应或已经实现方向条件化face token解码。该组尚无正式9508结果；完整配对、同Query修复／破坏、起点跨进程差异、分布及目标体积分组均在实际闭合后核对。终态收集、真实审阅回执及发布工具已准备，未把准备称为执行或精度证据。

证据：refine-logs/pvground_boundary_distribution_20261004/residual_phase_complete及本次收取／发布源码。当前同检查点5615／4754和Nr3D／Sr3D目标仍未完成，原G及必要V99权重链继续保护。
'''
new = old + section.encode('utf-8')
prefix = 'refine-logs/pvground_boundary_distribution_20261004/'
payloads = {}
for path in sorted((local/'residual_phase_complete').rglob('*')):
    if path.is_file():
        payloads[prefix+path.relative_to(local).as_posix()] = path.read_bytes()
names = ('collect_residual_phase_authorized.py','residual_phase_collector_preparation.json',
    'prepare_residual_phase_publisher.py','publish_residual_phase.py',
    'residual_phase_publisher_preparation.json','observation_04.json','observation_05.json',
    'record_terminal_audit_request.py','record_terminal_audit_result.py',
    'terminal_audit_recorders_preparation.json','prepare_terminal_publisher.py',
    'publish_terminal.py','terminal_publisher_preparation.json','terminal_publisher_trace_update.json',
    'terminal_publisher_phase_update.json')
for name in names:
    payloads[prefix+name] = (local/name).read_bytes()
assert all(not (repo/relative).exists() for repo in repos[:2] for relative in payloads)
client = paramiko.SSHClient(); client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',
    password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp = client.open_sftp(); remote = '/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(remote+'/'+doc,'rb') as stream:
    assert stream.read() == old
directories = sorted({str(Path(relative).parent).replace('\\','/') for relative in payloads},
    key=lambda item:(item.count('/'),item))
created = {'refine-logs','refine-logs/pvground_boundary_distribution_20261004'}
for directory in directories:
    parts = directory.split('/')
    for count in range(1,len(parts)+1):
        current = '/'.join(parts[:count])
        if current not in created:
            sftp.mkdir(remote+'/'+current); created.add(current)
for relative, raw in payloads.items():
    for repo in repos[:2]:
        target = repo/relative; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(raw)
    with sftp.open(remote+'/'+relative,'wx') as stream:
        stream.write(raw)
    with sftp.open(remote+'/'+relative,'rb') as stream:
        assert stream.read() == raw
for path in copies:
    path.write_bytes(new)
with sftp.open(remote+'/'+doc,'wb') as stream:
    stream.write(new)
assert all(path.read_bytes() == new for path in copies)
with sftp.open(remote+'/'+doc,'rb') as stream:
    assert stream.read() == new
sftp.close(); client.close()
heads = []
for index, repo in enumerate(repos):
    if index < 2:
        with (repo/'MANIFEST.md').open('a',encoding='utf-8') as stream:
            stream.write(f'\n- {stamp} frozen-G residual complete9508 bbs{bbs["rec_hits25"]}/{bbs["rec_hits50"]}; distribution still active, controller CPU proof only, owned nonbest retired.\n')
    stage = [doc] + (['MANIFEST.md',*payloads] if index < 2 else [])
    changed = subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git','-C',str(repo),'add','-f','--',*stage])
    subprocess.check_call(['git','-C',str(repo),'-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check','--',*stage])
    for relative, raw in (payloads.items() if index < 2 else []):
        assert subprocess.check_output(['git','-C',str(repo),'show',':'+relative]) == raw
    subprocess.check_call(['git','-C',str(repo),'commit','-m','Record completed residual boundary control while distribution arm runs'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'-c','http.version=HTTP/1.1','push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0] == heads[0]
for relative in (doc,*payloads):
    assert subprocess.check_output(['git','-C',str(repos[0]),'rev-parse','HEAD:'+relative]) == subprocess.check_output(['git','-C',str(repos[1]),'rev-parse','HEAD:'+relative])
digest = hashlib.sha256(new).hexdigest()
guard.write_bytes(guard_raw.replace(previous['handoff_sha256'].encode(),digest.encode()))
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.44',
    heads=heads,github_main=heads[0],handoff_sha256=digest,handoff_bytes=len(new),
    four_local_and_remote_equal=True,exact_committed_payloads=True,payload_count=len(payloads),
    residual_bbs_hits=[bbs['rec_hits25'],bbs['rec_hits50']],distribution_formal_complete=False,
    retained_metric_best=intake['retained_best']['name'],owned_deleted_bytes=deleted_bytes,
    weights_downloaded=0,fresh_pair_audit_complete=False,goal_achieved=False)
(local/'residual_phase_publication.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
with (workspace/'memory/2026-10-04.md').open('a',encoding='utf-8') as stream:
    stream.write(f'\nPVGround {record["time_cst"]} section20.376.44 published {heads[0]} fourlocal+remote SHA{digest}; residual9508 bbs{bbs["rec_hits25"]}/{bbs["rec_hits50"]}, distribution pending; controllerCPUproof only, {deleted_bytes}B verified owned retired/no archives. GoalACTIVE_UNMET.\n')
print(json.dumps(record),flush=True)
