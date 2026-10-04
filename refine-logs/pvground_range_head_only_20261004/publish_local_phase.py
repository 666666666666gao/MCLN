"""Publish completed local-arm receipts while the whole arm remains in progress."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

import paramiko

local = Path(__file__).resolve().parent
assert not (local/'local_phase_publication.json').exists()
previous = json.loads((local/'launch_publication.json').read_bytes())
intake = json.loads((local/'local_phase_complete/INTAKE.json').read_bytes())
fit = json.loads((local/'local_phase_complete/receipt.json').read_bytes())
formal = json.loads((local/'local_phase_complete/formal/receipt.json').read_bytes())
retention = json.loads((local/'local_phase_complete/weight_retention.json').read_bytes())
assert fit['training_steps'] == 3723 and fit['fit_rows'] == 29778
assert fit['head_only'] and fit['original_g_state_unchanged'] and fit['upstream_running_state_eval']
assert formal['status'] == 'pass' and formal['rows'] == 9508
assert not intake['fresh_independent_row_recount']
assert intake['stage'] == 'whole_range/train' and len(intake['completed']) == 2
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

## 20.376.41 固定原G范围头对照：local已完成，whole仍活动（{stamp}）

本节为第一组完成后的及时记录，不是local／whole配对终态。local训练含初始／终态6887模块留出于{intake['local_train_finished_cst']}结束；9508开发验证于{intake['local_formal_finished_cst']}结束。严格恢复10项head delta及其保护父模型后，原生bbs命中{bbs['rec_hits25']}／{bbs['rec_hits50']}，即{100*bbs['rec_hits25']/9508:.4f}%／{100*bbs['rec_hits50']/9508:.4f}%；bbf单独记录{bbf['rec_hits25']}／{bbf['rec_hits50']}，不拼接两模式最好列。

相对历史原G5615／4495，bbs变化{bbs['rec_hits25']-5615:+d}／{bbs['rec_hits50']-4495:+d}。Mask bbs命中{bbs['mask_hits25']}／{bbs['mask_hits50']}，mIoU{bbs['mask_miou']:.8f}%。初始6887 bbs为{fit['initial']['bbs']['rec_hits25']}／{fit['initial']['bbs']['rec_hits50']}，终态{fit['terminal']['bbs']['rec_hits25']}／{fit['terminal']['bbs']['rec_hits50']}；6887预训练见过场景留出与9508开发验证不能混算。

本组实际29778条fit各一次、3723更新、有效batch8（尾batch2）、seed2027、freshAdamW／LR1e-5／WD5e-4／clip0.1，10项400614范围头训练，原G所有参数及持久buffer完全保持，原模块eval。冻结协议同时改变共同适配、运行统计及Dropout等，不能仅归因为冻结某一参数的独立效果。native最终BBox／GIoU训练部署精修框，语义／Mask与其109范围观察冻结；local将109范围置零、保留同头及局部融合Mask支撑。

在{intake['time_cst']}收取10份已完成local配置／加载／初始终态／正式／退出／清理元数据；模型重放0、优化更新0、权重下载0，正式rows SHA256与controller记录一致。controller已CPU重算bbs／bbf所选框，与原生两阈值0变化。该检查是controller回执，尚不是新鲜独立全配对审查；没有收取本组全量rows，也没有重新计算原始Mask或Full256候选。本节不把历史原G计数称作本轮重新评估。

08:23计划只读观察曾报SSH banner EOF，原观察器82422自然exit1；未推断远端训练失败、未重启训练。仅将观察器沿原序列恢复，延迟240秒后08:28实际重新连接确认controller450028仍活动、whole训练进程460864，随后收取上述完成阶段元数据。一次连接错误不被记录成模型负结果，也未给训练添加重试／fallback。

依据既有清理授权，local退役自有非最佳权重{deleted_bytes}字节；未创建失败本地权重归档，保护父模型SHA仍为{retention['parent_sha256_after']}。此快照保留的指标最佳为{intake['retained_best']['name']}；系统盘可用{intake['system_disk_free_bytes']}字节、数据盘{intake['data_disk_free_bytes']}字节，owned活动权重{len(intake['owned_weights'])}份。这是收取时快照，不是之后实时磁盘状态。

controller继续执行whole_range/train，不修改活动源码、预算或损失，不新增GPU诊断／重复预检／P2／教师／质量／分布头。whole完整9508结果尚无，完整范围相对local的收益、同Query修复／破坏、跨进程初始差异及新鲜配对审查留到完整闭环；不能用第一组结果提前宣称109范围有效。ScanRefer同检查点5615／4754与Nr3D／Sr3D目标仍未完成，原G和历史V99仍保留。

证据：refine-logs/pvground_range_head_only_20261004/local_phase_complete；实际收取器和本节发布器也保留，未改已审查训练源码。
'''
new = old + section.encode('utf-8')
prefix = 'refine-logs/pvground_range_head_only_20261004/'
payloads = {}
for path in sorted((local/'local_phase_complete').rglob('*')):
    if path.is_file():
        payloads[prefix+path.relative_to(local).as_posix()] = path.read_bytes()
for name in ('collect_local_phase_authorized.py','local_phase_collector_preparation.json',
    'publish_local_phase.py','progress_observation_06_summary.json','observation_08.json',
    'resume_fit_observer_after_banner.py','observer_banner_failure.json','observer_resume_launch.json'):
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
created = {'refine-logs','refine-logs/pvground_range_head_only_20261004'}
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
            stream.write(f'\n- {stamp} head-only local complete9508 bbs{bbs["rec_hits25"]}/{bbs["rec_hits50"]}; whole still active, controller CPU proof only, owned nonbest retired.\n')
    stage = [doc] + (['MANIFEST.md',*payloads] if index < 2 else [])
    changed = subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git','-C',str(repo),'add','-f','--',*stage])
    subprocess.check_call(['git','-C',str(repo),'-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check','--',*stage])
    for relative, raw in (payloads.items() if index < 2 else []):
        assert subprocess.check_output(['git','-C',str(repo),'show',':'+relative]) == raw
    subprocess.check_call(['git','-C',str(repo),'commit','-m','Record completed local range head control while whole arm runs'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0] == heads[0]
for relative in (doc,*payloads):
    assert subprocess.check_output(['git','-C',str(repos[0]),'rev-parse','HEAD:'+relative]) == subprocess.check_output(['git','-C',str(repos[1]),'rev-parse','HEAD:'+relative])
digest = hashlib.sha256(new).hexdigest()
guard.write_bytes(guard_raw.replace(previous['handoff_sha256'].encode(),digest.encode()))
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.41',
    heads=heads,github_main=heads[0],handoff_sha256=digest,handoff_bytes=len(new),
    four_local_and_remote_equal=True,exact_committed_payloads=True,payload_count=len(payloads),
    local_bbs_hits=[bbs['rec_hits25'],bbs['rec_hits50']],whole_formal_complete=False,
    retained_metric_best=intake['retained_best']['name'],owned_deleted_bytes=deleted_bytes,
    weights_downloaded=0,fresh_pair_audit_complete=False,goal_achieved=False)
(local/'local_phase_publication.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
with (workspace/'memory/2026-10-04.md').open('a',encoding='utf-8') as stream:
    stream.write(f'\nPVGround {record["time_cst"]} §20.376.41 published {heads[0]} fourlocal+remote SHA{digest}; actualheadonly local9508 bbs{bbs["rec_hits25"]}/{bbs["rec_hits50"]}, wholepending; controllerCPUproof notfreshaudit, {deleted_bytes}B verifiedownedretired/noarchives. GoalACTIVE_UNMET.\n')
print(json.dumps(record),flush=True)
