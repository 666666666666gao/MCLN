"""Publish the completed pair and actual fresh audit; do not run models."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import paramiko

local = Path(__file__).parent
assert not (local/'terminal_publication.json').exists()
previous = json.loads((local/'local_progress_publication.json').read_bytes())
intake = json.loads((local/'complete/INTAKE.json').read_bytes())
audit = json.loads((local/'analysis/EXPERIMENT_AUDIT.json').read_bytes())
assert intake['status']['status'] == 'complete' and intake['controller_exit'] == 0
assert not intake['controller_alive'] and intake['weights_downloaded'] == 0
assert audit['verdict'] == 'WARN' and not audit['blocking_issues']
assert intake['original_g_sha256'] == '0575dfae333dabdf288a470fc09d8967f9867d1853a5f61d9cfc3cbcf7964522'
for arm, hits in [('local_range', (5603,4428)), ('whole_range', (5594,4461))]:
    receipt = json.loads((local/'complete'/arm/'formal/receipt.json').read_bytes())
    assert receipt['rows'] == 9508
    assert tuple(receipt['metrics']['bbs'][key] for key in ('rec_hits25','rec_hits50')) == hits
for name, entry in intake['files'].items():
    raw = (local/'complete'/name).read_bytes()
    assert len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['sha256']

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
section = '''

## 20.376.39 完整范围来源对照终态、逐行重算与新鲜审查（%s）

原controller403326于2026-10-04T05:07:39.434317+08:00完整结束，exit0，四阶段均正常完成；唯一观察器5205与collector13529均已自然关闭。whole训练阶段12817.98秒包含加载、6887初始／终态评估和实际fit，正式阶段1498.58秒；不将阶段耗时冒称纯训练GPU成本。原训练、源码和预算未中途改变，也没有重跑成功的预检或终态。

| 数据／模式 | local @.25／.50 | whole @.25／.50 | whole减local |
|---|---:|---:|---:|
| 6887初始bbs | 6176／5602 | 6176／5602 | 0／0 |
| 6887终态bbs | 6160／5591 | 6156／5559 | −4／−32 |
| 9508正式bbs | 5603／4428 | 5594／4461 | −9／+33 |
| 9508正式bbf | 5637／4431 | 5603／4458 | −34／+27 |

正式bbs：local58.9293%%／46.5713%%，whole58.8347%%／46.9184%%；同阈值whole相对local严格383修复／350破坏，净+33。相对记录的原G5615／4495，local−12／−67、whole−21／−34。原G仍是当前PV指标最佳；同检查点5615／4754目标未完成，不晋级这项方法，也没有新增Nr3D／Sr3D结果。

正式bbs Mask：local5806／5114、mIoU46.93181071%%；whole5811／5106、46.92723284%%，分别记录，不拼接模式或模型最好列。固定各自已选Query，local粗框→末次精修严格4428→4428（14修复／14破坏），whole4471→4461（13／23）。最大单面移动中位数local2.705010毫米、whole2.828427毫米，9508已选答案均小于1厘米。两模型间+33不能解释为末次精修直接修复33条；内部粗框已经共同训练，不是独立baseline。

严格错误中Full256仍有合格框／全缺：local3323／1757，whole3311／1736；Full256严格上界7751／7772。GT仅用于离线诊断，不能部署，也不能把几何重叠视为真实语义身份或可全部兑现的成绩。保留全部256候选，不按当前低排名一律裁掉。

两组同原G、freshAdamW、seed2027、有效batch8、LR1e-5、WD5e-4、clip0.1；各29778条fit一次、3723更新（最后batch2），实际batch行顺序一致。两组同400614参数／1302输入头，唯一计划来源变化是109完整范围统计可见或置零；零输入不意味着两者有效表达能力相同。6887为预训练见过场景留出，9508为开发验证；单seed不提供跨seed显著性结论。

跨进程起点并非逐位一致：每模式6881行框及5943行连续IoU不同；bbs选Query无变化，bbf有1行不同；初始REC阈值位图均相同，连续值／Mask及上界仍有差异。共享11份Python源码一致，实际差异如实保留，不称完美逐位配对或R／Mask来源独立因果增量。

收取实际63份文件，不下载权重；保存六套评估逐行输出、7446次实际fit记录、配置、源码、日志和退役回执。新鲜gpt-6-astra／max审查独立上下文、同模型家族，后端具体SKU未独立证实，接受状态provisional：WARN、0阻断。读取81份primary文件共63476687字节，343项CPU检查通过；186256次所选final／coarse框IoU重建与原生阈值无差异。未重建原始Mask／完整候选框／全部排序／模型优化器张量；原G计数为历史记录，未在本轮重新评估。审查不启动SSH／GPU或删除权重，不以执行者判断冒称独立审查。

按用户持续清理授权，controller分别于01:08:58.581941、05:07:39.433324仅删除本轮local／whole terminal.pth各347116945字节，共694233890字节；删除前已恢复评估、校验对应SHA和行输出，并CPU重算bbs／bbf所选框阈值。原G全SHA保持0575dfae333dabdf288a470fc09d8967f9867d1853a5f61d9cfc3cbcf7964522；无失败本地权重归档，本轮owned权重0。05:10:32快照可用2568286208字节，该值不是实时监控；保护父模型／V99和全部文字／逐行证据。

下一步遵循“whole相对local有信号、仍低于原G”的条件分支：先检查现有新头在固定原G表示上能否学出有效修正，再推进方向保留的六面边界。新冻结协议仅本地准备，尚未部署／GPU预检／训练：所有原G参数及运行状态固定eval，只训练现有10项精修参数，保留同头、同数据、同预算、同native几何监督和单一评分，不增加教师／质量／新loss。新协议必须先经新鲜源码审查和真实两步检查；旧联合预检不会被冒称这一新协议已通过。最终几何回写与训练期教师仍为后续独立变量。

证据目录：refine-logs/pvground_whole_mask_fit_20261003/complete、analysis、terminal_audit_trace；分析REPORT／审查EXPERIMENT_AUDIT保留实际限度。历史§20.376.38的whole活动状态是当时快照，本节为更新终态。
''' % stamp
new = old + section.encode('utf-8')
prefix = 'refine-logs/pvground_whole_mask_fit_20261003/'
payloads = {}
for directory in ('complete','analysis'):
    for path in sorted((local/directory).rglob('*')):
        if path.is_file():
            payloads[prefix+path.relative_to(local).as_posix()] = path.read_bytes()
for path in sorted((local/'.aris/traces/experiment-audit/2026-10-04_terminal_run01').iterdir()):
    if path.is_file():
        payloads[prefix+'terminal_audit_trace/'+path.name] = path.read_bytes()
for name in ('prepare_terminal_audit_request.py','TERMINAL_AUDIT_REQUEST.txt',
             'record_terminal_audit_request.py','record_terminal_audit_result.py',
             'publish_terminal.py','observation_18.json'):
    payloads[prefix+name] = (local/name).read_bytes()
record = dict(time_cst=stamp, section='20.376.39', status='PAIR_COMPLETE_ANALYZED_AUDITED_WARN0',
    local_bbs_hits=[5603,4428],whole_bbs_hits=[5594,4461],whole_minus_local=[-9,33],rows=9508,
    original_g_hits=[5615,4495], retained_best='original_g',goal_achieved=False,
    owned_endpoint_deleted_bytes=694233890,weights_downloaded=0,local_failed_archive_created=False,
    audit_verdict='WARN',audit_blocking_issues=0,review_independence='same-family',acceptance_status='provisional',
    next_control_status='LOCAL_PREPARATION_ONLY_NOT_REVIEWED_DEPLOYED_OR_TRAINED')
payloads[prefix+'terminal_observed.json'] = (json.dumps(record,indent=2)+'\n').encode()
assert all(not (repo/relative).exists() for repo in repos[:2] for relative in payloads)

client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp=client.open_sftp();remote='/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(remote+'/'+doc,'rb') as stream:
    assert stream.read()==old
directories=sorted({str(Path(relative).parent).replace('\\','/') for relative in payloads},key=lambda x:(x.count('/'),x))
created=set(['refine-logs','refine-logs/pvground_whole_mask_fit_20261003'])
for directory in directories:
    parts=directory.split('/')
    for count in range(1,len(parts)+1):
        current='/'.join(parts[:count])
        if current not in created:
            sftp.mkdir(remote+'/'+current);created.add(current)
for relative,raw in payloads.items():
    for repo in repos[:2]:
        target=repo/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    with sftp.open(remote+'/'+relative,'wx') as stream:
        stream.write(raw)
    with sftp.open(remote+'/'+relative,'rb') as stream:
        assert stream.read()==raw
for path in copies:path.write_bytes(new)
with sftp.open(remote+'/'+doc,'wb') as stream:stream.write(new)
assert all(path.read_bytes()==new for path in copies)
with sftp.open(remote+'/'+doc,'rb') as stream:assert stream.read()==new
sftp.close();client.close()
heads=[]
for index,repo in enumerate(repos):
    if index<2:
        with (repo/'MANIFEST.md').open('a',encoding='utf-8') as stream:
            stream.write('\n- '+stamp+' actual local/whole complete9508:5603/4428 versus5594/4461, originalG retained, fresh terminal audit WARN0; owned nonbest endpoints retired.\n')
    stage=[doc]+(['MANIFEST.md',*payloads] if index<2 else [])
    changed=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git','-C',str(repo),'add','-f','--',*stage])
    subprocess.check_call(['git','-C',str(repo),'-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check','--',*stage])
    for relative,raw in (payloads.items() if index<2 else []):
        assert subprocess.check_output(['git','-C',str(repo),'show',':'+relative])==raw
    subprocess.check_call(['git','-C',str(repo),'commit','-m','Record complete local whole range comparison and fresh integrity audit'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0]==heads[0]
for relative in (doc,*payloads):
    assert subprocess.check_output(['git','-C',str(repos[0]),'rev-parse','HEAD:'+relative])==subprocess.check_output(['git','-C',str(repos[1]),'rev-parse','HEAD:'+relative])
digest=hashlib.sha256(new).hexdigest()
guard.write_bytes(guard_raw.replace(previous['handoff_sha256'].encode(),digest.encode()))
record.update(time_cst=datetime.datetime.now().astimezone().isoformat(),heads=heads,github_main=heads[0],handoff_sha256=digest,handoff_bytes=len(new),four_local_and_remote_equal=True,exact_committed_payloads=True,payload_count=len(payloads))
(local/'terminal_publication.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
with (workspace/'memory/2026-10-04.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+record['time_cst']+' complete source-pair publication '+heads[0]+' §20.376.39/fourlocal+remote exact SHA'+digest+'; actualbbs local5603/4428 whole5594/4461, originalGbest5615/4495, auditWARN0 samefamily/provisional, owned694233890B retired/noarchives; nexthead-only localprepared/notreviewed/notlaunched. GoalACTIVE_UNMET.\n')
print(json.dumps(record),flush=True)
