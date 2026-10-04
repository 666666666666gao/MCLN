"""Publish actual closed readback results; retain canonical raw data on the data disk."""
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess

import paramiko


local = Path(__file__).resolve().parent
formal = local / 'formal_draft'
face = local.parent / 'pvground_face_conditioned_20261004'
receipt_path = local / 'terminal_publication_20261005.json'
assert not receipt_path.exists()
previous_path = local / 'hidden_control_publication_20261005.json'
previous = json.loads(previous_path.read_bytes())
intake = json.loads((formal / 'complete/INTAKE.json').read_bytes())
summary = json.loads((formal / 'analysis/SUMMARY.json').read_bytes())
review_call = json.loads((formal / 'analysis/TERMINAL_REVIEW_CALL.json').read_bytes())
review = json.loads((formal / 'analysis/EXPERIMENT_AUDIT.json').read_bytes())
assert review_call['actual_spawn_return_received'] and review_call['result_received']
assert review_call['returned_task_name'] == '/root/pvg_readback_terminal_integrity'
assert review['review_independence'] == 'same-family'
assert review['acceptance_status'] == 'provisional'
assert review['verdict'].upper() in ('PASS', 'WARN', 'FAIL')
terminal = intake['remote_terminal']
assert terminal['status']['status'] == 'complete' and terminal['exitcode'] == 0
assert not terminal['controller_alive'] and intake['downloaded_weights'] == 0
assert not intake['inference_or_optimizer_replayed']
assert summary['training_order_exact'] and summary['updates_per_arm'] == 3723
assert summary['samples_per_arm'] == 29778 and summary['effective_batch'] == 8
assert [(r['rec_hits25'], r['rec_hits50']) for r in summary['table']] == [(5616,4506),(5616,4475),(5615,4477)]
assert summary['retained_best']['bbs_hits50'] == 4506 and not summary['target_pass']
assert len(intake['files']) == 40
workspace = Path('C:/Users/gb')
repos = [workspace / '.codex_mcln_g0_20260905', workspace / '.codex_pvground_cs_20261002', workspace / '.codex_mcln_v99_internal_20260928']
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies = [repo / doc for repo in repos] + [workspace / 'Desktop/document' / Path(doc).name]
old = copies[0].read_bytes()
assert hashlib.sha256(old).hexdigest() == previous['handoff_sha256']
assert all(path.read_bytes() == old for path in copies)
assert b'## 20.376.57 ' not in old
for repo, head in zip(repos, previous['heads']):
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip() == head
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
guard = workspace / '.codex/tmp/sync_cs_handoff_remote_20260923.py'
guard_raw = guard.read_bytes()
assert guard_raw.count(previous['handoff_sha256'].encode()) == 1

prefix = 'refine-logs/pvground_geometry_readback_20261004/'
payloads = {}
for relative, identity in intake['files'].items():
    raw = (formal / 'complete' / relative).read_bytes()
    assert len(raw) == identity['bytes'] and hashlib.sha256(raw).hexdigest() == identity['sha256']
    payloads[prefix + 'formal_draft/complete/' + relative] = raw
payloads[prefix + 'formal_draft/complete/INTAKE.json'] = (formal / 'complete/INTAKE.json').read_bytes()
for path in sorted((formal / 'analysis').iterdir()):
    if path.is_file():
        payloads[prefix + 'formal_draft/analysis/' + path.name] = path.read_bytes()
payloads[prefix + 'formal_draft/wait.json'] = (formal / 'wait.json').read_bytes()
payloads[prefix + Path(__file__).name] = Path(__file__).read_bytes()
raw_large = {p:v for p,v in payloads.items() if PurePosixPath(p).suffix in ('.jsonl','.log')}
remote_payloads = {p:v for p,v in payloads.items() if p not in raw_large}

client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro',port=33476,username='root',password=os.environ['MCLN_SSH_PASSWORD'],timeout=30)
sftp = client.open_sftp()
remote_project = '/home/gb/new butd/butd_detr-main/MCLN-main'
remote_run = '/root/autodl-tmp/pvground_readback_fit_20261004'
with sftp.open(remote_project + '/' + doc,'rb') as stream:
    assert stream.read() == old
retentions = []
for arm in ('evidence_hidden','evidence_visible'):
    names = sftp.listdir(remote_run + '/' + arm)
    assert 'terminal.pth' not in names and 'latest.pth' not in names
    raw = (formal / 'complete' / arm / 'weight_retention.json').read_bytes()
    with sftp.open(remote_run + '/' + arm + '/weight_retention.json','rb') as stream:
        assert stream.read() == raw
    retention = json.loads(raw)
    assert retention['required_parent_chain_preserved'] and retention['cpu_box_threshold_changes'] == 0
    assert not retention['local_weight_archive_created']
    assert len(retention['deleted']) == 1 and retention['deleted'][0]['bytes'] == 1284112
    retentions.append(retention)
assert sftp.stat(summary['retained_best']['path']).st_size == 5587141
parent_paths = [
    '/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260918_semantic_assignment_v1/terminal.pth',
    '/root/autodl-tmp/mcln_pvground_scan_checkpoint_inspection_20260908_v1/PV-Ground_ScanRefer.pth']
for path in parent_paths:
    assert sftp.stat(path).st_size > 0
probe = "import json,os; print(json.dumps({p:os.statvfs(p).f_bavail*os.statvfs(p).f_frsize for p in ['/root/autodl-tmp','/home/gb']}))"
runtime = json.loads((formal/'evidence_hidden_fit_spec.json').read_bytes())['runtime']
_, stdout, stderr = client.exec_command(shlex.join([runtime+'/venv/bin/python','-c',probe]),timeout=60)
free = json.loads(stdout.read())
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
layout = dict(canonical_remote_run=remote_run, canonical_raw_intake='formal_draft/complete/INTAKE.json',
    github_raw_results_included=True, remote_project_large_raw_results_duplicated=False,
    large_raw_payload_count=len(raw_large), large_raw_bytes=sum(map(len,raw_large.values())),
    system_free_bytes_before=free['/home/gb'], data_free_bytes_before=free['/root/autodl-tmp'],
    decision='Keep large original rows/logs on the data disk; publish exact small receipts, code, audit and SHA256 index to the project directory.',
    weights_downloaded=0, new_inference_or_optimizer_updates=0)
layout_raw = (json.dumps(layout,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
(formal/'analysis/STORAGE_LAYOUT.json').write_bytes(layout_raw)
payloads[prefix+'formal_draft/analysis/STORAGE_LAYOUT.json'] = layout_raw
remote_payloads[prefix+'formal_draft/analysis/STORAGE_LAYOUT.json'] = layout_raw
stamp = datetime.datetime.now().astimezone().isoformat()
verdict = review['verdict'].upper()
visible = summary['stages']['evidence_visible']['formal']
direct = visible['fixed_frame_readback_effect']['0.5']
paired = summary['visible_vs_hidden']['formal']['0.5']
deleted_bytes = sum(item['bytes'] for retention in retentions for item in retention['deleted'])
section = f'''

## 20.376.57 固定几何后的回写对照已完成：可见证据未形成严格定位增量（{stamp}）

本节接续§20.376.54—56的结构、预检与启动记录，只追加实际完整终态。两组均从受保护的4506几何父模型构建，官方PV、原G和几何头保持冻结/eval；R均为96672参数、23项状态，只有44维几何证据是否可见不同。实际各遍历29778条fit一次，有效batch8、尾批2、3723次更新；完整训练行顺序一致。没有质量目标、教师、额外对比正例或第二套部署排名。

第二组训练与6887条模块留出于2026-10-05T03:18:21.066848+08:00结束；独立9508条正式验证于2026-10-05T03:43:12.687074+08:00结束。控制器完成核对及保留处理于{summary['actual_finished_cst']}结束，退出码0；原观察器66222已关闭。6887条是预训练见过场景的模块留出，不能替代下表正式结果。

| 同一检查点的原生last/bbs | Acc@0.25 | Acc@0.50 | 相对保护父模型严格命中 |
|---|---:|---:|---:|
| 保护的4506几何父模型 | 59.0660%（5616） | 47.3917%（4506） | — |
| 隐藏几何证据控制 | 59.0660%（5616） | 47.0656%（4475） | −31 |
| 可见几何证据回写 | 59.0555%（5615） | 47.0867%（4477） | −29 |

可见对隐藏严格修复{paired['repairs']}、破坏{paired['damages']}，净{paired['net']:+d}；这2条不是超过起点的新能力。两组正式Full256资格标记逐行一致，但不把两个独立完整前向的所有浮点值描述为逐位相同。可见组原生Mask为{visible['mask_hits25']/9508*100:.4f}%／{visible['mask_hits50']/9508*100:.4f}%／{visible['mask_miou']:.4f}% mIoU，和框使用同一个所选Query。

可见组同一前向、同一候选框的缓存语义头回放显示：R严格修复{direct['repairs']}、破坏{direct['damages']}，净{direct['net']:+d}；@0.25为55／56、净−1。部署语义头只调用一次；诊断另对缓存的回写前Query执行一次头回放。它是当前共同训练检查点的直接前向诊断，不是独立训练消融。父模型持久状态冻结验证通过，因此本轮不能再以共同适配破坏几何来解释负结果。

可见组5031条严格错误中，{visible['candidate_availability']['0.5']['errors_with_good_full256']}条仍有Full256合格框，{visible['candidate_availability']['0.5']['errors_without_good_full256']}条没有；Full256严格上界为7890。上界和GT体积分组只用于离线诊断，不进入推理。继续保留全部256候选，当前低分不等于候选应被删除。

本次实际新鲜结果完整性审查为{verdict}，报告在formal_draft/analysis/EXPERIMENT_AUDIT.md与.json。它属于独立上下文、同模型族的provisional复核；工具请求Astra/max，但没有实际后端路由证明。原始报告、CPU重算和文件身份清单保留各自检查范围，不将保存的Mask IoU或Full256布尔标记重数冒充重新计算完整Mask／候选张量，也不把来源预检PASS冒充精度PASS。

按用户持续授权，两份较差R终点在实际严格模型/AdamW恢复、9508行CPU阈值核对后已删除，共{deleted_bytes}字节；本次再次确认两组均无terminal/latest权重。本地收集40份文本／逐行证据（51378753字节），没有权重归档，没有新增推理或优化步。4506最佳及官方PV／原G重建父权重保留，历史V99链不改动。

完整逐行证据进入主工作树与GitHub的{prefix}formal_draft/complete/；远端原始数据留在{remote_run}。系统盘实测余量{free['/home/gb']}字节，约49MiB的逐行结果／日志保留在数据盘原址；项目目录同步精确摘要、审查、代码和SHA256索引，避免重复大文件。布局见analysis/STORAGE_LAYOUT.json。

当前保护最好仍为5616／4506，达到同一模型5615／4754的开发目标仍差248个严格净命中。普通证据回写在本配方下没有增量；后续不继续扩大普通attention或改变正例分母。下一项应直接研究最终框质量如何监督实际native bbs，在已匹配root与原G合格未匹配集合上明确实例接纳和范围优劣、保护其他已匹配实例，采用独立可检查的监督预算；具体损失尚待接线及独立控制验证，不能写成已实现或已有效。原生bbs包含token softmax后的正/修饰/代词/关系证据及其他实体减项，不能直接当成无界logit或普通[0,1]类别概率套用。模型、全部256候选及单一部署评分保持明确。

本节只报告单seed2027的ScanRefer结果；50%目标未达，Nr3D/Sr3D新结构训练和三个验证有效创新点仍未成立。复核与原始负结果保留为下一步决策证据，不改写为成功。
'''
new = old + section.encode('utf-8')
assert new.startswith(old) and new.count(b'## 20.376.57 ') == 1
remote_dirs = sorted({str(PurePosixPath(remote_project+'/'+p).parent) for p in remote_payloads})
_, stdout, stderr = client.exec_command(shlex.join(['mkdir','-p','--',*remote_dirs]),timeout=60)
assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
for relative, raw in payloads.items():
    for repo in repos[:2]:
        destination = repo / relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(raw)
for relative, raw in remote_payloads.items():
    with sftp.open(remote_project+'/'+relative,'wb') as stream:
        stream.write(raw)
    with sftp.open(remote_project+'/'+relative,'rb') as stream:
        assert stream.read() == raw
for path in copies:
    path.write_bytes(new)
with sftp.open(remote_project+'/'+doc,'wb') as stream:
    stream.write(new)
with sftp.open(remote_project+'/'+doc,'rb') as stream:
    assert stream.read() == new
sftp.close()
client.close()
heads = []
for index, repo in enumerate(repos):
    stage = [doc]
    if index < 2:
        with (repo/'MANIFEST.md').open('a',encoding='utf-8') as stream:
            stream.write('\n- '+stamp+' actual closed readback9508 hidden5616/4475 visible5615/4477; sameframe visible66fix95damage; best4506 protected, two inferior deltas removed; fresh same-family provisional audit '+verdict+'.\n')
        stage += ['MANIFEST.md',*payloads]
    changed = subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git','-C',str(repo),'add','-f','--',*stage])
    subprocess.check_call(['git','-C',str(repo),'diff','--cached','--check','--',*stage])
    for relative, raw in (payloads.items() if index < 2 else []):
        assert subprocess.check_output(['git','-C',str(repo),'show',':'+relative]) == raw
    subprocess.check_call(['git','-C',str(repo),'commit','--quiet','-m','Record completed frozen-geometry readback comparison and actual integrity audit'])
    heads.append(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip())
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
subprocess.check_call(['git','-C',str(repos[0]),'-c','http.version=HTTP/1.1','push','origin','HEAD:main'])
assert subprocess.check_output(['git','-C',str(repos[0]),'ls-remote','origin','refs/heads/main']).decode().split()[0] == heads[0]
digest = hashlib.sha256(new).hexdigest()
assert all(path.read_bytes() == new for path in copies)
guard.write_bytes(guard_raw.replace(previous['handoff_sha256'].encode(),digest.encode()))
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(),section='20.376.57',predecessor=str(previous_path),
    heads=heads,github_main=heads[0],handoff_sha256=digest,handoff_bytes=len(new),four_local_and_remote_equal=True,
    exact_committed_payloads=True,payload_count=len(payloads),remote_small_payload_count=len(remote_payloads),
    canonical_remote_raw=remote_run,remote_large_raw_duplicate=False,formal_pair_complete=True,
    hidden_formal_hits=[5616,4475],visible_formal_hits=[5615,4477],retained_best_hits=[5616,4506],
    actual_finished_cst=summary['actual_finished_cst'],fresh_terminal_audit_verdict=verdict,
    review_independence='same-family',acceptance_status='provisional',backend_attested=False,
    inferior_weights_removed_bytes=deleted_bytes,weights_downloaded=0,inference_or_optimizer_replayed=False,goal_achieved=False)
receipt_path.write_bytes((json.dumps(record,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
state_path = face/'active_continuation_state.json'
state = json.loads(state_path.read_bytes())
state.update(time_cst=record['time_cst'],status='READBACK_PAIR_TERMINAL_AUDITED_PUBLISHED',latest_publication=str(receipt_path),
    published_heads=heads,handoff_section='20.376.57',handoff_sha256=digest,handoff_bytes=len(new),
    fresh_formal_terminal_audit_complete=True,active_terminal_reviewer=None,terminal_audit=str(formal/'analysis/EXPERIMENT_AUDIT.json'),
    last_goal_turn_classification='PROGRESS_ACTUAL_TERMINAL_PAIR_AUDITED_PUBLISHED')
state_path.write_bytes((json.dumps(state,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
for path in (face/'NEXT_CONTINUATION.md',local/'NEXT_CONTINUATION.md'):
    body = path.read_text(encoding='utf-8')
    start = body.index('Latest actual publication:')
    end = body.index('\n\nProtected best',start)
    body = body[:start]+('Latest actual publication: '+str(receipt_path)+'\nMain '+heads[0]+'; section20.376.57, '+str(len(new))+'bytes; SHA256 '+digest+'. Fourlocal+remote exact. Never rerun completed publishers.')+body[end:]
    start = body.index('Fresh terminal integrity reviewer')
    end = body.index('\nAuthoritative raw:',start)
    body = body[:start]+('Fresh terminal integrity review actually completed: '+verdict+', same-family/provisional, requestedAstra/max not backend-attested. Actual report: '+str(formal/'analysis/EXPERIMENT_AUDIT.json'))+body[end:]
    path.write_bytes(body.encode('utf-8'))
assert (face/'NEXT_CONTINUATION.md').read_bytes() == (local/'NEXT_CONTINUATION.md').read_bytes()
with (workspace/'memory/2026-10-05.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+record['time_cst']+': actual R pair56072-row CPU/source audit '+verdict+' samefamilyprovisional, publishedmain'+heads[0]+'/doc57; fourlocal+remoteexact. Nativehidden5616/4475 visible5615/4477, best5616/4506 retained. Two inferiorR deltas2568224B removed/noarchive. Large rows/logs canonicalDATA plus exactGitHub; no systemdiskduplicate. GoalACTIVE_UNMET; next actual-final-quality/native-bbs supervision plan, no Nr/Sr result.\n')
print(json.dumps(record),flush=True)
