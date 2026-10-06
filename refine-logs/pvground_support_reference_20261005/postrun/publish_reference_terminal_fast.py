"""Append the actual closed comparison, audit and retention to all handoffs."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shlex
import paramiko

local = Path(__file__).resolve().parents[1]
workspace = Path('C:/Users/gb')
assert not (local / 'terminal_publication.json').exists()
state_path = local.parent / 'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state = json.loads(state_path.read_bytes())
previous = json.loads(Path(state['latest_publication']).read_bytes())
summary = json.loads((local / 'analysis/SUMMARY.json').read_bytes())
audit = json.loads((local / 'analysis/EXPERIMENT_AUDIT.json').read_bytes())
review_call = json.loads((local / 'analysis/TERMINAL_REVIEW_CALL.json').read_bytes())
retention = json.loads((local / 'weight_retention.json').read_bytes())
resources = json.loads((local / 'CLOSED_RESOURCES.json').read_bytes())
wait = json.loads((local / 'fit_wait.json').read_bytes())
assert summary['status'] == 'ACTUAL_CLOSED_REFERENCE_ROWS_ANALYZED' and summary['fit_order_exact']
assert review_call['result_received'] and audit['verdict'] in ('PASS', 'WARN') and not audit['blocking_findings']
assert wait['observer_closed'] and wait['terminal']['exitcode'] == 0 and not wait['terminal']['controller_alive']
assert retention['status'] == 'CLOSED_NONBEST_WEIGHTS_REMOVED' and resources['deleted_weights_absent']
assert not resources['gpu_compute_processes']
best = retention['retained_best']
assert best['system'] == summary['metric_best_candidate']['system']
assert best['hits'] == [summary['metric_best_candidate']['rec_hits25'], summary['metric_best_candidate']['rec_hits50']]
assert len(retention['deleted']) == 2 and retention['local_weight_archive_created'] is False
assert previous['section'] == '20.376.78'
repos = [workspace / '.codex_mcln_g0_20260905', workspace / '.codex_pvground_cs_20261002',
         workspace / '.codex_mcln_v99_internal_20260928']
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies = [repo / doc for repo in repos] + [workspace / 'Desktop/document' / Path(doc).name]
old = copies[0].read_bytes()
assert hashlib.sha256(old).hexdigest() == previous['handoff_sha256']
assert all(path.read_bytes() == old for path in copies) and b'## 20.376.79 ' not in old
for repo, head in zip(repos, previous['heads']):
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip() == head
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
stamp = datetime.datetime.now().astimezone().isoformat()
verdict=audit['verdict']
finished=wait['terminal']['status']['finished_cst']
effect=summary['support_vs_control']['0.5']
table='\n'.join(f"| {row['system']} | {100*row['rec_hits25']/9508:.4f}%（{row['rec_hits25']}） | {100*row['rec_hits50']/9508:.4f}%（{row['rec_hits50']}） |" for row in summary['table'])
details=[]
for arm,values in summary['systems'].items():
    ref=values['internal_reference']['0.5'];final=values['internal_final']['0.5'];delta=values['parent_delta']['0.5']
    details.append(f"- {arm}：相对4511起点严格修复{delta['repairs']}、破坏{delta['damages']}，净{delta['net']:+d}；同一已选Query粗框→参考框{ref['before_hits']}→{ref['after_hits']}，修复{ref['repairs']}、破坏{ref['damages']}；参考框→最终框{final['before_hits']}→{final['after_hits']}，修复{final['repairs']}、破坏{final['damages']}。内部阶段变化不算相对独立baseline的增量。额外候选学习职责累计{values['extra_roles']}次，额外DFL目标超节点范围累计{values['extra_outside_faces']}面；额外定位loss前／后100步均值{values['extra_loss_first100']:.8f}／{values['extra_loss_last100']:.8f}，参考定位loss对应{values['reference_loss_first100']:.8f}／{values['reference_loss_last100']:.8f}。已选Mask合格但框差{values['selected_mask_good_box_bad']}条；严格Full256几何上界{values['full256_strict_scalar_oracle']}，这些Mask与上界仅为已存标量核算。")
prefix='refine-logs/pvground_support_reference_20261005/'
intake=json.loads((local/'complete/INTAKE.json').read_bytes())
section=f"""

## 20.376.79 固定参考与自身／融合支撑参考正式终态、最佳保留（{stamp}）

承接§77—78，本轮实际于{finished}闭合，四个train／formal子任务与原控制器均退出0。原本地观察者40310在后续会话中句柄缺失且本地进程不存在，没有生成终态回执；{wait['time_cst']}的一次只读服务器核查确认原控制器已闭合、退出码与结果文件，并生成实际闭合回执。未重新启动训练，原观察者的退出码仍未知。

| 完整9508条ScanRefer开发验证，原生last/bbs | Acc@0.25 | Acc@0.50 |
|---|---:|---:|
{table}

两组从同一4511几何头与官方PV／原G重建，fresh optimizer，每组29778条fit各一次、3723更新、B8／有效8、累积1、seed2027、LR1e-5、WD5e-4、clip0.1；父模型、Mask、语义与全零R冻结。旧10项几何状态累计11169→14892更新；支撑参考新增3078参数只学习本轮3723更新。控制456102参数／10状态，支撑参考459180参数／12状态。本轮同时改变支撑参考、参数量及参考L1／GIoU监督，不是纯来源或单loss消融。保留全部256候选、唯一原生bbs及同Query框／Mask；没有教师、双源排名或推理GT资格。

支撑参考相对同预算控制严格修复{effect['repairs']}、破坏{effect['damages']}，净{effect['net']:+d}；已选Query变化{summary['support_vs_control']['selected_query_changes']}。额外定位资格仍由候选自身Query与融合Mask的训练GT支撑确认，排除所有原匹配Query，按表达内及实际batch平均，几何不足判断与目标采用原native GT。

{chr(10).join(details)}

两组训练顺序及29778唯一输入核对一致；每组初始／终点6887是预训练见过场景的模块留出，与9508正式开发验证分开。闭合收集{len(intake['files'])}份原始文本／源码、{sum(item['bytes'] for item in intake['files'].values())}字节，没有下载权重或重跑模型。CPU独立重算已选粗框／参考框／最终框对真实GT的严格阈值；Mask和Full256仅核对存储标量，未重放原始点级Mask或全部候选框。fresh experiment-audit结论{verdict}，无阻断项，实际调用和报告见{prefix}analysis/；同系列、暂定接受，后端身份未证实。单seed结果不证明统计稳定、三模块有效或Nr／Sr泛化。

按预定Acc@0.50规则保留{best['system']}：{best['hits'][0]}／{best['hits'][1]}，路径{best['path']}，SHA256 {best['sha256']}；与父4511严格指标持平时保留父权重。实际删除两份闭合非最佳几何头，释放{retention['released_bytes']}字节，不归档负结果权重；保留完整10或12项几何状态及优化器，重建需要官方PV、原G及相应源码／配置。实际CPU严格头部重载已通过，非全模型推理重放。原PV/G和V99历史链保留。清理后数据盘余量{resources['data_free_bytes']}字节、系统盘{resources['system_free_bytes']}字节，GPU计算进程为空；远端发布complete复用数据盘闭合原文件。

当前保留模型距离4754严格命中尚差{max(0,4754-best['hits'][1])}条，距离V99的4797尚差{max(0,4797-best['hits'][1])}条。开发双阈值通过标志为{summary['scanrefer_target_pass']}，完整三数据集目标仍ACTIVE_UNMET，没有新Nr3D／Sr3D结果。后续依据实际阶段修复／破坏及超范围目标决定是否开展渐进精修；不把本轮结果解释成仅需扩大参考头、普通attention或再次质量回读。
"""
assert chr(65533) not in section
new = old + section.encode('utf-8')
assert new.startswith(old) and new.count(b'## 20.376.79 ') == 1
names=['fit_wait.json','weight_retention.json','CLOSED_RESOURCES.json',
    'inspect_existing_fit_authorized.py','COLLECTION_PREFETCH_CHANGE.json','COLLECTION_STREAM_CHANGE.json','COLLECTION_ROW_FILE_SIZES.json',
    'postrun/collect_formal_prefetch_authorized.py','postrun/prepare_prefetch_collector.py',
    'postrun/collect_formal_stream_authorized.py',
    'postrun/check_closed_resources.py','postrun/prepare_reference_terminal_publication.py',
    'postrun/publish_reference_terminal.py','TERMINAL_PUBLICATION_PREPARATION.json',
    'postrun/publish_reference_terminal_fast.py','postrun/prepare_fast_terminal_publisher.py',
    'PUBLICATION_TRANSFER_OPTIMIZATION.json']
for directory in ('complete', 'analysis'):
    names.extend(str(path.relative_to(local)).replace('\\', '/') for path in sorted((local / directory).rglob('*')) if path.is_file())
assert all(not name.endswith(('.pth', '.pt')) and '.aris' not in name for name in names)
payloads = {prefix + name: (local / name).read_bytes() for name in names}
payloads[prefix + '.gitattributes'] = b'** -text\n'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
project = '/home/gb/new butd/butd_detr-main/MCLN-main'
with sftp.open(project + '/' + doc, 'rb') as stream:
    assert stream.read() == old
evidence = project + '/' + prefix.rstrip('/')
assert not any(entry.filename == 'complete' for entry in sftp.listdir_attr(evidence))
sftp.symlink('/root/autodl-tmp/pvground_support_reference_20261005', evidence + '/complete')
assert sftp.readlink(evidence + '/complete') == '/root/autodl-tmp/pvground_support_reference_20261005'
directories = set()
for name in payloads:
    parts = Path(name).parts[:-1]
    directories.update('/'.join(parts[:index]) for index in range(1, len(parts) + 1))
for relative in sorted(directories, key=lambda name: (name.count('/'), name)):
    parent, _, basename = relative.rpartition('/')
    if not any(entry.filename == basename for entry in sftp.listdir_attr(project + ('/' + parent if parent else ''))):
        sftp.mkdir(project + '/' + relative)
existing={name:hashlib.sha256(raw).hexdigest() for name,raw in payloads.items()
    if name.startswith(prefix+'complete/') and name!=prefix+'complete/INTAKE.json'}
probe="""
import hashlib,json,sys
from pathlib import Path
project=Path(sys.argv[1]);expected=json.loads(sys.argv[2])
for name,digest in expected.items():
    assert hashlib.sha256((project/name).read_bytes()).hexdigest()==digest,name
print(json.dumps(dict(verified_existing_closed_texts=len(expected))))
"""
runtime=json.loads((local/'control_spec.json').read_bytes())['runtime']
_,stdout,stderr=client.exec_command(shlex.join([runtime+'/venv/bin/python','-B','-c',probe,project,json.dumps(existing)]),timeout=120)
verified=json.loads(stdout.read())
assert stdout.channel.recv_exit_status()==0,stderr.read().decode()
assert verified['verified_existing_closed_texts']==len(existing)
for name, raw in payloads.items():
    for repo in repos[:2]:
        destination = repo / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    remote = project + '/' + name
    if not name.startswith(prefix + 'complete/') or name == prefix + 'complete/INTAKE.json':
        with sftp.open(remote, 'wb') as stream:
            stream.write(raw)
    if name not in existing:
        with sftp.open(remote, 'rb') as stream:
            assert stream.read() == raw
for path in copies:
    path.write_bytes(new)
with sftp.open(project + '/' + doc, 'wb') as stream:
    stream.write(new)
with sftp.open(project + '/' + doc, 'rb') as stream:
    assert stream.read() == new
sftp.close()
client.close()
heads = []
for index, repo in enumerate(repos):
    stage = [doc]
    if index < 2:
        with (repo / 'MANIFEST.md').open('a', encoding='utf-8') as stream:
            stream.write('\n- ' + stamp + ' Actual closed support-reference comparison published; metric-best ' + best['system'] + ' ' + str(best['hits']) + ', fresh audit ' + verdict + ' same-family/provisional. Two nonbest geometry heads removed, no archive.\n')
        stage += ['MANIFEST.md', *payloads]
    changed = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all']).decode()
    assert {line[3:] for line in changed.splitlines()}.issubset(set(stage))
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage], stderr=subprocess.DEVNULL)
    reports = [name for name in payloads if Path(name).name in ('EXPERIMENT_AUDIT.json', 'EXPERIMENT_AUDIT.md')] if index < 2 else []
    subprocess.check_call(['git', '-C', str(repo), '-c', 'core.whitespace=cr-at-eol', 'diff', '--cached', '--check', '--', *stage, *[':(exclude)' + name for name in reports]])
    subprocess.check_call(['git', '-C', str(repo), '-c', 'core.whitespace=cr-at-eol,-blank-at-eof', 'diff', '--cached', '--check', '--', *stage])
    if index < 2:
        for name, raw in payloads.items():
            assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + name]) == raw
    git_doc = subprocess.check_output(['git', '-C', str(repo), 'show', ':' + doc])
    old_git_doc = subprocess.check_output(['git', '-C', str(repo), 'show', previous['heads'][index] + ':' + doc])
    assert git_doc.startswith(old_git_doc) and git_doc.count(b'## 20.376.79 ') == 1
    subprocess.check_call(['git', '-C', str(repo), 'commit', '--quiet', '-m', 'Record closed support-reference comparison and best retention'])
    heads.append(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip())
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'])
subprocess.check_call(['git', '-C', str(repos[0]), '-c', 'http.version=HTTP/1.1', 'push', 'origin', 'HEAD:main'])
assert subprocess.check_output(['git', '-C', str(repos[0]), 'ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == heads[0]
digest = hashlib.sha256(new).hexdigest()
guard = workspace / '.codex/tmp/sync_cs_handoff_remote_20260923.py'
raw = guard.read_bytes()
assert raw.count(previous['handoff_sha256'].encode()) == 1
guard.write_bytes(raw.replace(previous['handoff_sha256'].encode(), digest.encode()))
assert all(path.read_bytes() == new for path in copies)
record = dict(time_cst=datetime.datetime.now().astimezone().isoformat(), section='20.376.79', heads=heads,
              github_main=heads[0], handoff_bytes=len(new), handoff_sha256=digest, four_local_and_remote_equal=True,
              payload_count=len(payloads), new_accuracy_result=True, table=summary['table'],
              retained_best=best, deleted_nonbest_bytes=retention['released_bytes'], negative_weight_archived=False,
              integrity_verdict=verdict, review_independence='same-family', acceptance_status='provisional',
              scanrefer_target_pass=summary['scanrefer_target_pass'], full_goal_status='ACTIVE_UNMET',
              remote_complete_reuses_closed_data_disk_originals=True,
              existing_closed_payloads_verified_by_remote_SHA256=len(existing))
(local / 'terminal_publication.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
state.update(time_cst=record['time_cst'],status='SUPPORT_REFERENCE_TERMINAL_PUBLISHED',
    latest_publication=str(local/'terminal_publication.json'),owned_gpu_job_active=False,
    active_reviewer=None,protected_best_hits=best['hits'],strict_target_gap=max(0,4754-best['hits'][1]),
    published_heads=heads,handoff_section='20.376.79',handoff_sha256=digest,
    support_reference_fit_observer_closed=True,support_reference_fit_observer_session_id=None,
    support_reference_full_pair_complete=True,support_reference_actual_fit_finished_cst=finished,
    support_reference_terminal_integrity_review_pending=False,support_reference_terminal_integrity_verdict=verdict,
    support_reference_postrun_tools_executed=True,
    support_reference_formal_hits_by_arm={row['system']:[row['rec_hits25'],row['rec_hits50']] for row in summary['table']},
    current_goal_turn_classification='PROGRESS_ACTUAL_TERMINAL_AUDIT_RETENTION_PUBLISHED')
state_path.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
with (workspace/'memory/2026-10-06.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPVGround '+record['time_cst']+': Support-reference actual closed pair '+str(summary['table'])+'. Fresh terminal audit '+verdict+' same-family/provisional; retained '+str(best['hits'])+', two closed nonbest heads removed without archive. Doc79 fourlocal+remote exact, main '+heads[0]+'. Original observer40310 missing; actual one-shot remote closure verified, no training restart. Geometry old states total14892, new reference3078params thissegment3723updates. GoalACTIVE_UNMET; no new Nr/Sr results.\n')

print(json.dumps(record), flush=True)
