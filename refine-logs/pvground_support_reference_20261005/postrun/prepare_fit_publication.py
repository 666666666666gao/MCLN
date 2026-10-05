"""Prepare append-only actual sanity/fit-launch publication; no SSH calls."""
import ast
from pathlib import Path

root=Path(__file__).resolve().parents[1]
destination=root/'postrun/publish_fit_launch.py'
assert not destination.exists()
source=(root/'publish_preflight_launch.py').read_text(encoding='utf-8')


def change(text,before,after):
    assert text.count(before)==1,before
    return text.replace(before,after)


source=change(source,'local=Path(__file__).resolve().parent','local=Path(__file__).resolve().parents[1]')
source=change(source,"old_publication=json.loads((local.parent/'pvground_auxiliary_target_20261005/terminal_publication.json').read_bytes())",
    "old_publication=json.loads((local/'preflight_launch_publication.json').read_bytes())")
source=source.replace("local/'preflight_launch.json'","local/'fit_launch.json'")
source=source.replace("'PREFLIGHT_LAUNCHED_NOT_COMPLETED'","'TWO_ARM_FIT_LAUNCHED_NOT_COMPLETED'")
source=source.replace("local/'preflight_launch_publication.json'","local/'fit_launch_publication.json'")
# The previous publication remains Doc76, not this new one's output.
source=change(source,"old_publication=json.loads((local/'fit_launch_publication.json').read_bytes())",
    "old_publication=json.loads((local/'preflight_launch_publication.json').read_bytes())")
start=source.index("section=f'''\n")
end=source.index("assert chr(65533) not in section",start)
section='''proofs={arm:json.loads((local/'preflight_complete'/arm/'preflight.json').read_bytes()) for arm in ('control','support_reference')}
wait=json.loads((local/'preflight_wait.json').read_bytes())
assert wait['observer_closed'] and not wait['controller_alive'] and wait['exitcode']==0
for arm,proof in proofs.items():
    assert proof['status']=='pass' and proof['optimizer_steps']==2 and proof['weight_files_created']==0
    assert proof['all_parent_and_R_states_exact'] and proof['optimizer_exact_check']['all_keys_moments_steps_and_groups_exact']
assert proofs['support_reference']['witnesses'][0]['zero_reference_cached_old_head_exact']
assert all(value['reference_direct_geometry_gradient']>0 for value in proofs['support_reference']['witnesses'])
section=f\'\'\'

## 20.376.77 固定／自身支撑参考：两组真实预检通过，正式对照已启动（{stamp}）

§76所述参考结构保持。两组预检实际于 {wait['status']['finished_cst']} 闭合，控制器退出0，耗时 {wait['status']['elapsed_seconds']:.2f} 秒；各2次真实更新，不写权重。缓存同一上游时，学习参考零初始化与旧头输出一致；参考直接定位梯度两步均非零，原额外定位目标仍只对其资格输出产生直接梯度。头更新后的缓存原生bbs和Mask保持，PV/G及全零R状态保持，CPU保存／严格重载和优化器全部键、矩、步及组检查通过。这些是工程预检，不是新精度结果，也不证明跨进程CUDA前向逐位一致。

实际allocator分配峰值：control {proofs['control']['peak_allocated_bytes']} 字节，support_reference {proofs['support_reference']['peak_allocated_bytes']} 字节；仅代表冻结父模型的预检batch，不外推到全模型联合训练或任意场景。当前batch及学习率不变。

正式控制器实际于 {launch['time_cst']} 启动，进程 {launch['process']}，screen {launch['screen']}；采用同一受保护4511权重、fresh optimizer，预检状态不承接。每组29778条fit各一次／3723更新，B8、累积1、seed2027、LR1e-5、WD5e-4、clip0.1。父PV/G、Mask、语义及全零R冻结；控制456102参数／10状态，学习参考459180参数／12状态。旧十项几何状态累计11169→14892更新；新增3078参考参数只训练本轮3723步，原G历史另算。

保留各自初始／终点6887 seen-scene模块留出，之后分别9508条原生开发验证。新增参考定位目标的预算及梯度范数单独记录；结构、参数量与监督预算同时变化，不把它称作纯输入或单项loss消融。唯一bbs、全256、同Query Box／Mask、原GT评价保持，没有教师或第二套排名。

正式预计约 {launch['estimated_seconds']} 秒；首次按 {launch['first_check_seconds']} 秒、控制组终点附近安排，随后240秒轮询。只有一个fit观察者，不因未到估计时刻或观察超时重启GPU训练。启动前数据盘余量 {launch['resources']['data_free_bytes']} 字节、系统盘 {launch['resources']['system_free_bytes']} 字节，所需保存及日志储备 {launch['resources']['required_reserve_bytes']} 字节，GPU空闲及预检闭合已实际检查。

实际预检原始记录见 {prefix}preflight_complete/INTAKE.json及两组preflight.json；正式启动见fit_launch.json／fit_resource_check.json。结果仍待正式9508评估；当前best仍5616／4511、Acc@0.50约47.4443%，到50%还差243净命中。终态经独立核验后才晋级与删除闭合非best，不归档负权重；原PV/G/V99保留。未有Nr3D／Sr3D新结果，总目标ACTIVE_UNMET。
\'\'\'
'''
source=source[:start]+section+source[end:]
source=source.replace("b'## 20.376.76 '","b'## 20.376.77 '")
start=source.index("names=[path.name for path in local.glob('*.py')]")
end=source.index("payloads[prefix+'.gitattributes']",start)
source=source[:start]+'''names=['fit_launch.json','fit_resource_check.json','preflight_wait.json']
names += [str(path.relative_to(local)).replace('\\\\','/') for path in sorted((local/'preflight_complete').rglob('*')) if path.is_file()]
names += ['postrun/publish_fit_launch.py','postrun/prepare_fit_publication.py','postrun/analyze_reference_formal.py']
payloads={prefix+name:(local/name).read_bytes() for name in names}
'''+source[end:]
source=change(source,"assert prefix.rstrip('/').split('/')[-1] not in sftp.listdir(project+'/refine-logs')\nsftp.mkdir(remote_directory)",
    "assert prefix.rstrip('/').split('/')[-1] in sftp.listdir(project+'/refine-logs')\n"
    "directories={str(Path(name).parent).replace('\\\\','/') for name in names if '/' in name}\n"
    "for relative in sorted(directories,key=lambda value:(value.count('/'),value)):\n"
    "    parts=relative.split('/')\n"
    "    for count in range(1,len(parts)+1):\n"
    "        directory='/'.join(parts[:count]);parent,_,basename=directory.rpartition('/')\n"
    "        if basename not in sftp.listdir(remote_directory+('/'+parent if parent else '')):\n"
    "            sftp.mkdir(remote_directory+'/'+directory)")
source=change(source,"with sftp.open(project+'/'+name,'wx') as stream:","with sftp.open(project+'/'+name,'wb') as stream:")
source=source.replace('Own/fused support reference SOURCE review complete; actual two-step GPU sanity launched, no accuracy result.',
    'Own/fused support reference two-step GPU sanity passed; same-start formal pair launched, no new accuracy result.')
source=source.replace('Prepare own-support reference and launch real GPU sanity','Record passed reference sanity and launch paired formal comparison')
source=source.replace("'section':'20.376.76'","'section':'20.376.77'")
source=source.replace("'preflight_launch':launch","'fit_launch':launch")
source=source.replace("status='SUPPORT_REFERENCE_PREFLIGHT_LAUNCHED_PUBLISHED'","status='SUPPORT_REFERENCE_FORMAL_FIT_ACTIVE_PUBLISHED'")
source=source.replace('support_reference_preflight_complete=False','support_reference_preflight_complete=True')
source=change(source,"next_action='Wait for the sole estimated preflight observer. Read actual primary logs if failed; collect successful closure, then launch the reviewed same-budget formal pair with fresh optimizer.'",
    "next_action='Keep sole fit observer40310 until estimated closure; first check2026-10-06 01:26:31 CST, later240s. Do not restart controller653283 or create another observer. Collect both actual9508 formal rows only after closure, fresh terminal audit, then strict-best retention and publication.'")
source=change(source,"support_reference_deployed=True,support_reference_preflight_complete=True,",
    "support_reference_deployed=True,support_reference_preflight_complete=True,\n"
    "    support_reference_preflight_observer_closed=True,support_reference_preflight_observer_session_id=None,\n"
    "    support_reference_fit_observer_session_id=40310,support_reference_fit_observer_closed=False,")
source=source.replace('Doc76 own/fused support reference SOURCE review and actual two-step preflight launch; no accuracy result.',
    'Doc77 own/fused support reference actual two-step sanity passed and formal pair launched; no new accuracy result.')
ast.parse(source)
destination.write_text(source,encoding='utf-8')
print('Prepared Doc77 publisher; no remote call or publication executed.')
