"""Derive the routine append-only publication from the completed publisher."""
import ast
from pathlib import Path


root = Path(__file__).resolve().parent
source = (root / 'publish_fit_launch.py').read_text(encoding='utf-8')
before = source.index("section = f'''")
after = source.index("new = old + section.encode('utf-8')", before)
section = """relocation = json.loads((root / 'PUBLICATION_STORAGE_RELOCATION.json').read_bytes())
assert relocation['status'] == 'PUBLISHED_EVIDENCE_RELOCATED_LOGICAL_PATH_PRESERVED'
assert relocation['source_and_destination_file_sizes_and_sha256_exact']
assert relocation['records_deleted'] == relocation['weights_deleted'] == 0
section = f'''

## 20.376.71 实验记录迁到数据盘，终态收集与非最佳权重清理工具已准备（{stamp}）

启动时系统盘仅约43MiB可用，存储核对确认refine-logs实际占用约420MiB且位于系统盘。{relocation['time_cst']}已将整个静态发布证据目录迁至{relocation['data_location']}，原路径{relocation['logical_path']}通过目录链接保持访问。迁移前后4355项文件／链接及428491229字节实际文件内容均核对一致，6个已有绝对链接保留；没有删除实验记录或权重，也没有训练／优化器重放。首次迁移检查因发现已有链接而在任何移动前停止；按真实链接状态修正后完成上述迁移。

此次释放系统盘439902208字节；实际可用空间由44732416增至484634624字节，数据盘余量1787027456字节。这里是存储操作时的测量，不推断之后余量。运行中的正式训练目录、PV／原G／4509权重及模型源码未迁移。后续SFTP发布仍使用原refine-logs路径，证据写入数据盘。

已复用上轮只读工具准备本轮闭合结果收集与CPU重算：collect_formal_authorized.py、geometry_result_metrics.py、analyze_closed_formal.py。新重算比较control／member_target与4509父模型，核对两组权重1、实际目标模式、29778条相同行序和3723更新，终点累计11169；同时记录9508实际命中、修复／破坏、Mask及候选覆盖，不下载权重或重放模型。工具尚未执行，没有新增精度结果。

闭合后的retain_metric_best.py及授权入口已准备，尚未审查或执行。必须待两组完整评估、CPU核对、终态审查和清理源码审查通过，确认完整10状态及模型／优化器恢复回执后，才按Acc@0.50优先的实际指标保留最好、删除本轮已结束非最佳头；不碰原G／官方PV／V99，不新建负结果权重归档。当前仍保留5614／4509强起点。

正式fit及唯一观察器沿用§20.376.70实际启动记录：controller630343、native观察器38531、18:54:13首查、之后240秒间隔，预计21:44:13附近闭合。存储操作没有新增GPU进程查询或提前训练轮询，也没有更改训练配置。本次没有formal终态或新REC结果，目标继续ACTIVE_UNMET。存储回执与上述源码位于refine-logs/pvground_auxiliary_target_20261005/。
'''
"""
source = source[:before] + section + source[after:]
changes = {
    "previous = json.loads((root / 'source_publication.json').read_bytes())":
        "previous = json.loads((root / 'fit_launch_publication.json').read_bytes())",
    "assert previous['section'] == '20.376.69'": "assert previous['section'] == '20.376.70'",
    "assert not (root / 'fit_launch_publication.json').exists()":
        "assert not (root / 'storage_tools_publication.json').exists()",
    "assert b'## 20.376.70 ' not in old": "assert b'## 20.376.71 ' not in old",
    "Auxiliary-target two-step sanity closed PASS; formal native/member pair launched from4509, no new accuracy yet.":
        "Published evidence moved to data disk with exact contents and logical paths; closed-result and retention tools prepared, not executed.",
    "Record closed auxiliary-target sanity and actual formal fit launch":
        "Preserve published evidence on data disk and prepare closed-result tools",
    "record.update(time_cst=stamp, section='20.376.70'":
        "record.update(time_cst=stamp, section='20.376.71'",
    "status='CLOSED_SANITY_AND_REAL_FORMAL_LAUNCH_PUBLISHED'":
        "status='STORAGE_RELOCATION_AND_CLOSED_TOOLS_PREPARATION_PUBLISHED'",
    "(root / 'fit_launch_publication.json').write_text":
        "(root / 'storage_tools_publication.json').write_text",
    "latest_publication=str(root / 'fit_launch_publication.json')":
        "latest_publication=str(root / 'storage_tools_publication.json')",
    "handoff_section='20.376.70'": "handoff_section='20.376.71'",
    "current_goal_turn_classification='PROGRESS_REAL_AUXILIARY_TARGET_FORMAL_FIT'":
        "current_goal_turn_classification='PROGRESS_STORAGE_RELOCATION_AND_CLOSED_TOOLS'",
    "stream.write('\\nPVGround ' + stamp + ': Auxiliary native/member target preflight closed PASS2steps/arm, no saved weights; formal pair actually launched17:10:53 from4509, each3723updates, nativeGT protocol untouched. Sole observer38531, first18:54:13 then240s, pairestimate21:44:13. Doc70 and raw evidence published. Best4509 retained; ACTIVE_UNMET. Main ' + heads[0] + '.\\n')":
        "stream.write('\\nPVGround ' + stamp + ': Published evidence relocated from system disk to /root/autodl-tmp/mcln_published_evidence_20261005 with original refine-logs symlink, 4355entries/428491229bytes exact and6absolute links preserved. System free44.7MB to484.6MB, no weights or records deleted; first attempt stopped before moving on existing-link assertion. Closed result/retention tools prepared but notexecuted; cleanup still needs actual terminal and source review. Doc71 Main ' + heads[0] + ', ACTIVE_UNMET; same live observer38531 waits18:54.\\n')",
}
for old, new in changes.items():
    assert source.count(old) == 1, old
    source = source.replace(old, new)
marker = "(root / 'storage_tools_publication.json').write_text"
assert source.count(marker) == 1
source = source.replace(marker, "record.update(publication_storage_relocation=relocation,closed_result_tools_executed=False,retention_tools_executed=False)\n" + marker)
assert not (root / 'publish_storage_tools.py').exists()
ast.parse(source)
(root / 'publish_storage_tools.py').write_text(source, encoding='utf-8')
print('Prepared append-only storage/tools publication; no publication executed.')
