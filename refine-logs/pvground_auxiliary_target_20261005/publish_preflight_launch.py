"""Publish reviewed auxiliary-target sources and actual preflight launch only."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

import paramiko

root = Path(__file__).resolve().parent
state_path = root.parent / 'pvground_query_supported_geometry_20261005/active_continuation_state.json'
state = json.loads(state_path.read_bytes())
previous = json.loads(Path(state['latest_publication']).read_bytes())
assert previous['section'] == '20.376.68' and not (root / 'source_publication.json').exists()
review = json.loads((root / 'SOURCE_REVIEW.json').read_bytes())
assert review['verdict'] == 'PASS' and not review['blocking_findings']
assert all(hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256'] for item in review['reviewed_files'])
launch = json.loads((root / 'preflight_launch.json').read_bytes())
assert launch['status'] == 'PREFLIGHT_LAUNCHED_NOT_COMPLETED'
repos = [Path(r'C:\Users\gb\.codex_mcln_g0_20260905'), Path(r'C:\Users\gb\.codex_pvground_cs_20261002'),
         Path(r'C:\Users\gb\.codex_mcln_v99_internal_20260928')]
doc = 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
copies = [repo / doc for repo in repos] + [Path(r'C:\Users\gb\Desktop\document') / Path(doc).name]
old = copies[0].read_bytes()
assert all(path.read_bytes() == old for path in copies)
assert hashlib.sha256(old).hexdigest() == previous['handoff_sha256'] and b'## 20.376.69 ' not in old


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args]).decode().strip()


for repo, head in zip(repos, previous['heads']):
    assert git(repo, 'rev-parse', 'HEAD') == head and not git(repo, 'status', '--porcelain')
stamp = datetime.datetime.now().astimezone().isoformat()
section = f'''

## 20.376.69 辅助定位目标坐标定义对照已实现、已审查，实际两步GPU预检已启动（{stamp}）

依据§20.376.68的原生数据核对，下一项先比较辅助几何目标，不同时增加空间参考、渐进头、注意力、质量评分或教师。网络仍为PV-Ground＋原G＋whole-range平铺33位置六面头，当前最优5614／4509的完整10状态作为共同父头；模型参数新增0，仅既有456102参数几何头可训练，父模型、Mask／语义与全零R冻结。

两组都保留原生native＋G＋匹配DFL和权重1的Query支撑额外定位项。control使用原生独立扰动的root框；member_target使用同一已共同增强的原生50000点scan表示中、独立GT框扰动之前的标注成员框。所选辅助框同时定义Box≤0.5资格及额外L1／GIoU／DFL目标；自身Query和融合Mask>0.5资格、排除全部原匹配Query、表达内再batch平均及空集合0均不变。因此这是辅助目标与其几何资格定义的整体对照，不冒充纯loss坐标替换。

额外root框在调用原生_get_target_boxes之前只读捕获，无新增随机调用；原生返回GT、检测对象输入、原Hungarian匹配、原损失和G标签不改。该字段不进入模型forward；评估仍按原生GT、关闭augment／augment_det。所有256候选及唯一last/bbs、同Query框和Mask保持，GT资格只用于训练。新初始化不依赖已删除4506或旧control头。

fresh SOURCE_ONLY审查实际PASS、阻塞0，45个当前文件核验；requested Astra/max，后端未独立见证，归属same-family/provisional。预检于{launch['time_cst']}实际启动，controller为{launch['process'].split()[0]}，screen={launch['screen']}。每组计划两步真实GPU更新，只在内存保存／恢复，不产生权重。第1批将精确核对已关闭CPU64的行ID、点字节、原生扰动root框和成员root框；并检验额外输出梯度只覆盖资格Query、父状态及缓存Mask／bbs保持、有限梯度和严格优化器恢复。

此处只记录实际启动，尚未发布预检终态，也没有本组正式精度。唯一owner观测按启动后300秒首次、未完成时240秒继续，不在多处重复轮询。只有预检通过后才启动正式fit；两组重新加载4509、重置优化器，不使用两步预检状态。每组29778条fit各一次、batch8／累积1／3723更新／尾batch2、seed2027、lr1e-5、WD0.0005、clip0.1；初末6887模块留出和9508正式验证分别记录。几何头父历史7446更新，正式终点累计11169，原G历史另计。

正式训练仍未启动，4509权重和原G／官方PV、V99链继续保护。新结果须同时比较本轮control及4509强起点，并记录严格修复／破坏、宽松及Mask代价；有结果再及时清理已完成非最佳权重。实现、两份spec、审查、实际启动和只读owner见refine-logs/pvground_auxiliary_target_20261005/。
'''
new = old + section.encode('utf-8')
prefix = 'refine-logs/pvground_auxiliary_target_20261005/'
payloads = {prefix + file.relative_to(root).as_posix(): file.read_bytes() for file in root.rglob('*')
            if file.is_file() and file.suffix in ('.py', '.json', '.md')}
payloads[prefix + '.gitattributes'] = b'** -text\n'
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('region-9.autodl.pro', port=33476, username='root',
    password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
project = '/home/gb/new butd/butd_detr-main/MCLN-main'
for name, raw in payloads.items():
    for repo in repos[:2]:
        destination = repo / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    parent, _, basename = name.rpartition('/')
    parts = parent.split('/')
    for index in range(1, len(parts) + 1):
        folder = '/'.join(parts[:index]); upper, _, leaf = folder.rpartition('/')
        if leaf not in sftp.listdir(project + '/' + upper):
            sftp.mkdir(project + '/' + folder)
    with sftp.open(project + '/' + name, 'wb') as stream:
        stream.write(raw)
    with sftp.open(project + '/' + name, 'rb') as stream:
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
reports = [prefix + 'SOURCE_REVIEW.json', prefix + 'SOURCE_REVIEW.md']
for index, repo in enumerate(repos):
    stage = [doc]
    if index < 2:
        with (repo / 'MANIFEST.md').open('a', encoding='utf-8') as stream:
            stream.write('\n- ' + stamp + ' Native/member auxiliary-target comparison sources reviewed and real two-step sanity launched; no formal result or new model parameter.\n')
        stage.append('MANIFEST.md')
    subprocess.check_call(['git', '-C', str(repo), 'add', '-f', '--', *stage], stderr=subprocess.DEVNULL)
    if index < 2:
        subprocess.check_call(['git', '-C', str(repo), '-c', 'core.autocrlf=false', 'add', '-f', '--', *payloads], stderr=subprocess.DEVNULL)
        stage += list(payloads)
    excludes = [':(exclude)' + name for name in reports] if index < 2 else []
    subprocess.check_call(['git', '-C', str(repo), '-c', 'core.whitespace=cr-at-eol', 'diff', '--cached', '--check', '--', *stage, *excludes])
    subprocess.check_call(['git', '-C', str(repo), '-c', 'core.whitespace=cr-at-eol,-blank-at-eof', 'diff', '--cached', '--check', '--', *stage])
    for name, raw in payloads.items() if index < 2 else []:
        assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + name]) == raw
    assert subprocess.check_output(['git', '-C', str(repo), 'show', ':' + doc]) == new.replace(b'\r\n', b'\n')
    subprocess.check_call(['git', '-C', str(repo), 'commit', '--quiet', '-m', 'Prepare controlled auxiliary-target geometry comparison'])
    heads.append(git(repo, 'rev-parse', 'HEAD'))
    assert not git(repo, 'status', '--porcelain')
subprocess.check_call(['git', '-C', str(repos[0]), '-c', 'http.version=HTTP/1.1', 'push', 'origin', 'HEAD:main'])
assert git(repos[0], 'ls-remote', 'origin', 'refs/heads/main').split()[0] == heads[0]
digest = hashlib.sha256(new).hexdigest()
guard = Path(r'C:\Users\gb\.codex\tmp\sync_cs_handoff_remote_20260923.py')
raw = guard.read_bytes()
assert raw.count(previous['handoff_sha256'].encode()) == 1
guard.write_bytes(raw.replace(previous['handoff_sha256'].encode(), digest.encode()))
record = dict(previous)
record.update(time_cst=stamp, section='20.376.69', heads=heads, github_main=heads[0], handoff_bytes=len(new),
    handoff_sha256=digest, four_local_and_remote_equal=True, payload_count=len(payloads),
    status='REVIEWED_AUXILIARY_TARGET_SANITY_LAUNCH_PUBLISHED', new_accuracy_result=False,
    preflight_started_cst=launch['time_cst'], formal_fit_launched=False)
(root / 'source_publication.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
current = json.loads(state_path.read_bytes())
current.update(latest_publication=str(root / 'source_publication.json'), published_heads=heads,
    handoff_section='20.376.69', handoff_sha256=digest, next_action='Observe existing auxiliary-target preflight owner, collect and validate actual completion before formal fit',
    current_goal_turn_classification='PROGRESS_LABEL_CHECK_IMPLEMENTATION_AND_REAL_GPU_SANITY')
state_path.write_text(json.dumps(current, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
with Path(r'C:\Users\gb\memory\2026-10-05.md').open('a', encoding='utf-8') as stream:
    stream.write('\nPVGround ' + stamp + ': Doc69 actual auxiliary-target source PASS45 and two-stepGPU sanity launch published; nativeGT preserved, extra target+qualification compares native vs member. Formal notstarted, best4509 preserved. Main ' + heads[0] + ', ACTIVE_UNMET.\n')
print(json.dumps({key: record[key] for key in ('time_cst', 'section', 'heads', 'handoff_sha256',
    'four_local_and_remote_equal', 'preflight_started_cst', 'formal_fit_launched')}), flush=True)
