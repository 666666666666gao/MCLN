import datetime
import hashlib
import json
import os
from pathlib import Path

import paramiko

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
prep = repo / 'refine-logs/native_range_preparation_20260907_v2'
master = repo / 'docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md'
desktop = Path('C:/Users/gb/Desktop/document') / master.name
old = master.read_bytes()
assert hashlib.sha256(old).hexdigest() == '6916c1d1c6de01887908e439db8108c1a3b8e5f2f417dc9fb8f27bb4da89198c'
assert desktop.read_bytes() == old
receipt_raw = (prep / 'receipt.json').read_bytes()
receipt = json.loads(receipt_raw)
assert receipt['status'] == 'pass' and receipt['reader_variant'] == 'extent'
assert (prep / 'controller.exit').read_text().strip() == '0'
assert '62 passed' in (prep / 'run.log').read_text(encoding='utf-8')
expected = json.loads((prep / 'expected.json').read_bytes())
for name, digest in expected['overlay_files'].items():
    assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == digest, name
assert receipt['gpu_forwards'] == receipt['native_model_optimizer_updates'] == receipt['checkpoint_writes'] == 0
state = json.loads((prep / 'collection.json').read_bytes())
observation = state['scanrefer_latest_queue_observation']
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
when = now.strftime('%Y-%m-%d %H:%M CST')
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=33476, username='root', password=os.environ['MCLN_SSH_PASSWORD'], timeout=30)
sftp = client.open_sftp()
runtime = '/home/gb/new butd/butd_detr-main/MCLN-main/'
with sftp.open(runtime + 'docs/' + master.name, 'rb') as stream:
    stream.prefetch(file_size=len(old))
    assert stream.read() == old
_, output, error = client.exec_command('ps -p 47112,47242 -o pid,stat,etime,args', timeout=30)
processes = output.read().decode()
assert output.channel.recv_exit_status() == 0, error.read().decode()
assert '47112' in processes and '47242' in processes
addition = '\n\n### 20.105 原生范围读取接入准备完成；ScanRefer原配对继续（' + when + '）\n\n'
addition += ('**本轮有实质准备进展，尚无新性能结果，目标保持active。** 原screen47112及接续screen47242已在本次写入时重新确认存活，'
    '没有重复启动ScanRefer。最近一条远端240秒观察为`' + json.dumps(observation, ensure_ascii=False, sort_keys=True) + '`。'
    '零更新基线与更新训练、模块holdout与正式9508评估继续分开报告；现有正式最好没有替换。\n\n'
    '**新增原生入口。** `--use_candidate_local_visual --candidate_local_visual_variant extent`显式构造与当前ScanRefer候选相同的'
    '`CandidateRangeVisual(extent)`；可选center作为对应控制，默认local保留已有局部读取实现。原有独立reader优化器分组继续使用，'
    '无额外优化器分支或新损失。`save_checkpoint`原有config字段记录该参数；恢复含reader权重的checkpoint时核对其计算类型。'
    '这是必要的实际接口约束：三种reader有完全相同的10个参数名称/形状，形状检查不能区分三种计算。'
    '既存local checkpoint没有该新增字段时保持其已知local语义；无reader的旧完整主干可用于新分支的model-only初始化。'
    '源码更新发生在独立原生快照和项目仓库，运行中的ScanRefer冻结源码及manifest没有改动。\n\n'
    '**原服务器CPU检查通过。** 62项检查覆盖读取器的有效点/空区域、已有模型加载与优化器分组、同形状异类型恢复拒绝以及正确类型的优化器/调度器恢复。'
    '另外实际构造Nr3D和Sr3D的MCLN，均加载已锁定Nr3D权重：1144个原有state tensor逐项完全一致，新增10个reader tensor保持零输出初值，'
    '五个优化器分组无重复/缺失、optimizer/scheduler保持全新。两个协议均有1154个state tensor，reader145008参数。'
    '使用原Py3.7/Torch1.10.2环境、CUDA_VISIBLE_DEVICES为空，真实GPU前向=0，真实数据采样=0，原生模型更新=0，持久checkpoint写入=0。'
    'Sr3D历史受保护权重仍未恢复；Nr权重在Sr配置上的可加载性不是Sr性能或直接迁移结果。\n\n'
    '**首轮CPU测试的失败口径。** preparation_v1的62项通过、1项失败；失败是新增toy模型只有2个backbone tensor，'
    '不满足既有model-only加载器至少90个backbone tensor的真实约束。保留该失败日志，移除不代表完整主干的重复toy正例，'
    '由上述实际1144-tensor Nr/Sr模型加载检查承担正例验证；生产加载约束未放宽。v2才是本次通过回执。\n\n'
    '**数据与后续接续。** 本次重新核对正确mesh目录的1201train/312val文件SHA以及其他数据项inode；'
    '没有重报旧64个输入样本为本次新采样。原始32条预检选行与标注凭证保留其旧原生source SHA，'
    '新GPU预检显式绑定新原生source与标注来源；涉及输入的数据集代码沿用原文件。'
    '`run_native_candidate_range_preflight.py`和`launch_native_candidate_range_preflight.py`已准备，后者只有在当前固定范围正式评估及独立审计完整通过后，'
    '才允许用extent在Nr/Sr各16条真实训练输入做2次临时更新、验证原生过滤/选择与梯度。'
    '该启动器本轮没有执行，GPU预检和Nr/Sr正式训练均未启动；当前远端队列仍只负责原定ScanRefer终态与正式审计，未在运行中替换。\n\n')
addition += '原生准备回执SHA`' + hashlib.sha256(receipt_raw).hexdigest() + '`；原生618文件source manifest SHA`' + receipt['source_manifest_sha256'] + '`。'
addition += '目录`/root/autodl-tmp/mcln_native_range_preparation_20260907_v2`；最近磁盘剩余' + str(state['disk']['free']) + ' bytes（' + format(state['disk']['free'] / 1024**3, '.3f') + 'GiB）。'
addition += '两次CPU准备均无训练权重；之前累计清理约9.529GiB未计作本轮新增释放。现有空间足够本轮两个固定终点，未删除受保护或正在使用的权重。\n'
new = old + addition.encode()
tracker = repo / 'refine-logs/EXPERIMENT_TRACKER.md'
lines = tracker.read_text(encoding='utf-8').splitlines()
lines[2] = 'Updated: ' + when + '. §20.105:native extent CPU62PASS and actual pretrained Nr/Sr loading verified;no native GPU/preflight/training. Original Scan pair47112/queue47242 remain live.'
lines.insert(6, '| Native extent integration preparation | CPU62PASS;actual Nr/Sr model/loading/optimizer PASS | 1144 pretrained tensors equal +10 new;0 GPU forwards/updates/weights;Scan promotion still required |')
tracker_raw = ('\n'.join(lines) + '\n').encode()
code_names = ['main_utils.py', 'train_dist_mod.py', 'models/candidate_local_visual_training.py',
              'models/candidate_range_visual.py', 'tests/test_candidate_local_visual_training.py',
              'scripts/run_native_candidate_range_preflight.py', 'scripts/launch_native_candidate_range_preflight.py']
for name in code_names:
    raw = (repo / name).read_bytes()
    with sftp.open(runtime + name, 'wb') as stream:
        stream.set_pipelined(True)
        stream.write(raw)
    with sftp.open(runtime + name, 'rb') as stream:
        stream.prefetch(file_size=len(raw))
        assert stream.read() == raw, name
for name, raw in [('docs/' + master.name, new), ('refine-logs/EXPERIMENT_TRACKER.md', tracker_raw)]:
    with sftp.open(runtime + name, 'wb') as stream:
        stream.set_pipelined(True)
        stream.write(raw)
    with sftp.open(runtime + name, 'rb') as stream:
        stream.prefetch(file_size=len(raw))
        assert stream.read() == raw
master.write_bytes(new)
desktop.write_bytes(new)
tracker.write_bytes(tracker_raw)
proof = {'time_cst': now.isoformat(), 'section': '20.105', 'bytes': len(new),
         'sha256': hashlib.sha256(new).hexdigest(), 'three_master_copies_equal': master.read_bytes() == desktop.read_bytes() == new,
         'training_screen': 47112, 'queue_screen': 47242, 'native_cpu_pass': True, 'native_gpu_started': False,
         'native_preparation_receipt_sha256': hashlib.sha256(receipt_raw).hexdigest(), 'goal_complete': False}
for name, raw in [('handoff_sync_20_105.json', (json.dumps(proof, indent=2, sort_keys=True) + '\n').encode()),
                  ('publish_from_local.py', Path(__file__).read_bytes()),
                  ('launch_conditional.py', (repo / 'scripts/launch_native_candidate_range_preflight.py').read_bytes()),
                  ('collection.json', (prep / 'collection.json').read_bytes())]:
    (prep / name).write_bytes(raw)
    with sftp.open('/root/autodl-tmp/mcln_native_range_preparation_20260907_v2/' + name, 'wx') as stream:
        stream.write(raw)
sftp.close()
client.close()
with (repo / 'MANIFEST.md').open('ab') as stream:
    stream.write(('| ' + when + ' | continuation | refine-logs/native_range_preparation_20260907_v2/receipt.json | preparation | Native extent CPU62PASS;real weights/load only;Scan pair and queue continue |\n').encode())
print(json.dumps(proof), flush=True)
