"""Correct the source review's concrete cross-benchmark defects, isolated only."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parent / 'referit_native_preparation_20261010'
review = json.loads((root / 'source_review/EXPERIMENT_CODE_REVIEW.json').read_bytes())
assert review['blocking_issue_count'] > 0
prior = json.loads((root / 'SOURCE_ADAPTER_PREPARATION.json').read_bytes())
for name, digest in prior['prepared_source_sha256'].items():
    assert hashlib.sha256((root / 'source' / name).read_bytes()).hexdigest() == digest
closed = root / 'revision1_before_metric_correction'
assert not closed.exists()
closed.mkdir()
for name in ('source', 'refine-logs', 'idea-stage'):
    shutil.copytree(root / name, closed / name)
for name in ('SOURCE_ADAPTER_PREPARATION.json', 'check_native_targets_cpu.py',
             'run_native_targets_cpu_authorized.py', 'CPU_INPUTS.json'):
    shutil.copyfile(root / name, closed / name)


def replace(path, old, new):
    raw = path.read_bytes()
    ending = b'\r\n' if path.suffix == '.py' and path.parent.name == 'source' else b'\n'
    old_bytes = old.encode('utf-8').replace(b'\n', ending)
    new_bytes = new.encode('utf-8').replace(b'\n', ending)
    assert raw.count(old_bytes) == 1, str(path)
    path.write_bytes(raw.replace(old_bytes, new_bytes))


replace(root / 'source/train_dist_mod.py',
    '        # NOTE Main eval branch\n        test_loader = tqdm(test_loader, ascii=True)',
    '        # NOTE Main eval branch\n        expected_rows = len(test_loader.dataset)\n        test_loader = tqdm(test_loader, ascii=True)')
replace(root / 'source/train_dist_mod.py',
    '        assert rows == [9508, 9508]\n        return dict(rows=9508, hits025=int(evaluator.dets[keys[0]]),',
    '        assert rows == [expected_rows, expected_rows]\n        return dict(rows=expected_rows, hits025=int(evaluator.dets[keys[0]]),')
replace(root / 'source/main_utils.py',
    "    return (metrics['hits025'] >= 5658 and metrics['hits050'] >= 4850,\n"
    "            metrics['hits050'] >= 4850, metrics['hits025'], metrics['hits050'])",
    "    # Nr/Sr each select strict REC first, then wide REC on the same checkpoint.\n"
    "    return (metrics['hits050'], metrics['hits025'])")
replace(root / 'source/main_utils.py',
    "    assert metrics['rows'] == 9508 and metrics['primary_score'] == 'last/bbs'",
    "    assert metrics['primary_score'] == 'last/bbs'")
replace(root / 'run_native_targets_cpu_authorized.py',
    "environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')",
    "runtime_spec=json.loads(Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/env_spec.json').read_bytes())\n"
    "environment=dict(os.environ,**runtime_spec['env'])\n"
    "environment.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')")
replace(root / 'check_native_targets_cpu.py',
    'import importlib.util\nimport json', 'import importlib.util\nimport inspect\nimport json')
replace(root / 'check_native_targets_cpu.py',
    'from models.losses import SetCriterion',
    "from models.losses import SetCriterion\n"
    "criterion_source = Path(inspect.getfile(SetCriterion)).resolve()\n"
    "assert criterion_source == (Path(bundle['native_model_source']) / 'models/losses.py').resolve()")
replace(root / 'check_native_targets_cpu.py',
    "    criterion_source=str(Path(bundle['native_model_source']) / 'models/losses.py'),",
    "    criterion_source=str(criterion_source),")
for name in ('EXPERIMENT_PLAN_20261010.md', 'EXPERIMENT_PLAN.md'):
    path = root / 'refine-logs' / name
    replace(path,
        'A 输出层为零；B 中性门控为固定 0.5 的初始值，因此整个方法的 E0 不等于原版作者模型。',
        'A 输出层为零；B 输出层为零，经 clamp 后门值为 0，初始直接使用 Mask 空间参考，因此整个方法的 E0 不等于原版作者模型。')
    replace(path,
        '源码 | 14 文件隔离副本，仅改初始化、入口、G/C 和 criterion 调用，共 5 文件；保持活动来源原字节',
        '源码 | 14 文件隔离副本；初始化、入口、G/C、criterion 调用以及必要的记录／选模适配，共 6 文件；保持活动来源原字节')
    with path.open('a', encoding='utf-8') as stream:
        stream.write('\n## 初次独立复核后的必要修正\n\n'
            '原 ScanRefer 验证／记录的 9508 硬断言及 5658/4850 保存门槛不能用于 Nr/Sr。'
            'Nr/Sr 验证改为核对实际完整 loader 数据集长度，两者分别按 Acc@0.5 命中、再 Acc@0.25 命中选择同一 best，仍报告固定终点。'
            '选模规则不是成功门槛；两项相对对应 baseline 的要求保留。'
            'CPU 工程子进程按已有实际 import 检查读取同一 runtime env/PYTHONPATH，再显式关闭 CUDA；'
            '原复核失败报告及本轮修正前源码已保存，活动训练未修改。\n')
prepared_hashes = {}
changed = []
parent_source = root.parent / 'source'
for name, old_digest in prior['original_source_sha256'].items():
    assert hashlib.sha256((parent_source / name).read_bytes()).hexdigest() == old_digest
    raw = (root / 'source' / name).read_bytes()
    ast.parse(raw, filename=name)
    digest = hashlib.sha256(raw).hexdigest()
    prepared_hashes[name] = digest
    if digest != old_digest:
        changed.append(name)
for name in ('check_native_targets_cpu.py', 'run_native_targets_cpu_authorized.py'):
    ast.parse((root / name).read_bytes(), filename=name)
assert len(changed) == 6 and 'main_utils.py' in changed
report = dict(prior)
report.update(status='REFERIT_NATIVE_SOURCE_ADAPTERS_R2_PREPARED_NOT_EXECUTED',
    time_cst=datetime.datetime.now().astimezone().isoformat(), changed_files=sorted(changed),
    prepared_source_sha256=prepared_hashes, source_revision=2,
    first_source_review=str(root / 'source_review/EXPERIMENT_CODE_REVIEW.json'),
    first_source_review_sha256=hashlib.sha256((root / 'source_review/EXPERIMENT_CODE_REVIEW.json').read_bytes()).hexdigest(),
    first_revision_snapshot=str(closed), source_review_pending=True,
    fresh_B_initial_gate=0.0, fresh_B_initial_geometry='Mask_reference',
    evaluation_rows='actual complete loader dataset length',
    Nr_Sr_selection='hits050_then_hits025_same_checkpoint',
    runtime_CPU_environment='same observed runtime env_spec env/PYTHONPATH then CUDA disabled',
    active_scanrefer_source_changed=False, GPU_calls=0, SSH_calls=0,
    any_Nr_Sr_training_launched=False)
(root / 'SOURCE_ADAPTER_PREPARATION_R2.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(root / 'SOURCE_ADAPTER_PREPARATION.json').write_bytes((root / 'SOURCE_ADAPTER_PREPARATION_R2.json').read_bytes())
print(json.dumps({k:report[k] for k in ('status','changed_files','fresh_B_initial_gate',
    'active_scanrefer_source_changed','any_Nr_Sr_training_launched')}))
