"""Prepare a lock-admitted normal controller and compact training diagnostics."""
import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parent
normal = (root / 'normal_run_controller.py').read_bytes().decode('utf-8')
needle = "assert admission['MCLN_terminal_and_device_released']"
assert normal.count(needle) == 1
normal = normal.replace(needle, "assert admission['MCLN_terminal_observed']\nassert admission['status'] == 'NORMAL_PENDING_LOCKED_DEVICE_ADMISSION'")
needle = "variables = dict(os.environ)"
assert normal.count(needle) == 1
normal = normal.replace(needle, r'''assert hashlib.sha256(json.dumps(environment, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == admission['env_spec_sha256']
terminal = json.loads(Path('/root/autodl-tmp/mcln_portable_best_transfer_20261012/normal_training/status.json').read_bytes())
assert terminal == admission['MCLN_terminal_status']
assert terminal['status'] == 'complete' and terminal['exit_code'] == 0
assert not Path('/proc/271902').exists() and not Path('/proc/271903').exists()
pair_root = Path('/root/autodl-tmp/pvground_deployment_geometry_training_20261012/preflight')
assert json.loads((pair_root / 'status.json').read_bytes()) == admission['actual_pair_status']
assert not Path('/proc/' + str(admission['actual_pair_controller_pid'])).exists()
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']).decode().strip()
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total', '--format=csv,noheader,nounits']).decode().strip().splitlines()
assert len(gpu) == 1
index, name, used, total = [part.strip() for part in gpu[0].split(',')]
assert index == '0' and 'A100' in name and int(used) < 500 and 40000 <= int(total) <= 45000
assert shutil.disk_usage(root).free > admission['reserve_bytes']
stage = json.loads(Path('/root/autodl-tmp/pvground_deployment_geometry_training_20261012/source_stage_manifest.json').read_bytes())
assert stage == admission['source_stage']
for row in stage['source_files'].values():
    assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() == row['sha256']
admission.update(status='NATIVE_NORMAL_TRAINING_ADMITTED_UNDER_GPU_LOCK',
    MCLN_terminal_and_device_released=True, GPU_device_checked=True,
    device_checked_cst=datetime.datetime.now().astimezone().isoformat())
(root / 'admission.json').write_text(json.dumps(admission, indent=2) + '\n')
variables = dict(os.environ)''')
normal = normal.replace('import datetime\n', 'import datetime\nimport hashlib\nimport shutil\n', 1)
ast.parse(normal)
(root / 'normal_run_controller_v3.py').write_bytes(normal.encode('utf-8'))

text = (root / 'source_overlay_v2/main_utils.py').read_bytes().decode('utf-8')
needle = "        stat_dict = {}  # collect statistics\n        model.train()"
assert text.count(needle) == 1
text = text.replace(needle, """        stat_dict = {}  # collect statistics
        if model.module.use_selected_geometry_supervision:
            geometry_responsibility = dict(epoch=epoch, batches=0, rows=0, extra_rows=0,
                extra_loss_sum=0.0, matched_root=0, matched_other=0, unmatched=0)
        model.train()""")
needle = "            optimizer.zero_grad()"
assert text.count(needle) == 1
text = text.replace(needle, """            if model.module.use_selected_geometry_supervision:
                geometry_record = end_points['selected_geometry_extra_record']
                geometry_responsibility['batches'] += 1
                geometry_responsibility['rows'] += len(geometry_record['rows'])
                geometry_responsibility['extra_rows'] += geometry_record['extra_rows']
                geometry_responsibility['extra_loss_sum'] += geometry_record['loss']
                for role, count in geometry_record['selected_roles'].items():
                    geometry_responsibility[role] += count
            optimizer.zero_grad()""")
needle = "        # tensorboard\n        if dist.get_rank() == 0:"
assert text.count(needle) == 1
text = text.replace(needle, """        if model.module.use_selected_geometry_supervision:
            assert geometry_responsibility['batches'] == len(train_loader)
            assert sum(geometry_responsibility[role] for role in
                       ('matched_root', 'matched_other', 'unmatched')) == geometry_responsibility['rows']
            assert geometry_responsibility['extra_rows'] <= geometry_responsibility['unmatched']
            with open(os.path.join(args.log_dir, 'selected_geometry_responsibility.jsonl'), 'a') as stream:
                stream.write(json.dumps(geometry_responsibility) + '\\n')
        # tensorboard
        if dist.get_rank() == 0:""")
ast.parse(text)
output = root / 'source_overlay_v3'
output.mkdir(exist_ok=True)
(output / 'main_utils.py').write_bytes(text.encode('utf-8'))
record = dict(status='LAUNCHER_ADMISSION_AND_TRAINING_COUNT_DIAGNOSTICS_PREPARED_NOT_EXECUTED',
    changed_model_source_files=['main_utils.py'], training_method_changed=False,
    additional_inference_parameters=0, count_diagnostics_only_in_strategy=True,
    R2_source_and_review_preserved=True, active_MCLN_changed=False,
    actual_GPU_calls=0, actual_remote_calls=0, actual_neural_calls=0,
    actual_normal_training_started=False, actual_cold_resume_verified=False)
(root / 'LAUNCHER_SOURCE_PREPARATION.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
print(json.dumps(record))
