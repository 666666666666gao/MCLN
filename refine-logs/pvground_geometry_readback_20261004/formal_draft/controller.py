"""Future bounded same-capacity readback comparison; keep the best and all required parents."""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def box_iou(box, target):
    assert len(box) == len(target) == 6
    assert all(math.isfinite(value) for value in box + target)
    assert all(value > 0 for value in box[3:] + target[3:])
    extent = [max(0., min(box[axis] + box[axis + 3] / 2, target[axis] + target[axis + 3] / 2)
        - max(box[axis] - box[axis + 3] / 2, target[axis] - target[axis + 3] / 2)) for axis in range(3)]
    intersection = extent[0] * extent[1] * extent[2]
    return intersection / (box[3] * box[4] * box[5] + target[3] * target[4] * target[5] - intersection)


root = Path(__file__).parent
assert not (root / 'status.json').exists()
first = json.loads((root / 'evidence_hidden/spec.json').read_bytes())
environment = json.loads((Path(first['runtime']) / 'env_spec.json').read_bytes())
parents = {Path(first[name]): first[name + '_sha256'] for name in ('base_terminal', 'geometry_terminal')}
official = environment['weight_dirs']['scanrefer']
parents[Path(official['path'])] = official['sha256']
assert all(sha(path) == digest for path, digest in parents.items())
best = dict(name='frozen4506_geometry', path=first['geometry_terminal'],
    sha256=first['geometry_terminal_sha256'], bbs_hits25=5616, bbs_hits50=4506, protected_parent=True)
record = dict(status='running', started_cst=now(), completed=[], retained_best=best,
    primary_mode='bbs', primary_threshold=.5, formal_training_started=True, goal_achieved=False)


def save():
    temporary = root / 'status.json.tmp'
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    os.replace(str(temporary), str(root / 'status.json'))


for arm in ('evidence_hidden', 'evidence_visible'):
    directory = root / arm
    spec = json.loads((directory / 'spec.json').read_bytes())
    assert spec['geometry_terminal_sha256'] == first['geometry_terminal_sha256']
    assert spec['base_terminal_sha256'] == first['base_terminal_sha256']
    assert spec['use_geometry_evidence'] == (arm == 'evidence_visible')
    proof = json.loads((Path(spec['preflight_root']) / 'preflight.json').read_bytes())
    assert proof['status'] == 'pass' and proof['optimizer_steps'] == 2 and proof['batch_size'] == 8
    assert proof['use_geometry_evidence'] == spec['use_geometry_evidence']
    assert proof['geometry_provider_and_g_states_exact'] and proof['zero_residual_native_semantic_exact']
    assert proof['readback_parameters'] == 96672 and proof['readback_state_tensors'] == 23
    assert proof['optimizer_exact_check']['all_keys_moments_steps_and_groups_exact']
    assert proof['weight_files_created'] == 0 and not proof['accuracy_result']
    required = 2 * proof['serialization_bytes'] + 256 * 1024**2
    assert shutil.disk_usage(root).free >= required
    variables = dict(environment['env'])
    variables['PYTHONPATH'] = str(directory) + ':' + variables['PYTHONPATH']
    for mode in ('train', 'formal'):
        begin = time.monotonic()
        command = ['env'] + [key + '=' + value for key, value in variables.items()] + [
            spec['runtime'] + '/venv/bin/python', '-B', '-u', str(directory / 'run_readback_fit.py'),
            '--spec', str(directory / 'spec.json'), '--mode', mode]
        record.update(stage=arm + '/' + mode, stage_started_cst=now())
        with (directory / (mode + '.log')).open('x') as output:
            child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
            record['process_pid'] = child.pid
            save()
            code = child.wait()
        (directory / (mode + '.exit')).write_text(str(code) + '\n')
        if code:
            record.update(status='failed', exit_code=code, finished_cst=now())
            save()
            raise SystemExit(code)
        record['completed'].append(dict(arm=arm, mode=mode, seconds=time.monotonic() - begin, finished_cst=now()))
        save()
    fit = json.loads((directory / 'receipt.json').read_bytes())
    formal = json.loads((directory / 'formal/receipt.json').read_bytes())
    restore = json.loads((directory / 'formal_restore.json').read_bytes())
    assert fit['status'] == 'complete' and fit['training_steps'] == 3723 and fit['fit_rows'] == 29778
    assert fit['fit_seen_exactly_once'] and fit['frozen_parent_states_exact']
    assert formal['status'] == 'pass' and formal['rows'] == formal['formal_rows'] == 9508
    assert restore['status'] == 'pass' and restore['strict_model_restore'] and restore['restored_steps'] == 3723
    terminal = directory / 'terminal.pth'
    terminal_sha = sha(terminal)
    assert terminal_sha == fit['terminal_sha256'] == restore['terminal_sha256']
    rows_path = directory / 'formal/rows.jsonl'
    assert sha(rows_path) == formal['rows_sha256']
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    assert [row['row_id'] for row in rows] == list(range(9508))
    ious = [box_iou(row['bbs']['box'], row['root_box']) for row in rows]
    assert sum((value > threshold) != (row['bbs']['iou'] > threshold)
        for value, row in zip(ious, rows) for threshold in (.25, .5)) == 0
    hits25 = sum(value > .25 for value in ious)
    hits50 = sum(value > .5 for value in ious)
    assert hits25 == formal['metrics']['bbs']['rec_hits25'] and hits50 == formal['metrics']['bbs']['rec_hits50']
    candidate = dict(name=arm, path=str(terminal), sha256=terminal_sha,
        bbs_hits25=hits25, bbs_hits50=hits50, protected_parent=False)
    retire = terminal
    retire_sha = terminal_sha
    if hits50 > best['bbs_hits50']:
        retire = None if best['protected_parent'] else Path(best['path'])
        retire_sha = best['sha256']
        best = candidate
    deleted = []
    if retire is not None:
        assert retire.resolve().parent in {(root / name).resolve() for name in ('evidence_hidden', 'evidence_visible')}
        assert retire.name == 'terminal.pth' and str(retire) != best['path'] and retire not in parents
        assert sha(retire) == retire_sha
        size = retire.stat().st_size
        retire.unlink()
        assert not retire.exists()
        deleted.append(dict(path=str(retire), bytes=size, sha256=retire_sha))
    assert all(sha(path) == digest for path, digest in parents.items())
    receipt = dict(time_cst=now(), arm=arm, formal_rows_sha256=formal['rows_sha256'],
        terminal_sha256=terminal_sha, cpu_box_threshold_changes=0, cpu_bbs_hits25=hits25, cpu_bbs_hits50=hits50,
        retained_best=best, deleted=deleted, required_parent_chain_preserved=True, local_weight_archive_created=False,
        authorization='user persistent instruction: retain metric best; remove our completed nonbest weights')
    (directory / 'weight_retention.json').write_text(json.dumps(receipt, indent=2) + '\n')
    record.update(retained_best=best, last_weight_retention=receipt)
    save()
record.update(status='complete', finished_cst=now(), scanrefer_target_pass=
    best['bbs_hits25'] >= 5615 and best['bbs_hits50'] >= 4754)
save()
print(json.dumps(record), flush=True)
