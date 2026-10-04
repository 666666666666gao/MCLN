"""Bounded final-quality test, using the executed native+G control unchanged."""
import argparse
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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu_iou(box, truth):
    assert len(box) == len(truth) == 6
    assert all(math.isfinite(value) for value in box + truth)
    assert all(value > 0 for value in box[3:] + truth[3:])
    sides = [max(0., min(box[axis] + box[axis+3]/2, truth[axis] + truth[axis+3]/2)
        - max(box[axis] - box[axis+3]/2, truth[axis] - truth[axis+3]/2)) for axis in range(3)]
    intersection = sides[0]*sides[1]*sides[2]
    return intersection / (box[3]*box[4]*box[5] + truth[3]*truth[4]*truth[5] - intersection)


parser = argparse.ArgumentParser()
parser.add_argument('--stage', choices=['preflight', 'fit'], required=True)
args = parser.parse_args()
root = Path(__file__).parent
stage_file = root / (args.stage + '_status.json')
assert not stage_file.exists()
directory = root / ('preflight' if args.stage == 'preflight' else 'quality')
spec = json.loads((directory / 'spec.json').read_bytes())
assert spec['root'] == str(directory) and spec['use_geometry_evidence'] and spec['quality_weight'] == 1.0
assert spec['batch_size'] == 8 and spec['updates'] == 3723 and spec['fit_passes'] == 1
environment = json.loads((Path(spec['runtime'])/'env_spec.json').read_bytes())
variables = dict(environment['env'])
variables['PYTHONPATH'] = str(directory) + ':' + variables['PYTHONPATH']
parents = {Path(spec[key]): spec[key+'_sha256'] for key in ('base_terminal', 'geometry_terminal')}
official = environment['weight_dirs']['scanrefer']
parents[Path(official['path'])] = official['sha256']
assert all(sha(path) == digest for path, digest in parents.items())
record = dict(status='running', stage=args.stage, started_cst=now(), completed=[],
    protected_best_hits=[5616,4506], goal_achieved=False)


def save():
    temporary = Path(str(stage_file)+'.tmp')
    temporary.write_text(json.dumps(record, indent=2)+'\n')
    os.replace(str(temporary), str(stage_file))


if args.stage == 'fit':
    proof = json.loads((root/'preflight/preflight.json').read_bytes())
    assert proof['status'] == 'pass' and proof['optimizer_steps'] == 2 and proof['batch_size'] == 8
    assert proof['isolated_quality_route_verified'] and proof['geometry_provider_and_g_states_exact']
    assert proof['initial_zero_output_native_exact'] and proof['quality_weight'] == 1.0
    assert proof['optimizer_exact_check']['all_keys_moments_steps_and_groups_exact']
    assert proof['weight_files_created'] == 0 and not proof['accuracy_result']
    tested = json.loads((root/'preflight/spec.json').read_bytes())
    assert tested['runner_files'] == spec['runner_files']
    assert shutil.disk_usage(root).free >= 2*proof['serialization_bytes'] + 256*1024**2
    modes = ('train','formal')
else:
    modes = ('preflight',)

for mode in modes:
    begin = time.monotonic()
    command = ['env'] + [key+'='+value for key,value in variables.items()] + [
        spec['runtime']+'/venv/bin/python','-B','-u',str(directory/'run_final_quality_fit.py'),
        '--spec',str(directory/'spec.json'),'--mode',mode]
    record.update(mode=mode, mode_started_cst=now())
    with (directory/(mode+'.log')).open('x') as output:
        child = subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT)
        record['process_pid'] = child.pid
        save()
        code = child.wait()
    (directory/(mode+'.exit')).write_text(str(code)+'\n')
    if code:
        record.update(status='failed', exit_code=code, finished_cst=now())
        save()
        raise SystemExit(code)
    record['completed'].append(dict(mode=mode, elapsed_seconds=time.monotonic()-begin, finished_cst=now()))
    save()

if args.stage == 'fit':
    receipt = json.loads((directory/'receipt.json').read_bytes())
    formal = json.loads((directory/'formal/receipt.json').read_bytes())
    restore = json.loads((directory/'formal_restore.json').read_bytes())
    assert receipt['training_steps'] == 3723 and receipt['fit_rows'] == 29778
    assert receipt['fit_seen_exactly_once'] and receipt['frozen_parent_states_exact']
    assert formal['rows'] == formal['formal_rows'] == 9508 and formal['status'] == 'pass'
    assert restore['strict_model_restore'] and restore['restored_steps'] == 3723
    control = Path(spec['control_root'])
    old = [json.loads(line) for line in (control/'train.jsonl').read_text().splitlines()]
    new = [json.loads(line) for line in (directory/'train.jsonl').read_text().splitlines()]
    assert len(old) == len(new) == 3723
    assert [r['rows'] for r in new] == [r['rows'] for r in old]
    assert [r['step'] for r in new] == list(range(1,3724))
    assert all(math.isfinite(r['quality_loss']) for r in new)
    terminal = directory/'terminal.pth'
    assert sha(terminal) == receipt['terminal_sha256'] == restore['terminal_sha256']
    row_file = directory/'formal/rows.jsonl'
    assert sha(row_file) == formal['rows_sha256']
    rows = [json.loads(line) for line in row_file.read_text().splitlines()]
    assert [r['row_id'] for r in rows] == list(range(9508))
    ious = [cpu_iou(r['bbs']['box'],r['root_box']) for r in rows]
    assert sum((u>t)!=(r['bbs']['iou']>t) for u,r in zip(ious,rows) for t in (.25,.5)) == 0
    hits = [sum(u>t for u in ious) for t in (.25,.5)]
    assert hits == [formal['metrics']['bbs']['rec_hits25'],formal['metrics']['bbs']['rec_hits50']]
    best = dict(path=spec['geometry_terminal'],sha256=spec['geometry_terminal_sha256'],hits=[5616,4506])
    deleted = []
    if hits[1] > 4506:
        best = dict(path=str(terminal),sha256=sha(terminal),hits=hits)
    else:
        assert terminal.resolve().parent == directory.resolve() and terminal.name == 'terminal.pth'
        assert terminal not in parents and str(terminal) != best['path']
        deleted.append(dict(path=str(terminal),bytes=terminal.stat().st_size,sha256=sha(terminal)))
        terminal.unlink()
        assert not terminal.exists()
    assert all(sha(path) == digest for path,digest in parents.items())
    retention = dict(time_cst=now(),formal_rows_sha256=formal['rows_sha256'],cpu_box_threshold_changes=0,
        terminal_sha256=restore['terminal_sha256'],cpu_bbs_hits25=hits[0],cpu_bbs_hits50=hits[1],
        same_control_sample_order=True,retained_best=best,deleted=deleted,
        required_parent_chain_preserved=True,local_weight_archive_created=False,
        authorization='user: remove our completed nonbest weights, retain metric best and required parents')
    (directory/'weight_retention.json').write_text(json.dumps(retention,indent=2)+'\n')
    record.update(retained_best=best,weight_retention=retention,
        scanrefer_target_pass=best['hits'][0]>=5615 and best['hits'][1]>=4754)
record.update(status='complete',finished_cst=now(),exit_code=0)
save()
print(json.dumps(record),flush=True)
