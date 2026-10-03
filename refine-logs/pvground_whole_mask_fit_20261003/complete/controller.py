"""Serial fixed-budget pair; retire verified endpoints that are not the metric best."""
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
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def box_iou(box, target):
    assert len(box) == len(target) == 6
    assert all(math.isfinite(x) for x in box+target)
    assert all(x > 0 for x in box[3:]+target[3:])
    extent = [max(0., min(box[i]+box[i+3]/2, target[i]+target[i+3]/2)
        - max(box[i]-box[i+3]/2, target[i]-target[i+3]/2)) for i in range(3)]
    intersection = extent[0]*extent[1]*extent[2]
    return intersection/(box[3]*box[4]*box[5]+target[3]*target[4]*target[5]-intersection)


root = Path(__file__).parent
assert not (root/'status.json').exists()
spec = json.loads((root/'whole_range/spec.json').read_bytes())
environment = json.loads((Path(spec['runtime'])/'env_spec.json').read_bytes())
python = spec['runtime']+'/venv/bin/python'
parent = Path(spec['base_terminal'])
parent_sha = spec['base_terminal_sha256']
assert sha(parent) == parent_sha
best = dict(name='original_g', path=str(parent), sha256=parent_sha,
    bbs_hits25=5615, bbs_hits50=4495, protected_parent=True)
record = dict(status='running', started_cst=now(), completed=[], retained_best=best,
    primary_mode='bbs', primary_threshold=.5, goal_achieved=False)


def save():
    temporary = root/'status.json.tmp'
    temporary.write_text(json.dumps(record, indent=2)+'\n')
    os.replace(str(temporary), str(root/'status.json'))


for arm in ('local_range', 'whole_range'):
    directory = root/arm
    arm_spec = json.loads((directory/'spec.json').read_bytes())
    assert arm_spec['base_terminal_sha256'] == parent_sha
    proof = json.loads((Path(arm_spec['preflight_root'])/'preflight.json').read_bytes())
    assert proof['status'] == 'pass' and proof['batch_size'] == 8 and proof['optimizer_steps'] == 2
    assert proof['support_arm'] == arm and proof['use_whole_range'] == (arm == 'whole_range')
    assert proof['same_cached_inputs_zero_head_pair_exact']
    assert proof['optimizer_exact_check']['all_keys_moments_steps_and_groups_exact']
    required = 2*proof['serialization_bytes'] + 256*1024**2
    assert shutil.disk_usage(root).free >= required
    variables = dict(environment['env'])
    variables['PYTHONPATH'] = str(directory)+':'+variables['PYTHONPATH']
    for mode in ('train', 'formal'):
        begin = time.monotonic()
        command = ['env']+[key+'='+value for key, value in variables.items()]+[
            python, '-B', '-u', str(root/'run_whole_mask_fit.py'), '--spec', str(directory/'spec.json'), '--mode', mode]
        record.update(stage=arm+'/'+mode, stage_started_cst=now())
        with (directory/(mode+'.log')).open('x') as output:
            child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
            record['process_pid'] = child.pid
            save()
            code = child.wait()
        (directory/(mode+'.exit')).write_text(str(code)+'\n')
        if code:
            record.update(status='failed', exit_code=code, finished_cst=now())
            save()
            raise SystemExit(code)
        record['completed'].append(dict(arm=arm, mode=mode, seconds=time.monotonic()-begin, finished_cst=now()))
        save()
    fit = json.loads((directory/'receipt.json').read_bytes())
    formal = json.loads((directory/'formal/receipt.json').read_bytes())
    assert fit['status'] == 'complete' and fit['training_steps'] == 3723
    assert fit['fit_seen_exactly_once'] and fit['fit_rows'] == 29778
    assert formal['status'] == 'pass' and formal['rows'] == formal['formal_rows'] == 9508
    rows_path = directory/'formal/rows.jsonl'
    assert sha(rows_path) == formal['rows_sha256']
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    assert [r['row_id'] for r in rows] == list(range(9508))
    cpu = {}
    for mode in ('bbs', 'bbf'):
        ious = [box_iou(r[mode]['box'], r['root_box']) for r in rows]
        threshold_changes = sum((u > t) != (r[mode]['iou'] > t)
            for u, r in zip(ious, rows) for t in (.25, .5))
        assert threshold_changes == 0, (arm, mode, threshold_changes)
        hit25 = sum(u > .25 for u in ious)
        hit50 = sum(u > .5 for u in ious)
        metrics = formal['metrics'][mode]
        assert hit25 == metrics['rec_hits25'] and hit50 == metrics['rec_hits50']
        cpu[mode] = dict(rec_hits25=hit25, rec_hits50=hit50, cpu_threshold_changes=0)
    terminal = directory/'terminal.pth'
    terminal_sha = sha(terminal)
    assert terminal_sha == fit['terminal_sha256']
    assert sha(parent) == parent_sha
    candidate = dict(name=arm, path=str(terminal), sha256=terminal_sha,
        bbs_hits25=cpu['bbs']['rec_hits25'], bbs_hits50=cpu['bbs']['rec_hits50'], protected_parent=False)
    retire = terminal
    retire_sha = terminal_sha
    if candidate['bbs_hits50'] > best['bbs_hits50']:
        retire = None if best['protected_parent'] else Path(best['path'])
        retire_sha = best['sha256']
        best = candidate
    deleted = []
    if retire is not None:
        assert retire.parent in (root/'local_range', root/'whole_range')
        assert retire.name == 'terminal.pth' and str(retire) != best['path']
        assert retire != parent and sha(retire) == retire_sha
        size = retire.stat().st_size
        retire.unlink()
        assert not retire.exists()
        deleted.append(dict(path=str(retire), bytes=size, sha256=retire_sha))
    assert sha(parent) == parent_sha
    receipt = dict(time_cst=now(), arm=arm, formal_rows_sha256=formal['rows_sha256'],
        terminal_sha256=terminal_sha, cpu_box_threshold_recount=cpu,
        model_or_optimizer_replayed=False, retained_best=best, deleted=deleted,
        authorization='user persistent instruction: retain metric best; remove our nonbest weights',
        local_weight_archive_created=False, parent_sha256_after=parent_sha)
    (directory/'weight_retention.json').write_text(json.dumps(receipt, indent=2)+'\n')
    record.update(retained_best=best, last_weight_retention=receipt)
    save()
record.update(status='complete', finished_cst=now(), scanrefer_target_pass=
    best['bbs_hits25'] >= 5615 and best['bbs_hits50'] >= 4754)
save()
print(json.dumps(record), flush=True)
