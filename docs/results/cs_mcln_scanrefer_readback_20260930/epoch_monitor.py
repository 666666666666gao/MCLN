import json
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


CST = timezone(timedelta(hours=8))
RUN = Path('/root/cs_mcln_scanrefer_readback_20260930')
BEST = Path('/root/cs_mcln_scanrefer_readback_best_20260930')
LOG = Path(str(RUN) + '.log')
EXIT = Path(str(RUN) + '.exit.txt')
PID = 187859
LATEST_BYTES = 814770370
BEST_BYTES = 606018228
SAVE_PEAK = max(2 * LATEST_BYTES + BEST_BYTES, LATEST_BYTES + 2 * BEST_BYTES)
FIRST = datetime(2026, 9, 30, 19, 10, tzinfo=CST).timestamp()


def stamp(value):
    return datetime.fromtimestamp(value, CST).isoformat()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def command(*args):
    return subprocess.run(args, capture_output=True, text=True).stdout.strip()


due = FIRST
seen = 0
while True:
    write_json(RUN / 'monitor_state.json', {
        'updated_at_cst': stamp(time.time()),
        'last_observed_complete_epoch': seen,
        'next_check_cst': stamp(due),
        'recheck_if_incomplete_seconds': 240,
        'training_pid': PID,
    })
    time.sleep(max(0, due - time.time()))
    lines = LOG.read_text(encoding='utf-8').splitlines()
    committed = [json.loads(line[len('EPOCH cs_readback '):].split(' CHECKPOINT_BYTES ')[0])
                 for line in lines if line.startswith('EPOCH cs_readback ')]
    last = committed[-1]['epoch'] if committed else 0
    alive = Path(f'/proc/{PID}').exists()
    terminal = EXIT.read_text(encoding='utf-8').strip() if EXIT.exists() else None
    free = {name: os.statvfs(name).f_bavail * os.statvfs(name).f_frsize
            for name in ('/root', '/root/autodl-tmp')}
    weights = {}
    for name, path in (('latest', RUN / 'latest.pth'), ('best', BEST / 'best.pth')):
        if path.exists():
            info = path.stat()
            weights[name] = {'path': str(path), 'bytes': info.st_size, 'mtime_cst': stamp(info.st_mtime)}
    receipt = {
        'checked_at_cst': stamp(time.time()),
        'expected_epoch': seen + 1,
        'completed_epoch': last,
        'training_alive': alive,
        'exit_code': terminal,
        'process': command('ps', '-p', str(PID), '-o', 'pid,stat,etime,rss,cmd', '--no-headers'),
        'gpu': command('nvidia-smi', '--query-gpu=name,memory.used,utilization.gpu', '--format=csv,noheader'),
        'disk_free_bytes': free,
        'weights': weights,
        'required_extra_atomic_save_bytes': SAVE_PEAK - sum(x['bytes'] for x in weights.values()),
        'log_tail': lines[-20:],
    }
    if last:
        result = json.loads((RUN / f'epoch_{last}.json').read_text(encoding='utf-8'))
        assert result == committed[-1]
        receipt['epoch'] = result
        receipt['best'] = json.loads((BEST / 'best.json').read_text(encoding='utf-8'))
        endpoint = (RUN / f'epoch_{last}.json').stat().st_mtime
        previous = (RUN / f'epoch_{last - 1}.json').stat().st_mtime
        duration = endpoint - previous
        receipt['measured_latest_epoch_including_eval_seconds'] = duration
        receipt['estimated_epoch21_cst'] = stamp(endpoint + (21 - last) * duration)
        receipt['baseline'] = json.loads((RUN / 'epoch_0.json').read_text(encoding='utf-8'))
    if last > seen:
        write_json(RUN / f'monitor_epoch_{last:02d}.json', receipt)
        print('COMPLETE_EPOCH_OBSERVATION', json.dumps(receipt, ensure_ascii=False), flush=True)
        seen = last
    if terminal is not None or not alive:
        write_json(RUN / 'monitor_terminal.json', receipt)
        write_json(RUN / 'monitor_state.json', {
            'updated_at_cst': stamp(time.time()), 'last_observed_complete_epoch': seen,
            'next_check_cst': None, 'training_pid': PID, 'exit_code': terminal,
        })
        print('TERMINAL_OBSERVATION', json.dumps(receipt, ensure_ascii=False), flush=True)
        break
    write_json(RUN / 'monitor_latest_observation.json', receipt)
    due = max(time.time() + 240, endpoint + duration + 300) if last == seen and last else time.time() + 240
    if seen == 21:
        due = time.time() + 240
