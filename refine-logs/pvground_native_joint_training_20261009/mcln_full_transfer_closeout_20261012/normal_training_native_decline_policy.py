"""Stop only after two consecutive complete native Acc@0.5 decreases."""
import json
from pathlib import Path


def decline_decision(history):
    assert history
    assert [r['epoch'] for r in history] == list(range(history[-1]['epoch'] + 1))
    assert all(r['rows'] == 9508 and r['primary_score'] == 'last/bbs' for r in history)
    streak = 0
    for previous, current in zip(history, history[1:]):
        streak = streak + 1 if current['hits050'] < previous['hits050'] else 0
    return dict(epoch=history[-1]['epoch'], patience=2,
                monitor='complete9508_last/bbs_hits050',
                consecutive_declines=streak, stop=streak >= 2,
                equality_or_increase_resets_streak=True,
                hits050=[row['hits050'] for row in history])


def record_decline_decision(log_dir):
    root = Path(log_dir)
    history = [json.loads(line) for line in (root / 'native_metrics.jsonl').read_text().splitlines()]
    result = decline_decision(history)
    (root / 'native_early_stop.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
