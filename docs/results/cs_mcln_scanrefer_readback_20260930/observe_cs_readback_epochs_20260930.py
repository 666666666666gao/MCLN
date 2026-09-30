from datetime import datetime, timezone, timedelta
from pathlib import Path
import importlib.util
import time


SOURCE = Path(r'C:\Users\gb\.codex\tmp\collect_cs_readback_epochs_20260930.py')
spec = importlib.util.spec_from_file_location('cs_readback_epoch_collector', SOURCE)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)
due = datetime(2026, 9, 30, 19, 10, 30, tzinfo=timezone(timedelta(hours=8))).timestamp()
print('FIRST_COLLECTION_CST', '2026-09-30 19:10:30', flush=True)
while True:
    time.sleep(max(0, due - time.time()))
    packet = collector.collect()
    if packet['terminal']:
        break
    due = max(time.time() + 240, datetime.fromisoformat(packet['next_check_cst']).timestamp())
