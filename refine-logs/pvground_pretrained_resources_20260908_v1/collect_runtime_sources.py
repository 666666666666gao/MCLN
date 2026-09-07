import concurrent.futures
import datetime
import hashlib
import json
import urllib.request
from pathlib import Path

root = Path('C:/Users/gb/.codex_mcln_g0_20260905/refine-logs/pvground_pretrained_resources_20260908_v1')
commit = '262e2592589baec7bb83a0d46aae6542d4ccedfb'
names = ['models/pv_backbone.py', 'models/pv_utils.py', 'prepare_data.py', 'wandb_config.yaml']
tree_raw = Path('C:/Users/gb/.codex/tmp/pvground_runtime_tree_20260908.json').read_bytes()
tree = {row['path']: row for row in json.loads(tree_raw)['tree']}

def fetch(name):
    url = 'https://raw.githubusercontent.com/AaNnWwTt/PV-Ground/' + commit + '/' + name
    with urllib.request.urlopen(url, timeout=30) as response:
        raw = response.read()
    assert len(raw) == tree[name]['size']
    git_sha = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    assert git_sha == tree[name]['sha']
    destination = root / 'source' / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        stream.write(raw)
    return name, {'url': url, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'git_blob': git_sha}

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    files = dict(executor.map(fetch, names))
receipt = {'status': 'complete', 'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(), 'github_commit': commit, 'files': files, 'model_executed': False}
(root / 'runtime_source_receipt.json').write_bytes((json.dumps(receipt, indent=2) + '\n').encode())
(root / 'github_tree.json').write_bytes(tree_raw)
(root / 'collect_runtime_sources.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(receipt))
