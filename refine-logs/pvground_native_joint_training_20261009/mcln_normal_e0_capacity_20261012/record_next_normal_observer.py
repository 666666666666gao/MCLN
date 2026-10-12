"""Persist the already launched local timer owner; no remote query."""
import argparse
import datetime
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--owner', required=True)
options = parser.parse_args()
root = Path(__file__).resolve().parent
owner_path = Path(options.owner)
assert owner_path.parent.resolve() == (root / 'normal_training').resolve()
owner = json.loads(owner_path.read_bytes())
assert owner['remote_controller_pid'] == 271902
state = json.loads((root / 'CURRENT_PROGRESS.json').read_bytes())
now = datetime.datetime.now().astimezone().isoformat()
state.update(updated_cst=now, observer_pid=owner['local_pid'], observer_native_session=None,
             observer_created_cst=owner['created_cst'], next_observation_cst=owner['due_cst'],
             next_observer_owner=str(owner_path), full_goal_complete=False)
(root / 'CURRENT_PROGRESS.json').write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
base = Path('C:/Users/gb/.codex/tmp')
for path in (base / 'pvground_native_joint_training_20261009/native_joint_v2/NORMAL_CONTINUATION_STATE.json',
             base / 'pvground_native_joint_training_20261009/native_joint_v2/face_native_gpu_v2_20261011/CURRENT_PROGRESS.json',
             base / 'pvground_referit_mask_reference_20261006/current_research_goals.json',
             base / 'pvground_query_supported_geometry_20261005/active_continuation_state.json'):
    data = json.loads(path.read_bytes())
    data.update(user_mcln_baseline_transfer=state, updated_cst=now, full_goal_complete=False)
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(observer_pid=owner['local_pid'], next_observation_cst=owner['due_cst'],
                     new_remote_queries=0)))
