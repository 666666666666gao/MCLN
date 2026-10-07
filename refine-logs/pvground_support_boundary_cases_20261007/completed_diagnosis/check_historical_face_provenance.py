"""Three meaningful certainty checks; synthetic and not formal results."""
import json
from pathlib import Path

import numpy as np
from historical_face_provenance import historical_faces

root = Path(__file__).resolve().parent
xyz = np.zeros((50000, 3), dtype=np.float32)
xyz[-6:] = np.asarray([[-1,0,0], [1,0,0], [0,-1,0], [0,1,0], [0,0,-1], [0,0,1]])
target = np.ones(50000, dtype=bool)
target[-6:] = False
superpoint = np.zeros(50000, dtype=np.int32)
superpoint[-6:] = np.arange(1, 7)
truth = np.asarray([0,0,0,2.4,2.4,2.4], dtype=np.float32)
arrays = dict(xyz=xyz, target=target, superpoint=superpoint, root_gt=truth)
cached = dict(row_id=0, query=0, reference_box=[0,0,0,2,2,2], reference_valid=True,
    root_gt=truth.tolist())
first = historical_faces(arrays, cached)['faces'][0]
assert first['member_certainty'] == 'only_background_member_possible'
assert first['all_possible_superpoints_pure_background']
xyz[-7] = [-1,0,0]
second = historical_faces(arrays, cached)['faces'][0]
assert second['member_certainty'] == 'target_and_background_members_possible'
assert not second['all_possible_superpoints_pure_background']
xyz[-7] = [0,0,0]
superpoint[-6] = 0
third = historical_faces(arrays, cached)['faces'][0]
assert third['member_certainty'] == 'only_background_member_possible'
assert third['all_possible_superpoints_mixed']
assert not third['all_possible_superpoints_pure_background']
receipt = dict(status='PASS_SYNTHETIC_CERTAINTY_CHECKS', checks=3, neural_forwards=0,
    formal_validation_cases=0, checks_names=['pure-background source',
        'coincident target/background ambiguity', 'background extremum in mixed superpoint'])
(root / 'HISTORICAL_FACE_LOCAL_CHECK.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
print(json.dumps(receipt), flush=True)
