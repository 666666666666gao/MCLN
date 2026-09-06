
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch
directory, source = map(Path, sys.argv[1:])
sys.path.insert(0, str(source))
spec = importlib.util.spec_from_file_location('audit', directory / 'audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
B, Q = 4, 4
boxes = torch.zeros(B, Q, 6)
boxes[..., 3:] = 1.
boxes[:, :, 0] = torch.arange(Q).float() * 3.
boxes[1, 0, 0] = .4
roots = torch.zeros(B, 1, 6)
roots[..., 3:] = 1.
scores = torch.tensor([[0., 4., 2., 1.], [4., 0., 2., 1.],
                       [4., 0., 2., 1.], [4., 0., 2., 1.]])
parents = torch.zeros(B, 16, dtype=torch.long)
parents[:, 1] = 1
variants = boxes.gather(1, parents[..., None].expand(-1, -1, 6))
variants = variants[:, :, None].expand(-1, -1, 7, -1).reshape(B, 112, 6).clone()
variants[1, 1] = roots[1, 0]
variants[3, 8] = roots[3, 0]
variant_scores = torch.zeros(B, 112)
variant_scores[torch.arange(B), torch.tensor([0, 1, 7, 8])] = 1.
runtime = {'rec_geometry_boxes': variants, 'rec_geometry_scores': variant_scores,
           'rec_geometry_valid_mask': torch.ones(B, 112, dtype=torch.bool)}
matches = [(torch.tensor([0]), torch.tensor([0])) for _ in range(B)]
default = scores.argmax(dim=1)
rows = audit.transfer_rows(boxes, scores, default, parents, runtime, matches, roots)
summary = audit.summarize(rows)
assert summary['teacher_effects']['native']['0.5'] == {'repair': 2, 'damage': 1, 'net': 1}
assert summary['teacher_iou_greater_than']['native_best'] == 1
assert summary['teacher_iou_greater_than']['hungarian_root'] == 1
assert summary['teacher_passing_but_corresponding_query_failing']['0.5'] == 1
assert summary['geometry_correspondence_differs_from_source_query'] == 1
assert rows[2]['ious']['teacher'] == 0. and rows[2]['ious']['native'] == 1.
permutation = torch.tensor([2, 0, 3, 1])
inverse = permutation.argsort()
permuted_matches = [(inverse[pred], target) for pred, target in matches]
permuted = audit.transfer_rows(boxes[:, permutation], scores[:, permutation], inverse[default],
    inverse[parents], runtime, permuted_matches, roots)
assert audit.summarize(permuted) == summary
assert all(a['ious'] == b['ious'] and a['corresponding_box'] == b['corresponding_box']
           for a, b in zip(rows, permuted))
receipt = {'schema': 'mcln-scanrefer-teacher-transfer-cpu-v1', 'status': 'pass',
    'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'synthetic_rows': 4, 'real_data_rows': 0, 'optimizer_steps': 0, 'gpu_forwards': 0,
    'teacher_worse_than_native_retained_as_damage': True,
    'ranking_gain_separated_from_geometry_gain': True,
    'geometry_correspondence_invariant_to_query_permutation': True,
    'teacher_better_than_top1_is_not_assumed_better_than_hungarian': True,
    'summary': summary, 'source_sha256': hashlib.sha256((directory / 'audit.py').read_bytes()).hexdigest()}
audit.write_json(directory / 'receipt.json', receipt)
print(json.dumps(receipt), flush=True)
