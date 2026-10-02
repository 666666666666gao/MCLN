"""Keep all native heads and append one box refinement after Mask generation."""
import argparse
import hashlib
import json
from pathlib import Path
import py_compile
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    shutil.copytree(str(args.parent), str(args.output), ignore=shutil.ignore_patterns('__pycache__'))
    path = args.output/'models/pv_ground.py'
    text = path.read_text()
    before = '        self.init_bn_momentum()\n'
    assert text.count(before) == 1
    text = text.replace(before, before+'        self.candidate_box_refiner = None\n')
    before = '        # step Seg Prediction head\n'
    assert text.count(before) == 1
    text = text.replace(before, '        tail_geometry_query = geometry_query\n\n'+before)
    before = '        end_points[\'super_xyz_list\'] = super_xyz_list \n'
    assert text.count(before) == 1
    after = before + """
        if self.candidate_box_refiner is not None:
            raw_points = inputs['points'][:, 1:].view(inputs['batch_size'], 50000, 6)
            refined_center, refined_size = self.candidate_box_refiner(
                tail_geometry_query, raw_points, end_points['last_center'],
                end_points['last_pred_size'], end_points)
            end_points['last_center'] = refined_center
            end_points['last_pred_size'] = refined_size
"""
    text = text.replace(before, after)
    path.write_text(text)
    py_compile.compile(str(path), doraise=True)
    record = dict(parent_source=str(args.parent),model_source=str(args.output),
        changed_files=['models/pv_ground.py'],call_position='after native Mask generation',
        candidate_box_refinement=True,p2=False,
        files={str(p.relative_to(args.output)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in args.output.rglob('*') if p.is_file() and '__pycache__' not in p.parts})
    (args.output.parent/'source_port.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
