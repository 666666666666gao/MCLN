"""Copy the sealed G/D source and add final-layer P3 wiring only."""
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
    old = '        self.init_bn_momentum()\n'
    assert text.count(old) == 1
    text = text.replace(old, old+'        self.candidate_box_refiner = None\n')
    old = '            base_xyz = base_xyz.detach().clone()\n'
    assert text.count(old) == 1
    new = """            if i == self.num_decoder_layers - 1 and self.candidate_box_refiner is not None:
                raw_points = inputs['points'][:, 1:].view(inputs['batch_size'], 50000, 6)
                base_xyz, base_size = self.candidate_box_refiner(
                    geometry_query, raw_points, base_xyz, base_size, end_points)
                end_points[prefix + 'center'] = base_xyz
                end_points[prefix + 'pred_size'] = base_size
""" + old
    text = text.replace(old, new)
    path.write_text(text)
    py_compile.compile(str(path), doraise=True)
    files = {str(p.relative_to(args.output)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in args.output.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    record = {'parent_source':str(args.parent), 'model_source':str(args.output),
              'changed_files':['models/pv_ground.py'], 'candidate_box_refinement':True,
              'p2':False, 'files':files}
    (args.output.parent/'source_port.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k != 'files'}))


if __name__ == '__main__':
    main()
