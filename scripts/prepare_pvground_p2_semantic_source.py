"""Create a semantic-only P2 routing control from the sealed joint P2 source."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    name = 'models/encoder_decoder_layers.py'
    text = (args.parent/name).read_text()
    old = '                residuals = tuple(residual + evidence for residual in residuals)'
    new = '                residuals = (residuals[0] + evidence, residuals[1])'
    assert text.count(old) == 1
    text = text.replace(old, new)
    compile(text, name, 'exec')
    shutil.copytree(args.parent, args.output, ignore=shutil.ignore_patterns('__pycache__'))
    (args.output/name).write_text(text)
    names = ['models/pv_ground.py', name]
    record = {'parent_source':str(args.parent),'model_source':str(args.output),
              'changed_files':[name],'p2_routing':'semantic_only',
              'files':{n:hashlib.sha256((args.output/n).read_bytes()).hexdigest() for n in names}}
    (args.output.parent/'source_port.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
