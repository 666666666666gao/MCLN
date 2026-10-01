"""Create the G/P2 shared source from the already verified D/G source."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    texts = {name: (args.parent / name).read_text() for name in
             ('models/pv_ground.py', 'models/encoder_decoder_layers.py')}
    name = 'models/encoder_decoder_layers.py'
    texts[name] = replace_once(texts[name], '        self.task_read = False',
                              '        self.task_read = False\n        self.candidate_evidence_read = None')
    texts[name] = replace_once(texts[name],
        'source_features=None, source_position=None, source_observations=None):',
        'source_features=None, source_position=None, source_observations=None,\n'
        '                source_xyz=None, candidate_centers=None, candidate_sizes=None):')
    texts[name] = replace_once(texts[name],
        '            return finish_task_queries(self, query, query2, residuals)',
        '            if self.candidate_evidence_read is not None:\n'
        '                evidence = self.candidate_evidence_read(\n'
        '                    query + query_pos, lang_feats, text_key_padding_mask,\n'
        '                    source_features, source_observations, source_xyz,\n'
        '                    candidate_centers, candidate_sizes)\n'
        '                residuals = tuple(residual + evidence for residual in residuals)\n'
        '            return finish_task_queries(self, query, query2, residuals)')
    name = 'models/pv_ground.py'
    texts[name] = replace_once(texts[name],
        "                source_observations=end_points['source_observations']\n",
        "                source_observations=end_points['source_observations'],\n"
        '                source_xyz=points_xyz, candidate_centers=base_xyz,\n'
        '                candidate_sizes=base_size\n')
    for name, text in texts.items():
        compile(text, name, 'exec')
    shutil.copytree(args.parent, args.output, ignore=shutil.ignore_patterns('__pycache__'))
    for name, text in texts.items():
        (args.output / name).write_text(text)
    record = {'parent_source': str(args.parent), 'model_source': str(args.output),
              'changed_files': list(texts), 'files': {}}
    for name in texts:
        record['files'][name] = hashlib.sha256((args.output / name).read_bytes()).hexdigest()
    (args.output.parent / 'source_port.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
