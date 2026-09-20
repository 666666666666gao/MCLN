"""Apply the task-read experiment to an isolated, verified EG source copy."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


EXPECTED = {
    'models/encoder_decoder_layers.py': '61251d06736222c4b5182fe6817cb47d32e1f93157f094a8c5aeb6f99bb25235',
    'models/eg.py': 'b65e77b9e3dd095591e37cbd88f8a6c3179b3499f4ed027569f9a33870438ee1',
    'models/modules.py': '425bf67eba989d43d7d7ef23dcf8d6a46e02a8998cfe2da9187c7d5dbcf8edf8',
}


def replace_one(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--module', type=Path, required=True)
    args = parser.parse_args()
    originals = {}
    for name, digest in EXPECTED.items():
        p = args.source / name
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, name
        originals[name] = p.read_text()
    text = originals['models/encoder_decoder_layers.py']
    text = replace_one(text, 'from .eg_attention import MultiheadAttention\n',
                       'from .eg_attention import MultiheadAttention\nfrom .eg3dvg_task_read import read_tasks\n')
    text = replace_one(text, '        # STEP 4. Cross attention in vision\n',
                       '        self.task_queries = None\n        self.semantic_query = None\n\n        # STEP 4. Cross attention in vision\n')
    start = text.index('        # step 4. Cross attend to vision\n')
    stop = text.index('        # step 6. Refine superpoint feature\n', start)
    original_read = text[start:stop]
    text = text[:start] + (
        '        if self.task_queries is not None:\n'
        '            query, self.semantic_query = read_tasks(\n'
        '                self, query, query_pos, vis_feats, spatial_feature,\n'
        '                padding_mask, lang_feats, super_features, attn_mask_list)\n'
        '        else:\n' + ''.join('    ' + line if line.strip() else line
                                    for line in original_read.splitlines(True))
    ) + text[stop:]
    changed = {'models/encoder_decoder_layers.py': text}
    text = originals['models/eg.py']
    text = replace_one(text, '            # step project\n',
        '            semantic_query = (self.decoder[i].semantic_query\n'
        '                              if self.decoder[i].task_queries is not None else query)\n'
        '            # step project\n')
    text = replace_one(text, "end_points[f'{prefix}proj_queries'] = F.normalize(\n                    self.contrastive_align_projection_image(query)",
                       "end_points[f'{prefix}proj_queries'] = F.normalize(\n                    self.contrastive_align_projection_image(semantic_query)")
    text = replace_one(text, '                prefix=prefix\n',
                       '                prefix=prefix,\n                semantic_features=semantic_query.transpose(1, 2).contiguous()\n')
    # The ordinary Python attribute is transient, never checkpoint state.
    text = replace_one(text, "            base_xyz = base_xyz.detach().clone()\n",
                       '            self.decoder[i].semantic_query = None\n            base_xyz = base_xyz.detach().clone()\n')
    changed['models/eg.py'] = text
    text = originals['models/modules.py']
    text = replace_one(text, "    def forward(self, features, base_xyz, end_points, prefix=''):",
                       "    def forward(self, features, base_xyz, end_points, prefix='', semantic_features=None):")
    text = replace_one(text, '            sem_cls_scores = self.sem_cls_scores_head(features).transpose(2, 1)',
                       '            sem_cls_scores = self.sem_cls_scores_head(\n                features if semantic_features is None else semantic_features).transpose(2, 1)')
    changed['models/modules.py'] = text
    for name, text in changed.items():
        compile(text, name, 'exec')
    for name, text in changed.items():
        (args.source / name).write_text(text)
    shutil.copyfile(str(args.module), str(args.source / 'models/eg3dvg_task_read.py'))
    report = {'before': EXPECTED, 'after': {
        name: hashlib.sha256((args.source / name).read_bytes()).hexdigest()
        for name in list(changed) + ['models/eg3dvg_task_read.py']},
        'new_parameters': 165888, 'scope': 'last-layer visual read requests and explicit task output routing'}
    (args.source.parent / 'task_read_patch.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
