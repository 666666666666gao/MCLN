"""Create an isolated PV-Ground source with explicit CS insertion points."""

import argparse
import json
from pathlib import Path
import shutil


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--implementation', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    shutil.copytree(str(args.parent_source), str(args.output),
                    ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc'))
    shutil.copy2(str(args.implementation / 'models/cs_pvground_modules.py'),
                 str(args.output / 'models/cs_pvground_modules.py'))
    shutil.copy2(str(args.implementation / 'utils/scatter_util.py'),
                 str(args.output / 'utils/scatter_util.py'))
    path = args.output / 'models/pv_ground.py'
    text = path.read_text(encoding='utf-8')
    text = 'from models.cs_pvground_modules import (PointVoxelObservationEnhancer, ContextSupportReader, MaskSupportBoxRefiner, GeometryEvidenceReadback)\nfrom utils.scatter_util import deterministic_scatter_mean_dim0\n' + text
    text = replace_once(text, '                 self_attend=True):',
                        '                 self_attend=True, use_cs=False, use_readback=False):')
    text = replace_once(text, '        self.init_bn_momentum()', '''        assert not use_readback or use_cs
        self.cs_structure = PointVoxelObservationEnhancer(d_model) if use_cs else None
        self.cs_context_reader = ContextSupportReader(d_model) if use_cs else None
        self.cs_box_refiner = MaskSupportBoxRefiner(d_model) if use_cs else None
        self.cs_geometry_readback = GeometryEvidenceReadback(d_model) if use_readback else None
        self.init_bn_momentum()''')
    text = replace_once(text, "        points_features = end_points['seed_features']", '''        points_features = end_points['seed_features']
        if self.cs_structure is not None:
            points_features = self.cs_structure.enhance_seeds(
                end_points['source_features'], points_features)''')
    text = replace_once(text, 'scatter_mean(source_xzy[bs], superpoint[bs], dim=0)',
                        'deterministic_scatter_mean_dim0(source_xzy[bs], superpoint[bs].long())')
    text = replace_once(text, '        # STEP 5. Query Points Generation', '''        if self.cs_structure is not None:
            super_features = self.cs_structure.enhance_superpoints(
                super_features, super_xyz_list,
                inputs['points'][:, 1:].view(inputs['batch_size'], -1, 6),
                superpoint, end_points['seed_inds'], points_xyz,
                end_points['source_features'], points_features)

        # STEP 5. Query Points Generation''')
    text = replace_once(text, '            # step project\n', '''            if self.cs_context_reader is not None and i >= self.num_decoder_layers - 2:
                query = self.cs_context_reader(
                    query, points_features.transpose(1, 2), super_features,
                    super_xyz_list, base_xyz)
            # step project
''')
    text = replace_once(text, '                prefix=prefix\n            )\n            base_xyz = base_xyz.detach().clone()', '''                prefix=prefix,
                defer_semantic=self.cs_geometry_readback is not None and prefix == 'last_',
            )
            if self.cs_box_refiner is not None and i >= self.num_decoder_layers - 2:
                mask_query = self.x_query(query.transpose(1, 2)).transpose(1, 2)
                base_xyz, base_size, evidence = self.cs_box_refiner(
                    query, mask_query, super_features, super_xyz_list,
                    base_xyz, base_size, return_evidence=True)
                end_points[prefix + 'center'] = base_xyz
                end_points[prefix + 'pred_size'] = base_size
                if self.cs_geometry_readback is not None and prefix == 'last_':
                    final_query = self.cs_geometry_readback(query, evidence)
                    self.prediction_heads[i].write_semantic_scores(
                        final_query.transpose(1, 2).contiguous(), end_points, prefix)
            base_xyz = base_xyz.detach().clone()''')
    path.write_text(text, encoding='utf-8')
    path = args.output / 'models/modules.py'
    text = path.read_text(encoding='utf-8')
    text = replace_once(text, "    def forward(self, features, base_xyz, end_points, prefix=''):",
                        "    def forward(self, features, base_xyz, end_points, prefix='', defer_semantic=False):")
    text = replace_once(text, '''        if self.compute_sem_scores:
            sem_cls_scores = self.sem_cls_scores_head(features).transpose(2, 1)''', '''        if self.compute_sem_scores and not defer_semantic:
            sem_cls_scores = self.sem_cls_scores_head(features).transpose(2, 1)''')
    text = replace_once(text, '''        if self.compute_sem_scores:
            end_points[f'{prefix}sem_cls_scores'] = sem_cls_scores''', '''        if self.compute_sem_scores and not defer_semantic:
            end_points[f'{prefix}sem_cls_scores'] = sem_cls_scores''')
    text = replace_once(text, '        return center, pred_size', '''        return center, pred_size

    def write_semantic_scores(self, features, end_points, prefix):
        assert self.compute_sem_scores
        end_points[prefix + 'sem_cls_scores'] = self.sem_cls_scores_head(features).transpose(2, 1)''')
    path.write_text(text, encoding='utf-8')
    (args.output / 'cs_port.json').write_text(json.dumps({
        'parent_source': str(args.parent_source),
        'implementation': str(args.implementation),
        'modules': ['M1_point_voxel', 'M2', 'M3', 'R'],
        'shared_superpoint_centers': 'deterministic_scatter_mean_dim0',
        'native_heads_retained': True,
        'quality_loss': False, 'G': False, 'teacher': False,
        'inference_sidechains': False,
    }, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
