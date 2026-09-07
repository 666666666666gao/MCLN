"""Check normal native imports from the assembled tree, with no package overlay."""
import json
import os
from pathlib import Path
import sys

source = Path(sys.argv[1]).resolve()
assert Path.cwd() == source
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
import torch
import main_utils
import train_dist_mod
import models.losses
import models.rec_mask_geometry
import scripts.native_mask_geometry_supervision
import scripts.native_teacher_box_transfer
import scripts.prototype_probability_geometry

torch.set_num_threads(1)
assert not torch.cuda.is_available()
modules = [main_utils, train_dist_mod, models.losses, models.rec_mask_geometry,
           scripts.native_mask_geometry_supervision, scripts.native_teacher_box_transfer,
           scripts.prototype_probability_geometry]
origins = {module.__name__: str(Path(module.__file__).resolve().relative_to(source))
           for module in modules}
assert list(models.__path__) == [str(source / 'models')]
assert {str(Path(path).resolve()) for path in scripts.__path__} == {str(source / 'scripts')}
contracts = []
for dataset in ['nr3d', 'sr3d']:
    sys.argv = ['native-source-check', '--dataset', dataset, '--test_dataset', dataset,
                '--model', 'MCLN', '--num_decoder_layers', '6', '--num_target', '256',
                '--use_soft_token_loss', '--use_contrastive_align', '--local_rank', '0',
                '--native_mask_geometry_supervision']
    args = main_utils.parse_option()
    assert args.native_mask_geometry_supervision and not args.use_candidate_local_visual
    criterion, set_criterion = train_dist_mod.TrainTester.get_criterion(args)
    assert callable(criterion) and isinstance(set_criterion, models.losses.SetCriterion)
    contracts.append({'dataset': dataset, 'criterion_type': type(set_criterion).__name__,
                      'native_auxiliary_enabled': True, 'candidate_local_visual_enabled': False})
import pytest
status = pytest.main(['-q', '-p', 'no:cacheprovider',
    str(source / 'tests/test_native_mask_geometry_training.py')])
print(json.dumps({'normal_import_origins': origins, 'contracts': contracts,
                  'pytest_exit': int(status), 'no_package_path_rewriting': True}), flush=True)
raise SystemExit(status)
