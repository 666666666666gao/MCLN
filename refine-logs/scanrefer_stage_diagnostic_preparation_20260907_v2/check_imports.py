import datetime, hashlib, importlib, json, os, sys
from pathlib import Path
root = Path('/root/autodl-tmp/mcln_scanrefer_stage_diagnostic_preparation_20260907_v2')
source = Path('/root/autodl-tmp/mcln_scanrefer_local_visual_preflight_20260906_v2/model_source')
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
os.chdir(str(source))
sys.path.insert(0, str(source))
import scripts
scripts.__path__ = [str(root / 'scripts'), str(source / 'scripts')]
names = ['main_utils', 'train_dist_mod', 'models.rec_reranker', 'models.candidate_local_visual',
         'scripts.run_frozen_v99_pareto_contextual_official', 'scripts.scanrefer_joint_readout',
         'scripts.scanrefer_rec_evaluation', 'scripts.scanrefer_data_contract',
         'scripts.trace_scanrefer_readout_stages', 'scripts.scanrefer_stage_diagnostics']
loaded = {}
for name in names:
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    expected = root if name in names[5:] else source
    assert path == expected / (name.replace('.', '/') + '.py'), (name, str(path))
    loaded[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
import torch
assert not torch.cuda.is_available()
result = {'status': 'runtime_imports_pass', 'modules': loaded, 'torch': torch.__version__,
          'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
          'gpu_forwards': 0, 'data_rows_loaded': 0, 'optimizer_updates': 0, 'checkpoint_writes': 0}
with (root / 'import_receipt.json').open('x') as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
print(json.dumps(result))
