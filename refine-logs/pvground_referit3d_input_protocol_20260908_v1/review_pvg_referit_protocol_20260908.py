import ast
import datetime
import hashlib
import json
from pathlib import Path
import shlex

repo = Path('C:/Users/gb/.codex_mcln_g0_20260905')
archive = repo / 'refine-logs/pvground_referit3d_input_protocol_20260908_v1'
receipt = json.loads((archive / 'receipt.json').read_bytes())
execution = json.loads((archive / 'execution.json').read_bytes())
source = json.loads((archive / 'source_receipt.json').read_bytes())
assert hashlib.sha256((archive / 'receipt.json').read_bytes()).hexdigest() == execution['receipt_sha256']
assert hashlib.sha256((repo / 'scripts/audit_pvground_referit3d_inputs.py').read_bytes()).hexdigest() == execution['audit_source_sha256']
assert (archive / 'audit.py').read_bytes() == (repo / 'scripts/audit_pvground_referit3d_inputs.py').read_bytes()
assert (archive / 'audit.exit').read_text().strip() == '0'
assert not (archive / 'stderr.txt').read_bytes()
for name, metadata in source['files'].items():
    assert hashlib.sha256((archive / name).read_bytes()).hexdigest() == metadata['sha256'], name
    if name.endswith('.py'):
        ast.parse((archive / name).read_bytes())
physical = {}
for dataset, data in receipt['datasets'].items():
    sets = {split: {scene.split('_')[0] for scene in values['scenes']} for split, values in data['splits'].items()}
    assert not sets['train'] & sets['test']
    physical[dataset] = {split: len(values) for split, values in sets.items()}
    for split, values in data['splits'].items():
        assert values['expression_scenes'] == len(set(values['scenes'])) == values['predicted_class_scene_coverage']
        declared = ast.literal_eval((archive / 'native/data/meta_data' / (dataset + '_' + split + '_scans.txt')).read_text())
        assert set(values['scenes']) <= set(declared)
recipes = {}
for path in sorted((archive / 'upstream/scripts').glob('*.sh')):
    shell_text = path.read_text().replace('\\\n', ' ').rstrip().removesuffix('\\')
    tokens = shlex.split(shell_text, comments=True)
    tokens = tokens[tokens.index('train_dist_mod.py') + 1:]
    parsed = {}
    index = 0
    while index < len(tokens):
        key = tokens[index]
        assert key.startswith('--'), key
        index += 1
        if '=' in key:
            key, value = key.split('=', 1)
            parsed[key] = [value]
        else:
            value = []
            while index < len(tokens) and not tokens[index].startswith('--'):
                value.append(tokens[index])
                index += 1
            parsed[key] = value
    for key in ['--butd_cls', '--self_attend', '--joint_det', '--use_color', '--use_soft_token_loss', '--use_contrastive_align']:
        assert key in parsed and parsed[key] == []
    assert parsed['--num_decoder_layers'] == ['6']
    recipes[path.name] = parsed
models = json.loads((repo / 'refine-logs/pvground_pretrained_resources_20260908_v1/huggingface_metadata.json').read_bytes())
weights = {item['rfilename']: {'bytes': item['size'], 'sha256': item['lfs']['sha256']}
           for item in models['siblings'] if item['rfilename'] in ['PV-Ground_NR3D.pth', 'PV-Ground_SR3D.pth']}
result = {'status': 'pass', 'time_cst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
          'source_files_verified': len(source['files']), 'receipt_sha256': execution['receipt_sha256'],
          'review_scope': 'local provenance, split-scene recount and exact shell flags; no independent full-CSV rerun',
          'physical_scene_counts': physical, 'physical_train_test_overlap': {'nr3d': 0, 'sr3d': 0},
          'upstream_recipes': recipes, 'published_model_revision': models['sha'], 'not_downloaded_weights': weights,
          'formal_evaluation_rows': 0, 'optimizer_steps': 0}
(archive / 'local_review.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': result['status'], 'physical_scene_counts': physical, 'recipe_checkpoint_paths': {name: data['--checkpoint_path'] for name, data in recipes.items()}}))
