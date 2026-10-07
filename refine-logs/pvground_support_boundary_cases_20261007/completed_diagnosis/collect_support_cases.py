"""Read retained-best actual support for191 error cases; no training."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

from support_evidence import analyze_case


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


root = Path(__file__).resolve().parent
started = time.perf_counter()
spec = json.loads((root / 'diagnostic_spec.json').read_bytes())
packet = json.loads((root / 'case_manifest.json').read_bytes())
assert spec['optimizer_steps'] == spec['weights_created'] == 0 and spec['seed'] == 2027
runtime = Path(spec['runtime'])
env = json.loads((runtime / 'env_spec.json').read_bytes())
assert hashlib.sha256(json.dumps(env, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == spec['env_spec_sha256']
manifest = json.loads(Path(spec['input_manifest']).read_bytes())
dataset_source = Path(manifest['model_source'])
port = json.loads(Path(spec['source_port']).read_bytes())
assert sha(spec['source_port']) == spec['source_port_sha256']
assert sha(dataset_source / 'appearance_source_manifest.json') == manifest['source_manifest_sha256']
for name, digest in json.loads((dataset_source / 'appearance_source_manifest.json').read_bytes())['files'].items():
    assert sha(dataset_source / name) == digest
for name, digest in port['files'].items():
    assert sha(Path(spec['model_source']) / name) == digest
for name, digest in spec['runner_files'].items():
    assert sha(Path(spec['helper_root']) / name) == digest
assert sha(root / 'mask_reference.py') == spec['mask_reference_sha256']
assert sha(root / 'selected_mask_reference_factory.py') == spec['selected_factory_sha256']
weights = {spec['selected_terminal']: spec['selected_terminal_sha256'], spec['base_terminal']: spec['base_terminal_sha256'],
           env['weight_dirs']['scanrefer']['path']: spec['checkpoint_sha256']}
assert all(sha(path) == digest for path, digest in weights.items())

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

def reset_rng():
    random.seed(2027); np.random.seed(2027); torch.manual_seed(2027); torch.cuda.manual_seed_all(2027)

reset_rng()
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
os.chdir(str(dataset_source)); sys.path.insert(0, str(dataset_source))
from src.joint_det_dataset import Joint3DDataset
from scripts.scanrefer_data_contract import verify_scanrefer_superpoints
assert Path(sys.modules['src.joint_det_dataset'].__file__).resolve() == dataset_source / 'src/joint_det_dataset.py'
os.chdir(spec['model_source'])
sys.path.insert(0, spec['helper_root']); sys.path.insert(0, str(root)); sys.path.insert(0, spec['model_source'])
from prepare_data import DataProcessor
from pcdet.config import cfg, cfg_from_yaml_file
from selected_mask_reference_factory import build_selected_mask_reference_model
from readback_preflight_checks import observed_readback_forward
from native_root_bbs import native_root_bbs

import_names = ('src.joint_det_dataset', 'models.pv_ground', 'prepare_data', 'selected_mask_reference_factory', 'mask_reference', 'native_root_bbs')
imported = {name: str(Path(sys.modules[name].__file__).resolve()) for name in import_names}
assert Path(imported['models.pv_ground']) == Path(spec['model_source']) / 'models/pv_ground.py'
assert Path(imported['prepare_data']) == Path(spec['model_source']) / 'prepare_data.py'
assert Path(imported['selected_mask_reference_factory']) == root / 'selected_mask_reference_factory.py'
assert Path(imported['mask_reference']) == root / 'mask_reference.py'
assert Path(imported['native_root_bbs']) == Path(spec['helper_root']) / 'native_root_bbs.py'
(root / 'imports.json').write_text(json.dumps(dict(files=imported, sha256={name: sha(path) for name, path in imported.items()}), indent=2) + '\n')

cfg_from_yaml_file(str(runtime / 'PV-Ground/wandb_config.yaml'), cfg)
selected = torch.load(spec['selected_terminal'], map_location='cpu')
official = torch.load(env['weight_dirs']['scanrefer']['path'], map_location='cpu')
g = torch.load(spec['base_terminal'], map_location='cpu')
assert selected['step'] == 0 and selected['reference_mode'] == 'fused_mask'
model, config, load = build_selected_mask_reference_model(cfg, official, g, selected, manifest['data_root'])
assert load['full_state_tensors'] == 1304
assert torch.count_nonzero(model.candidate_box_refiner.output.weight) == 0
assert torch.count_nonzero(model.candidate_box_refiner.output.bias) == 0
(root / 'load.json').write_text(json.dumps(load, indent=2) + '\n')
for parameter in model.parameters():
    parameter.requires_grad_(False)
model.cuda().eval()
assert not any(part.training for part in model.modules())
processor = DataProcessor(cfg.DATA_PROCESSOR, np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE), False, 6)
verify_scanrefer_superpoints(manifest['data_root'], 'val', manifest['superpoint_files']['val'])

class FormalDataset(Joint3DDataset):
    def _scene_graph_parse(self, annos):
        assert len(annos) == 9508
        for index, row in enumerate(annos):
            row['_local_training_id'] = index
        super()._scene_graph_parse(annos)

    def __getitem__(self, index):
        result = super().__getitem__(index)
        result['local_training_id'] = self.annos[index]['_local_training_id']
        assert np.isin(result['gt_masks'], [0, 1]).all()
        result['gt_masks'] = result['gt_masks'].astype(np.bool_)
        return result

os.chdir(str(dataset_source))
dataset = FormalDataset(dataset_dict={'scanrefer': 1}, test_dataset='scanrefer', split='val',
    data_path=manifest['data_root'], use_color=True, use_height=False, use_multiview=False,
    detect_intermediate=True, butd=True, butd_cls=False, butd_gt=False, augment_det=False, skip_missing_superpoints=True)
assert len(dataset) == 9508
dataset.augment = dataset.augment_det = False
ids = packet['forward_row_ids']
loader = DataLoader(Subset(dataset, ids), batch_size=8, shuffle=False, num_workers=2,
    generator=torch.Generator().manual_seed(2027), pin_memory=True, drop_last=False)
reset_rng()
cases = {item['cached']['row_id']: item for item in packet['diagnostic_rows']}
directory = root / 'arrays'; directory.mkdir()
seen = []
forwards = 0
torch.cuda.reset_peak_memory_stats()
with (root / 'rows.jsonl').open('x') as stream, torch.no_grad():
    for batch_cpu in loader:
        voxel_rows = [processor.forward(dict(points=pc.numpy().copy(), use_lead_xyz=True)) for pc in batch_cpu['point_clouds']]
        voxels = processor.collate_batch(voxel_rows)
        size = len(batch_cpu['utterances'])
        assert np.array_equal(voxels['points'][:, 1:].reshape(size, 50000, 6), batch_cpu['point_clouds'].numpy())
        batch = {key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value for key, value in batch_cpu.items()}
        inputs = {key: torch.from_numpy(voxels[key]).float().cuda() for key in ('points', 'voxels', 'voxel_coords', 'voxel_num_points')}
        inputs.update(batch_size=size, text=batch['utterances'], superpoint=batch['superpoint'], train=False,
            det_boxes=batch['all_detected_boxes'], det_bbox_label_mask=batch['all_detected_bbox_label_mask'], det_class_ids=batch['all_detected_class_ids'])
        predictions, call = observed_readback_forward(model, inputs)
        assert call['final_semantic_head_calls'] == 1
        assert predictions['last_sem_cls_scores'].shape[1] == 256
        assert torch.equal(predictions['last_center'], predictions['geometry_reference_center'])
        assert torch.equal(predictions['last_pred_size'], predictions['geometry_reference_size'].clamp(min=1e-6))
        scores = native_root_bbs(predictions['last_sem_cls_scores'], batch)
        forwards += 1
        for bid in range(size):
            row_id = int(batch_cpu['local_training_id'][bid])
            if row_id not in cases:
                continue
            case = cases[row_id]; cached = case['cached']
            cloud = batch_cpu['point_clouds'][bid].numpy()
            assert hashlib.sha256(cloud.tobytes()).hexdigest() == case['point_sha256']
            truth = torch.cat((batch['center_label'][bid, 0, :3], batch['size_gts'][bid, 0])).cpu().numpy()
            assert truth.tolist() == cached['root_gt']
            assert batch['scan_ids'][bid] == cached['scan_id'] and int(batch['target_id'][bid]) == cached['target_id']
            queries = [int(scores[bid].argsort(descending=True)[0]), cached['query']]
            text = predictions['last_pred_masks'][bid][0]
            own = predictions['sp_last_pred_masks'][bid]
            assert torch.equal(text[queries[0]], text[queries[1]])
            text = text[queries[0]]
            query_logits = own[queries]
            alpha = predictions['adaptive_weights'][bid]
            assert alpha.ndim == 0
            fused = alpha * text[None] + (1 - alpha) * query_logits
            sp = predictions['superpoints'][bid]
            assert torch.equal(sp, batch['superpoint'][bid])
            center = predictions['geometry_reference_center'][bid, queries]
            widths = predictions['geometry_reference_size'][bid, queries].clamp(min=1e-6)
            coarse = torch.cat((predictions['native_coarse_center'][bid, queries], predictions['native_coarse_size'][bid, queries]), -1)
            arrays = dict(xyz=cloud[:, :3], superpoint=sp.cpu().numpy().astype(np.int32),
                target=batch_cpu['gt_masks'][bid, 0].numpy(), queries=np.asarray(queries, dtype=np.int32), root_gt=truth,
                text_logits=text.cpu().numpy(), query_logits=query_logits.cpu().numpy(), fused_logits=fused.cpu().numpy(),
                fused_active=fused.sigmoid().gt(.5).cpu().numpy(), alpha=alpha.cpu().numpy(),
                reference_boxes=torch.cat((center, widths), -1).cpu().numpy(),
                reference_valid=predictions['mask_reference_valid'][bid, queries].cpu().numpy(),
                coarse_boxes=coarse.cpu().numpy(), all256_scores=scores[bid].cpu().numpy())
            path = directory / ('row_%05d.npz' % row_id)
            np.savez_compressed(path, **arrays)
            evidence = analyze_case(arrays)
            record = dict(row_id=row_id, group=case['group'], scan_id=cached['scan_id'], target_id=cached['target_id'],
                point_sha256=case['point_sha256'], deployed_query=queries[0], historical_query=queries[1],
                selection_changed=queries[0] != queries[1], arrays=path.name, bytes=path.stat().st_size, sha256=sha(path),
                native_semantic_head_calls=call['final_semantic_head_calls'], **evidence)
            stream.write(json.dumps(record) + '\n'); seen.append(row_id)
        stream.flush()
        print('SUPPORT_CASE_PROGRESS ' + json.dumps(dict(forwards=forwards, total=spec['forward_batches'], cases=len(seen), seconds=time.perf_counter()-started)), flush=True)
        del inputs, predictions, batch, batch_cpu
assert set(seen) == set(cases) and len(seen) == 191 and forwards == spec['forward_batches']
assert all(sha(path) == digest for path, digest in weights.items())
assert not any(part.training for part in model.modules()) and not any(parameter.requires_grad for parameter in model.parameters())
receipt = dict(status='PASS_ACTUAL_READ_ONLY_DIAGNOSIS', time_cst=datetime.datetime.now().astimezone().isoformat(),
    cases=191, groups=packet['groups'], forward_batches=forwards, original_batch_context_rows=len(ids),
    all256_candidates_retained=True, optimizer_updates=0, weights_created=0, protected_weight_chain_exact=True,
    peak_allocated_bytes=torch.cuda.max_memory_allocated(), elapsed_seconds=time.perf_counter()-started,
    rows_sha256=sha(root / 'rows.jsonl'), interpretation='Targeted development error cases only, not new accuracy or training evidence')
(root / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
