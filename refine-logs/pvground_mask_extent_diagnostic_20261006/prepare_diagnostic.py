"""Reuse the verified PV factory/data protocol for one read-only extent probe."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
previous = root.parent / 'pvground_support_reference_20261005'
original = (previous / 'run_geometry_fit.py').read_text(encoding='utf-8')
prefix = original[:original.index('def main():')]
setup = original[original.index('    begin = time.perf_counter()'):original.index('    trainable = {name: parameter')]
setup = setup.replace("choices=['preflight', 'train', 'formal']", "choices=['preflight', 'formal']")
setup = setup.replace("    output = Path(spec['root'])", "    output = Path(spec['root']) / args.mode\n    output.mkdir()")
setup = setup.replace("    assert sha(output.parent/'support_reference.py')==spec['support_reference_sha256']\n", '')
setup = setup.replace('    from support_reference import install_support_reference, reference_localization_loss\n', '')
for line in ('    from pvground_semantic_assignment import semantic_assignment_correction\n',
             '    from pvground_boundary_box_refiner import distribution_loss, BoundaryBoxRefiner\n',
             '    from query_supported_geometry import query_supported_geometry_loss\n',
             '    from whole_model_preflight_checks import optimizer_restore_exact\n'):
    setup = setup.replace(line, '')
setup = setup.replace("    if spec['reference_enabled']:\n        install_support_reference(model)\n", "    assert not spec['reference_enabled']\n")
setup = setup.replace('    for parameter in geometry_head.parameters():\n        parameter.requires_grad_(True)',
                      '    assert not any(parameter.requires_grad for parameter in model.parameters())')
setup = setup.replace('frozen_geometry_provider=False,frozen_parent_and_R=True,geometry_head_only=True,',
                      'frozen_geometry_provider=True,frozen_parent_and_R=True,geometry_head_only=False,')
setup = setup.replace('fresh_geometry_optimizer_required=True)',
                      'fresh_geometry_optimizer_required=False,read_only_diagnostic=True,optimizer_created=False)')
dataset = original[original.index('        class FormalDataset'):original.index('    else:\n        dataset = FitDataset')]
dataset = '\n'.join(line[4:] if line.startswith('    ') else line for line in dataset.splitlines()) + '\n'
prepare = original[original.index('    def prepare(batch, mode):'):original.index('    def native_loss(predictions, batch):')]
native_loss = original[original.index('    def native_loss(predictions, batch):'):original.index('    def box_iou(boxes, truth):')]
gpu_iou = original[original.index('    def box_iou(boxes, truth):'):original.index('    @torch.no_grad()\n    def evaluate(stage):')]
evaluate = '''    from extent_evidence import member_geometry, exact_box, quantile_box, box_iou as cpu_iou, QUANTILE
    from native_root_bbs import native_root_bbs
    assert spec['read_only_diagnostic'] and spec['quantile'] == QUANTILE
    assert sha(Path(spec['root']) / 'extent_evidence.py') == spec['extent_evidence_sha256']
    evaluator_path = runtime / 'PV-Ground/src/grounding_evaluator.py'
    assert sha(evaluator_path) == spec['native_evaluator_sha256']
    evaluator_spec = importlib.util.spec_from_file_location('pvground_official_evaluator', str(evaluator_path))
    evaluator_module = importlib.util.module_from_spec(evaluator_spec)
    evaluator_spec.loader.exec_module(evaluator_module)
    evaluator = evaluator_module.GroundingEvaluator(only_root=True, thresholds=[.25, .5],
        topks=[1, 5, 10], prefixes=['last_'], filter_non_gt_boxes=False, model='PVGround')
    assert sha(spec['parent_formal_rows']) == spec['parent_formal_rows_sha256']
    parent_rows = [json.loads(line) for line in Path(spec['parent_formal_rows']).read_text().splitlines()]
    assert len(parent_rows) == 9508
    length = 8 if args.mode == 'preflight' else 9508
    dataset.augment = False
    dataset.augment_det = False
    model.eval()
    reset_rng()
    loader = DataLoader(Subset(dataset, list(range(length))), batch_size=8, shuffle=False,
        num_workers=2, generator=torch.Generator().manual_seed(spec['seed']), pin_memory=True, drop_last=False)
    rows = []
    preflight_formal_batch_bytes = None
    inference_start = time.perf_counter()
    with torch.no_grad(), (output / 'rows.jsonl').open('w') as stream:
        for batch_index, batch_cpu in enumerate(loader):
            inputs, batch = prepare(batch_cpu, 'eval')
            predictions, calls = observed_readback_forward(model, inputs)
            assert torch.equal(predictions['last_semantic_query_before_readback'],
                               predictions['last_semantic_query_after_readback'])
            scores = native_root_bbs(predictions['last_sem_cls_scores'], batch)
            boxes = torch.cat([predictions['last_center'], predictions['last_pred_size']], -1)
            assert (boxes[..., 3:] > 0).all()
            _, predictions = native_loss(predictions, batch)
            for key in predictions:
                if 'pred_size' in key:
                    predictions[key] = predictions[key].clamp(min=1e-6)
            evaluator.evaluate(predictions, 'last_')
            truths = torch.cat([batch['center_label'][:, 0, :3], batch['size_gts'][:, 0]], -1)
            evidence = {}
            for bid in range(len(batch['utterances'])):
                row_id = int(batch['local_training_id'][bid])
                assert row_id == len(rows)
                query = int(scores[bid].argsort(descending=True)[0])
                points = batch['point_clouds'][bid].cpu().numpy()
                point_sha = hashlib.sha256(points.tobytes()).hexdigest()
                root_box = truths[bid].cpu().numpy()
                historic = parent_rows[row_id]
                assert historic['row_id'] == row_id and historic['scan_id'] == batch['scan_ids'][bid]
                assert historic['target_id'] == int(batch['target_id'][bid])
                assert historic['point_sha256'] == point_sha
                assert np.array_equal(np.asarray(historic['root_box']), root_box)
                superpoint = predictions['superpoints'][bid].cpu().numpy()
                target = batch['gt_masks'][bid, 0].bool().cpu().numpy()
                ids, inverse, count, low, high, target_count = member_geometry(points[:, :3], superpoint, target)
                text = predictions['last_pred_masks'][bid][0, query]
                own = predictions['sp_last_pred_masks'][bid][query]
                alpha = predictions['adaptive_weights'][bid]
                fused = alpha * text + (1 - alpha) * own
                assert text.shape == own.shape == fused.shape
                own_active = (own.sigmoid() > .5).cpu().numpy()[ids]
                active = (fused.sigmoid() > .5).cpu().numpy()[ids]
                point_active = active[inverse]
                exact = exact_box(active, low, high)
                trimmed, samples, ranks = quantile_box(points[:, :3], point_active)
                intersection = int(target_count[active].sum())
                foreground_count = int(count[active].sum())
                assert foreground_count == int(point_active.sum())
                mask_iou = intersection / (foreground_count + int(target.sum()) - intersection)
                own_intersection = int(target_count[own_active].sum())
                own_count = int(count[own_active].sum())
                own_iou = own_intersection / (own_count + int(target.sum()) - own_intersection)
                learned = boxes[bid, query].cpu().numpy()
                native_iou = float(box_iou(boxes[bid], truths[bid])[query])
                record = dict(row_id=row_id, scan_id=batch['scan_ids'][bid], target_id=int(batch['target_id'][bid]),
                    query=query, root_box=root_box.tolist(), point_sha256=point_sha,
                    learned_box=learned.tolist(), learned_iou=native_iou,
                    exact_box=None if exact is None else exact.tolist(), exact_iou=cpu_iou(exact, root_box),
                    quantile_box=None if trimmed is None else trimmed.tolist(), quantile_iou=cpu_iou(trimmed, root_box),
                    foreground_count=foreground_count, fused_mask_iou=mask_iou, own_mask_iou=own_iou,
                    historic_query=historic['bbs']['query'], historic_learned_iou=historic['bbs']['iou'],
                    native_head_calls=calls['final_semantic_head_calls'], diagnostic_head_replay_calls=0,
                    same_forward_geometry_exact=calls['fixed_geometry'])
                rows.append(record)
                stream.write(json.dumps(record) + '\\n')
                key = 'r' + str(row_id) + '__'
                arrays = dict(ids=ids, count=count, lower=low.astype(points.dtype), upper=high.astype(points.dtype), target_count=target_count,
                    own_active=own_active, fused_active=active, own_logits=own.cpu().numpy()[ids],
                    text_logits=text.cpu().numpy()[ids], fused_logits=fused.cpu().numpy()[ids],
                    quantile_order_values=samples, quantile_order_ranks=ranks,
                    alpha=alpha.cpu().numpy())
                if args.mode == 'preflight':
                    arrays.update(point_clouds=points, superpoint=superpoint, gt_point_mask=target)
                evidence.update({key + name: value for name, value in arrays.items()})
            np.savez_compressed(str(output / ('batch_%04d.npz' % batch_index)), **evidence)
            if args.mode == 'preflight':
                formal_payload = {name:value for name,value in evidence.items()
                    if not name.endswith(('__point_clouds','__superpoint','__gt_point_mask'))}
                buffer = io.BytesIO()
                np.savez_compressed(buffer, **formal_payload)
                preflight_formal_batch_bytes = buffer.tell()
            if len(rows) % 512 < 8:
                stream.flush()
                print('EXTENT_PROGRESS ' + json.dumps(dict(rows=len(rows), total=length,
                    seconds=time.perf_counter()-inference_start)), flush=True)
            del inputs, batch, predictions, evidence
    assert len(rows) == length
    counts = {name: {str(t): sum(row[name + '_iou'] > t for row in rows) for t in (.25, .5)}
              for name in ('learned', 'exact', 'quantile')}
    for threshold in (.25, .5):
        assert counts['learned'][str(threshold)] == evaluator.dets[('last_', threshold, 1, 'bbs')]
    assert abs(sum(row['fused_mask_iou'] for row in rows) - float(evaluator.dets['mask_pos'])) < 1e-3
    assert all(torch.equal(value.detach().cpu(), initial[name]) for name, value in model.state_dict().items())
    receipt = dict(status='pass', mode=args.mode, rows=length, hits=counts,
        inference_seconds=time.perf_counter()-inference_start, total_seconds=time.perf_counter()-begin,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        preflight_formal_batch_bytes=preflight_formal_batch_bytes,
        evidence_bytes_written=sum(path.stat().st_size for path in output.glob('batch_*.npz')),
        model_states_unchanged=True, optimizer_created=False, optimizer_updates=0, new_weights=0,
        all256_candidates_retained=True, native_head_calls_per_batch=1, scoring_or_box_outputs_overwritten=False,
        native_mask_hits50=sum(row['fused_mask_iou'] > .5 for row in rows),
        history_query_changes=sum(row['query'] != row['historic_query'] for row in rows),
        history_hit50_changes=sum((row['learned_iou'] > .5) != (row['historic_learned_iou'] > .5) for row in rows),
        invalid_exact=sum(row['exact_box'] is None for row in rows),
        invalid_quantile=sum(row['quantile_box'] is None for row in rows),
        quantile=QUANTILE, quantile_precision='NumPy float64 linear interpolation from actual float32 XYZ',
        geometry_terminal=spec['geometry_terminal'], geometry_terminal_sha256=spec['geometry_terminal_sha256'],
        spec_sha256=sha(args.spec), rows_sha256=sha(output / 'rows.jsonl'),
        time_cst=datetime.datetime.now().astimezone().isoformat())
    write_json(output / 'receipt.json', receipt)
    print('EXTENT_COMPLETE ' + json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
'''
contract = "    verify_scanrefer_superpoints(manifest['data_root'], 'val', manifest['superpoint_files']['val'])\n"
script = prefix + 'def main():\n' + setup + contract + '    os.chdir(str(dataset_source))\n' + dataset + prepare + native_loss + gpu_iou + evaluate
ast.parse(script)
(root / 'run_extent_diagnostic.py').write_text(script, encoding='utf-8', newline='\n')
spec = json.loads((previous / 'control_spec.json').read_bytes())
spec.update(root='/root/autodl-tmp/pvground_mask_extent_diagnostic_20261006', read_only_diagnostic=True,
            quantile=.005, parent_formal_rows='/root/autodl-tmp/pvground_auxiliary_target_20261005/control/formal/rows.jsonl',
            parent_formal_rows_sha256='ec73b6acb9691d6c591136c5e5a34e80a8d7d716bedef714ca960bbbd324716c',
            extent_evidence_sha256=hashlib.sha256((root / 'extent_evidence.py').read_bytes()).hexdigest())
(root / 'spec.json').write_text(json.dumps(spec, indent=2) + '\n', encoding='utf-8')
for name in ('EXPERIMENT_PLAN', 'EXPERIMENT_TRACKER'):
    (root / (name + '.md')).write_bytes((root / (name + '_20261006.md')).read_bytes())
(root / 'research_contract.md').write_text('本阶段只有冻结4511同Query空间范围诊断；不能声称方法增益、V99等价或跨基准有效。正式目标仍ScanRefer严格4754及随后公平独立Nr/Sr。\n', encoding='utf-8')
print(json.dumps({'status': 'DIAGNOSTIC_SOURCE_PREPARED_NOT_EXECUTED', 'runner_bytes': len(script.encode()),
                  'source_template_sha256': hashlib.sha256(original.encode()).hexdigest()}))
