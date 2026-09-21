"""Real Nr3D fit-only, zero-update witness for native final-token replacement."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-manifest', type=Path, required=True)
    parser.add_argument('--module', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    options = parser.parse_args()
    manifest = json.loads(options.input_manifest.read_text())
    source = Path(manifest['model_source'])
    spec = importlib.util.spec_from_file_location('nr_assignment', str(options.module))
    assignment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(assignment)
    os.chdir(str(source))
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from main_utils import parse_option
    from train_dist_mod import TrainTester
    from src.joint_det_dataset import Joint3DDataset
    random.seed(2027); np.random.seed(2027)
    torch.manual_seed(2027); torch.cuda.manual_seed_all(2027)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    payload = torch.load(manifest['checkpoint'], map_location='cpu')
    assert payload['evaluation_only'] and 'optimizer' not in payload
    sys.argv = [sys.argv[0]]
    args = parse_option()
    vars(args).update(vars(payload['config']))
    assert args.num_decoder_layers == 6 and args.butd_cls
    assert args.use_contrastive_align and args.use_soft_token_loss
    model = TrainTester.get_model(args).cuda().eval()
    state = {k[7:]: v for k, v in payload['model'].items()}
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    del payload
    criterion, set_criterion = TrainTester.get_criterion(args)
    fit_ids = manifest['row_ids']['fit'][:16]

    class FitRows(Joint3DDataset):
        def _scene_graph_parse(self, annos):
            annos[:] = [annos[i] for i in fit_ids]
            super()._scene_graph_parse(annos)

    print('NR_ASSIGNMENT_DATASET_LOADING', flush=True)
    dataset = FitRows(dataset_dict={'nr3d': 1}, test_dataset='nr3d', split='train',
                      data_path='/root/autodl-tmp/DATA_ROOT/', use_color=args.use_color,
                      detect_intermediate=args.detect_intermediate, butd_cls=args.butd_cls,
                      skip_missing_superpoints=args.skip_missing_superpoints)
    dataset.augment = False
    assert len(dataset) == 16
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0,
                                        generator=torch.Generator().manual_seed(2027))
    results = []
    start = time.time()
    torch.cuda.reset_peak_memory_stats()
    for bid, raw in enumerate(loader):
        batch = TrainTester._to_gpu(raw)
        assert batch['sample_dataset'] == ['nr3d'] * 4
        assert bool(batch['box_label_mask'][:, 0].all())
        inputs = TrainTester._get_inputs(batch)
        inputs['train'] = False
        with torch.no_grad():
            outputs = model(inputs)
        outputs.update(batch)
        for key in ['last_sem_cls_scores', 'last_center', 'last_pred_size']:
            outputs[key].requires_grad_(True)
        captured = {}
        original = set_criterion.loss_pos_align

        def capture(o, t, idx, n, aux):
            if o['pred_logits'] is outputs['last_sem_cls_scores']:
                captured.update(outputs=o, targets=t, indices=idx, num_boxes=n)
            return original(o, t, idx, n, aux)

        set_criterion.loss_pos_align = capture
        native_total, outputs = TrainTester._compute_loss(outputs, criterion, set_criterion, args)
        set_criterion.loss_pos_align = original
        native_ce = outputs['last__loss_ce']
        intervention = assignment.LastLayerSemanticAssignment(set_criterion)
        intervention.bind(outputs, batch['sample_dataset'])
        changed_total, outputs = TrainTester._compute_loss(outputs, criterion, set_criterion, args)
        assert len(intervention.records) == 1
        record = intervention.records[0]
        intervention.remove()
        scale = 1. / (args.num_decoder_layers + 1)
        assert torch.allclose(changed_total, native_total + scale * record['delta'], rtol=1e-6, atol=1e-5)
        assert torch.equal(record['native_ce'], native_ce)
        logits = outputs['last_sem_cls_scores']
        labels = torch.zeros_like(logits)
        labels[..., -1] = 1
        weights = torch.full(logits.shape[:2], set_criterion.eos_coef, device=logits.device)
        for row, (queries, tids) in enumerate(captured['indices']):
            t = captured['targets'][row]
            target_labels = (t['positive_map'] * .6 + t['modify_positive_map'] * .2
                             + t['pron_positive_map'] * .2 + t['rel_positive_map'] * .1)
            labels[row, queries] = target_labels[tids]
            weights[row, queries] = 1
            labels[row, record['selected'][row]] = target_labels[0]
        explicit_ce = (((labels * torch.log(labels + 1e-6) - labels * logits.log_softmax(-1)).sum(-1)) * weights).sum() / captured['num_boxes']
        corrected_ce = native_ce + record['delta']
        assert torch.allclose(explicit_ce, corrected_ce, rtol=1e-6, atol=1e-6)
        gradients = [torch.autograd.grad(loss, logits, retain_graph=True)[0]
                     for loss in [explicit_ce, corrected_ce, record['delta']]]
        assert torch.allclose(gradients[0], gradients[1], rtol=1e-5, atol=1e-7)
        assert bool((gradients[2][~record['selected']] == 0).all())
        geometry = torch.autograd.grad(record['delta'],
            [outputs['last_center'], outputs['last_pred_size']], allow_unused=True)
        assert all(g is None for g in geometry)
        results.append({'fit_row_ids': fit_ids[bid*4:(bid+1)*4],
                        'selected_per_sample': record['selected'].sum(-1).tolist(),
                        'native_ce': float(native_ce), 'delta_ce': float(record['delta']),
                        'total_delta': float(changed_total-native_total),
                        'ce_error': float((explicit_ce-corrected_ce).abs()),
                        'gradient_error': float((gradients[0]-gradients[1]).abs().max())})
        print('NR_ASSIGNMENT_BATCH '+json.dumps(results[-1]), flush=True)
    assert sum(sum(r['selected_per_sample']) for r in results) > 0
    assert all(torch.equal(value.detach().cpu(), state[name]) for name, value in model.state_dict().items())
    report = {'status': 'pass', 'real_fit_rows': 16, 'seed': 2027,
              'model_source': str(source), 'checkpoint': manifest['checkpoint'],
              'optimizer_steps': 0, 'formal_eval_rows': 0, 'model_state_unchanged': True,
              'native_total_loss_scaling_verified': True, 'geometry_qualification_detached': True,
              'explicit_relabel_value_and_gradient_verified': True, 'batches': results,
              'elapsed_seconds_excluding_dataset_init': time.time()-start,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
    options.receipt.write_text(json.dumps(report, indent=2)+'\n')
    print('NR_ASSIGNMENT_PROBE_COMPLETE '+json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
