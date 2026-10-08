"""Train two independent native Mask correction heads on one frozen PV forward."""
import copy
import hashlib
import io
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from mask_support_corrector import CandidateMaskSupportCorrector
from mask_support_model_factory import ARMS, build_support_model
from matched_mask_objective import frozen_parent_assignments, matched_mask_loss
from native_root_bbs import native_root_bbs
from readback_preflight_checks import observed_readback_forward
from support_pair_forward import support_pair
from whole_model_preflight_checks import optimizer_restore_exact


class PairedSupportRun:
    def __init__(self, model, config, spec, output, args, initial, prepare, loader,
                 dataset, partitions, reset_rng, box_iou, evaluator, sha, write_json,
                 cfg, official, original_g, selected, data_root):
        self.model, self.config, self.spec, self.output = model, config, spec, output
        self.args, self.initial, self.prepare, self.loader = args, initial, prepare, loader
        self.dataset, self.partitions, self.reset_rng = dataset, partitions, reset_rng
        self.box_iou, self.evaluator, self.sha, self.write_json = box_iou, evaluator, sha, write_json
        self.cfg, self.official, self.original_g = cfg, official, original_g
        self.selected, self.data_root = selected, data_root
        warm = torch.load(spec['warm_support_terminal'], map_location='cpu')
        assert self.sha(spec['warm_support_terminal']) == spec['warm_support_terminal_sha256']
        assert warm['arm'] == 'content' and warm['step'] == 3723
        assert warm['reference_mode'] == 'fused_mask' and warm['mask_loss_coefficients'] == [5, 1, 10, 2]
        for key in ('checkpoint_sha256', 'base_terminal_sha256', 'selected_terminal_sha256', 'source_port_sha256'):
            assert warm[key] == spec[key]
        prefix = 'candidate_support_corrector.'
        assert len(warm['state_delta']) == 10 and all(name.startswith(prefix) for name in warm['state_delta'])
        self.warm_state = {name[len(prefix):]: value for name, value in warm['state_delta'].items()}
        self.heads = {'content': CandidateMaskSupportCorrector(False).cuda()}
        self.heads['content'].load_state_dict(self.warm_state, strict=True)
        assert torch.count_nonzero(self.heads['content'].output.weight) > 0
        with torch.no_grad():
            self.heads['content'].member[0].weight[:, -9:].zero_()
        for name, value in self.heads['content'].state_dict().items():
            expected = self.warm_state[name].clone()
            if name == 'member.0.weight':
                expected[:, -9:].zero_()
            assert torch.equal(value.cpu(), expected)
        self.heads['box_conditioned'] = copy.deepcopy(self.heads['content'])
        self.heads['box_conditioned'].use_box_geometry = True
        assert all(torch.equal(value, self.heads['box_conditioned'].state_dict()[name])
                   for name, value in self.heads['content'].state_dict().items())
        assert all(a.data_ptr() != b.data_ptr() for a, b in zip(
            self.heads['content'].parameters(), self.heads['box_conditioned'].parameters()))
        self.optimizers = {arm: torch.optim.AdamW(head.parameters(), lr=spec['lr'],
            weight_decay=spec['weight_decay']) for arm, head in self.heads.items()}
        self.steps, self.seen, self.forward_calls = 0, [], 0
        assert not any(parameter.requires_grad for parameter in model.parameters())
        for head in self.heads.values():
            assert len(head.state_dict()) == 10
            assert sum(parameter.numel() for parameter in head.parameters()) == 27841

    def forward_pair(self, inputs):
        with torch.no_grad():
            parent, call = observed_readback_forward(self.model, inputs)
        self.forward_calls += 1
        raw = inputs['points'][:, 1:].reshape(inputs['batch_size'], 50000, 6)
        pair = support_pair(self.model, self.heads, parent, raw)
        if self.steps == 0:
            assert all(torch.equal(a, b) for a, b in zip(
                pair['content']['sp_last_pred_masks'], pair['box_conditioned']['sp_last_pred_masks']))
            for key in ('last_center', 'last_pred_size', 'last_sem_cls_scores'):
                assert torch.equal(pair['content'][key], pair['box_conditioned'][key])
        return parent, pair, call

    def frozen_state_exact(self):
        assert set(self.model.state_dict()) == set(self.initial)
        assert all(torch.equal(value.detach().cpu(), self.initial[name])
                   for name, value in self.model.state_dict().items())
        return True

    def native_mask_witness(self, predictions, indices, targets, criterion, active):
        outputs = dict(pred_masks=predictions['last_pred_masks'],
            sp_pred_masks=predictions['sp_last_pred_masks'],
            adaptive_weights=predictions['adaptive_weights'],
            superpoints=predictions['superpoints'], super_xyz_list=predictions['super_xyz_list'])
        native = criterion.loss_masks(outputs, targets, indices,
            sum(len(value['boxes']) for value in targets), None)
        expected = (5 * native['sp_loss_mask'] + native['sp_loss_dice']
            + 10 * native['adaptive_weight_loss_mask'] + 2 * native['adaptive_weight_loss_dice'])
        assert torch.allclose(active, expected, rtol=1e-6, atol=1e-6)
        return dict(original_criterion_active_terms_reconciled=True,
            original_active_loss=float(expected), coefficients=[5, 1, 10, 2],
            supervision_denominator='actual valid GT count', division_by_seven=False)

    def step(self, batch_cpu, criterion, preflight=False):
        inputs, batch = self.prepare(batch_cpu, 'train')
        parent, pair, call = self.forward_pair(inputs)
        indices, targets = frozen_parent_assignments(parent, batch, criterion.matcher)
        assignment = [dict(queries=q.tolist(), actual_gt_ids=t.tolist()) for q, t in indices]
        records = {}
        for arm in ARMS:
            other = ARMS[1 - ARMS.index(arm)]
            head, optimizer = self.heads[arm], self.optimizers[arm]
            before_other = {name: value.detach().clone() for name, value in
                            self.heads[other].state_dict().items()} if preflight else None
            predictions = pair[arm]
            if preflight and self.steps == 0:
                original = CandidateMaskSupportCorrector(False).cuda()
                original.load_state_dict(self.warm_state, strict=True)
                original.eval()
                with torch.no_grad():
                    raw = inputs['points'][:, 1:].reshape(inputs['batch_size'], 50000, 6)
                    expected_masks, _ = original(parent['support_query_features'],
                        parent['support_super_features'], raw, parent['native_coarse_center'],
                        parent['native_coarse_size'], parent)
                assert all(torch.equal(a, b) for a, b in zip(
                    predictions['sp_last_pred_masks'], expected_masks))
                del original, expected_masks
            loss, record = matched_mask_loss(predictions, indices, targets)
            if preflight:
                record.update(self.native_mask_witness(predictions, indices, targets, criterion, loss))
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
            assert all(p.grad is None for p in self.heads[other].parameters())
            assert all(p.grad is None for p in self.model.parameters())
            if preflight:
                gradients = {name: float(p.grad.norm()) for name, p in head.named_parameters()}
                assert gradients['output.weight'] > 0
                if self.steps:
                    assert sum(norm for name, norm in gradients.items() if not name.startswith('output.')) > 0
                record['raw_parameter_gradient_norms'] = gradients
                record['zero_update_warm_mask_exact'] = self.steps == 0
                record['warm_arm_mask_and_box_exact'] = self.steps == 0
                record['geometry_column_gradient_norm'] = float(head.member[0].weight.grad[:, -9:].norm())
                if self.steps == 0:
                    assert (record['geometry_column_gradient_norm'] > 0) == (arm == 'box_conditioned')
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), self.spec['clip_norm'])
            assert torch.isfinite(norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if preflight:
                assert all(torch.equal(value, self.heads[other].state_dict()[name])
                           for name, value in before_other.items())
            record.update(gradient_norm=float(norm), independent_head_gradients=True)
            records[arm] = record
        self.steps += 1
        if preflight:
            self.frozen_state_exact()
        return dict(step=self.steps, rows=batch['local_training_id'].cpu().tolist(),
            arms=records, frozen_parent_forwards_per_batch=1,
            final_semantic_head_calls=call['final_semantic_head_calls'],
            original_parent_assignment=assignment, expanded_positive_queries=0)

    def payload(self, arm):
        return dict(state_delta={'candidate_support_corrector.' + name: value.detach().cpu().clone()
                                 for name, value in self.heads[arm].state_dict().items()},
            optimizer=self.optimizers[arm].state_dict(), arm=arm, step=self.steps,
            row_ids=list(self.seen), reference_mode='fused_mask', mask_loss_coefficients=[5, 1, 10, 2],
            trainable_parameters=27841, trainable_state_tensors=10, parent_and_box_head_frozen=True,
            spec_sha256=self.sha(self.args.spec), checkpoint_sha256=self.spec['checkpoint_sha256'],
            base_terminal_sha256=self.spec['base_terminal_sha256'],
            selected_terminal_sha256=self.spec['selected_terminal_sha256'],
            source_port_sha256=self.spec['source_port_sha256'],
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            numpy_rng=np.random.get_state(), python_rng=random.getstate(),
            geometry_encoding='signed_log' if arm == 'box_conditioned' else 'zero',
            warm_support_terminal_sha256=self.spec['warm_support_terminal_sha256'],
            support_prior_updates=3723, total_support_updates=3723 + self.steps,
            optimizer_reinitialized=True, unused_geometry_columns_zeroed_at_initialization=True)

    def save(self):
        for arm in ARMS:
            temporary = self.output / arm / 'terminal.pth.tmp'
            torch.save(self.payload(arm), str(temporary))
            os.replace(str(temporary), str(self.output / arm / 'terminal.pth'))

    def restore_terminals(self):
        for arm in ARMS:
            path = self.output / arm / 'terminal.pth'
            payload = torch.load(str(path), map_location='cpu')
            assert payload['arm'] == arm and payload['step'] == 3723
            assert payload['spec_sha256'] == self.sha(self.args.spec)
            assert payload['warm_support_terminal_sha256'] == self.spec['warm_support_terminal_sha256']
            assert payload['total_support_updates'] == 7446 and payload['optimizer_reinitialized']
            assert payload['geometry_encoding'] == ('signed_log' if arm == 'box_conditioned' else 'zero')
            for key in ('checkpoint_sha256', 'base_terminal_sha256', 'selected_terminal_sha256', 'source_port_sha256'):
                assert payload[key] == self.spec[key]
            assert Counter(payload['row_ids']) == Counter(self.partitions['fit_saved'])
            state = {name[len('candidate_support_corrector.'):]: value
                     for name, value in payload['state_delta'].items()}
            self.heads[arm].load_state_dict(state, strict=True)
            self.optimizers[arm].load_state_dict(payload['optimizer'])
            assert all(int(value['step']) == 3723 for value in self.optimizers[arm].state.values())
            self.write_json(self.output / arm / 'formal_restore.json', dict(status='pass',
                terminal_sha256=self.sha(path), optimizer=optimizer_restore_exact(
                    self.optimizers[arm], payload['optimizer'])))
        self.steps = 3723

    def restore_and_integration_witness(self, batch_cpu):
        checks = {}
        for arm in ARMS:
            memory = io.BytesIO()
            torch.save(self.payload(arm), memory)
            length = memory.tell()
            memory.seek(0)
            payload = torch.load(memory, map_location='cpu')
            self.reset_rng()
            rebuilt, _, receipt = build_support_model(self.cfg, self.official,
                self.original_g, self.selected, self.data_root, payload)
            expected = dict(self.initial, **payload['state_delta'])
            assert set(rebuilt.state_dict()) == set(expected)
            assert all(torch.equal(value.detach().cpu(), expected[name])
                       for name, value in rebuilt.state_dict().items())
            restored_head = rebuilt.candidate_support_corrector
            cpu_optimizer = torch.optim.AdamW(restored_head.parameters(), lr=self.spec['lr'],
                weight_decay=self.spec['weight_decay'])
            cpu_optimizer.load_state_dict(payload['optimizer'])
            optimizer_restore_exact(cpu_optimizer, payload['optimizer'])
            restored_head = restored_head.cuda()
            gpu_optimizer = torch.optim.AdamW(restored_head.parameters(), lr=self.spec['lr'],
                weight_decay=self.spec['weight_decay'])
            gpu_optimizer.load_state_dict(payload['optimizer'])
            optimizer = optimizer_restore_exact(gpu_optimizer, payload['optimizer'])
            assert all(int(value['step']) == 2 for value in gpu_optimizer.state.values())
            assert all(p.is_cuda for p in restored_head.parameters())
            inputs, _ = self.prepare(batch_cpu, 'train')
            self.reset_rng()
            captured_inputs, captured_outputs = [], []

            def capture_before(module, arguments):
                query, features, points, center, size, predictions = arguments
                snapshot = dict(query=query.detach().clone(),
                    features=[value.detach().clone() for value in features],
                    points=points.detach().clone(), center=center.detach().clone(), size=size.detach().clone())
                for key in ('last_pred_masks', 'sp_last_pred_masks', 'adaptive_weights', 'superpoints'):
                    snapshot[key] = [value.detach().clone() for value in predictions[key]]
                captured_inputs.append(snapshot)

            before_hook = restored_head.register_forward_pre_hook(capture_before)
            after_hook = restored_head.register_forward_hook(lambda module, arguments, result:
                captured_outputs.append([value.detach().clone() for value in result[0]]))
            self.model.candidate_support_corrector = restored_head
            self.model.eval()
            with torch.no_grad():
                integrated, call = observed_readback_forward(self.model, inputs)
            before_hook.remove()
            after_hook.remove()
            self.model.candidate_support_corrector = None
            assert len(captured_inputs) == len(captured_outputs) == 1
            cache = captured_inputs[0]
            parent = dict(integrated)
            parent.update(native_coarse_center=cache['center'], native_coarse_size=cache['size'],
                support_query_features=cache['query'], support_super_features=cache['features'])
            for key in ('last_pred_masks', 'sp_last_pred_masks', 'adaptive_weights', 'superpoints'):
                parent[key] = cache[key]
            with torch.no_grad():
                pair = support_pair(self.model, self.heads, parent, cache['points'])
            target = pair[arm]
            keys = ('last_center', 'last_pred_size', 'last_sem_cls_scores')
            errors = {key: float((integrated[key] - target[key]).abs().max()) for key in keys}
            assert all(torch.equal(integrated[key], target[key]) for key in keys)
            mask_errors = [float((a - b).abs().max()) for a, b in zip(
                integrated['sp_last_pred_masks'], target['sp_last_pred_masks'])]
            assert all(value == 0 for value in mask_errors)
            assert all(torch.equal(a, b) for a, b in zip(captured_outputs[0], target['sp_last_pred_masks']))
            assert call['final_semantic_head_calls'] == 1
            assert torch.equal(integrated['last_semantic_query_before_readback'],
                               integrated['last_semantic_query_after_readback'])
            self.frozen_state_exact()
            checks[arm] = dict(full_cpu_state_exact=True, full_state_tensors=receipt['full_state_tensors'],
                gpu_head_and_optimizer_restore=optimizer, actual_native_gpu_integration=True,
                integration_comparison_scope='same native forward captured uncorrected inputs',
                corrector_native_calls=1, same_cache_masks_and_geometry_exact=True, max_tensor_errors=errors,
                query_mask_max_errors=mask_errors, serialization_bytes=length,
                gpu_parent_cold_reconstruction_executed=False, weight_files_created=0)
            del rebuilt, cpu_optimizer, gpu_optimizer, restored_head, parent, pair, integrated, target
        return checks

    def evaluate(self, stage, formal):
        self.dataset.augment = self.dataset.augment_det = False
        self.model.eval()
        for head in self.heads.values():
            head.eval()
        self.reset_rng()
        directory = self.output / stage
        directory.mkdir()
        evaluators = {arm: self.evaluator(only_root=True, thresholds=[.25, .5], topks=[1, 5, 10],
            prefixes=['last_'], filter_non_gt_boxes=False, model='PVGround') for arm in ARMS}
        rows, start = [], time.time()
        with torch.no_grad(), (directory / 'rows.jsonl').open('w') as stream:
            for batch_cpu in self.loader('holdout', False):
                inputs, batch = self.prepare(batch_cpu, 'eval')
                parent, pair, call = self.forward_pair(inputs)
                scores = native_root_bbs(parent['last_sem_cls_scores'], batch)
                finals = {arm: torch.cat([value['last_center'], value['last_pred_size']], -1)
                          for arm, value in pair.items()}
                parent_boxes = torch.cat([parent['last_center'], parent['last_pred_size']], -1)
                truth = torch.cat([batch['center_label'][:, 0, :3], batch['size_gts'][:, 0]], -1)
                if formal:
                    np.savez_compressed(directory / ('batch_%05d.npz' % len(rows)),
                        row_ids=batch['local_training_id'].cpu().numpy(), root_gt=truth.cpu().numpy(),
                        scores=scores.cpu().numpy(), parent=parent_boxes.cpu().numpy(),
                        content=finals['content'].cpu().numpy(), box_conditioned=finals['box_conditioned'].cpu().numpy())
                for arm in ARMS:
                    assert (finals[arm][..., 3:] > 0).all()
                    assert not set(pair[arm]).intersection(batch)
                    pair[arm].update(batch)
                    for key in pair[arm]:
                        if 'pred_size' in key:
                            pair[arm][key] = pair[arm][key].clamp(min=1e-6)
                    evaluators[arm].evaluate(pair[arm], 'last_')
                for bid in range(len(batch['utterances'])):
                    row_id = int(batch['local_training_id'][bid])
                    assert row_id == self.partitions['holdout'][len(rows)]
                    ranked = scores[bid].argsort(descending=True)
                    query, alpha = int(ranked[0]), parent['adaptive_weights'][bid]
                    original_iou = self.box_iou(parent_boxes[bid], truth[bid])
                    record = dict(row_id=row_id, scan_id=batch['scan_ids'][bid], target_id=int(batch['target_id'][bid]),
                        query=query, root_box=truth[bid].cpu().tolist(),
                        point_sha256=hashlib.sha256(batch['point_clouds'][bid].cpu().numpy().tobytes()).hexdigest(),
                        parent_box=parent_boxes[bid, query].cpu().tolist(), parent_iou=float(original_iou[query]),
                        parent_forwards=1, final_semantic_head_calls=call['final_semantic_head_calls'], arms={})
                    for arm in ARMS:
                        value = pair[arm]
                        mask = ((alpha * value['last_pred_masks'][bid][0, query]
                            + (1 - alpha) * value['sp_last_pred_masks'][bid][query]).sigmoid() > .5)[value['superpoints'][bid]]
                        target_mask = batch['gt_masks'][bid, 0].bool()
                        mask_iou = float((mask & target_mask).sum().float() / (mask | target_mask).sum())
                        iou = self.box_iou(finals[arm][bid], truth[bid])
                        record['arms'][arm] = dict(box=finals[arm][bid, query].cpu().tolist(),
                            iou=float(iou[query]), mask_iou=mask_iou,
                            reference_valid=bool(value['mask_reference_valid'][bid, query]),
                            oracle25=[int((iou[ranked[:count]] > .25).any()) for count in (16, 32, 64, 256)],
                            oracle50=[int((iou[ranked[:count]] > .5).any()) for count in (16, 32, 64, 256)])
                    rows.append(record)
                    stream.write(json.dumps(record) + '\n')
                if len(rows) % 512 < 8:
                    stream.flush()
                    print('PAIRED_SUPPORT_EVAL_PROGRESS ' + json.dumps(dict(stage=stage, rows=len(rows),
                        total=len(self.partitions['holdout']), seconds=time.time() - start)), flush=True)
                del parent, pair, inputs, batch
        assert len(rows) == len(self.partitions['holdout'])
        metrics = {}
        for arm in ARMS:
            hits25, hits50 = (sum(row['arms'][arm]['iou'] > threshold for row in rows) for threshold in (.25, .5))
            assert hits25 == evaluators[arm].dets[('last_', .25, 1, 'bbs')]
            assert hits50 == evaluators[arm].dets[('last_', .5, 1, 'bbs')]
            mask_sum = sum(row['arms'][arm]['mask_iou'] for row in rows)
            assert abs(mask_sum - float(evaluators[arm].dets['mask_pos'])) < 1e-3
            metrics[arm] = dict(rec_hits25=hits25, rec_hits50=hits50,
                mask_hits25=sum(row['arms'][arm]['mask_iou'] > .25 for row in rows),
                mask_hits50=sum(row['arms'][arm]['mask_iou'] > .5 for row in rows),
                mask_miou=mask_sum / len(rows) * 100,
                repairs25=sum(row['parent_iou'] <= .25 < row['arms'][arm]['iou'] for row in rows),
                damages25=sum(row['arms'][arm]['iou'] <= .25 < row['parent_iou'] for row in rows),
                repairs50=sum(row['parent_iou'] <= .5 < row['arms'][arm]['iou'] for row in rows),
                damages50=sum(row['arms'][arm]['iou'] <= .5 < row['parent_iou'] for row in rows))
        receipt = dict(status='pass', stage=stage, rows=len(rows), metrics=metrics,
            formal_rows=len(rows) if formal else 0, primary_mode='bbs', all256_retained=True,
            same_selected_query_box_and_mask=True, frozen_parent_forward_per_batch=1,
            parent_hits25=sum(row['parent_iou'] > .25 for row in rows),
            parent_hits50=sum(row['parent_iou'] > .5 for row in rows),
            elapsed_seconds=time.time() - start, rows_sha256=self.sha(directory / 'rows.jsonl'))
        if stage == 'initial_formal':
            actual_warm = [metrics['content']['rec_hits25'], metrics['content']['rec_hits50']]
            assert [metrics['box_conditioned']['rec_hits25'], metrics['box_conditioned']['rec_hits50']] == actual_warm
            receipt.update(historical_protected_hits=[5598, 4856],
                historical_protected_difference=[actual-old for actual,old in zip(actual_warm,[5598,4856])],
                historical_best_updated=False, both_warm_heads_same_forward_exact=True)
        self.write_json(directory / 'receipt.json', receipt)
        print('PAIRED_SUPPORT_EVAL_COMPLETE ' + json.dumps(receipt), flush=True)
        return rows, receipt

    def run(self, criterion, manifest, started):
        if self.args.mode in ('initial_formal', 'formal'):
            if self.args.mode == 'formal':
                self.restore_terminals()
            self.evaluate(self.args.mode, True)
            self.frozen_state_exact()
            return
        if self.args.mode == 'preflight':
            self.dataset.augment = self.dataset.augment_det = True
            self.reset_rng()
            for index, batch_cpu in enumerate(self.loader('fit', True)):
                if index == self.spec['preflight_batch_index']:
                    break
            assert index == self.spec['preflight_batch_index']
            reference = [json.loads(line) for line in Path(
                self.spec['preflight_reference_rows']).read_text().splitlines()]
            for bid, row_id in enumerate(batch_cpu['local_training_id'].tolist()):
                row = reference[index * 8 + bid]
                assert row_id == row['row_id']
                assert hashlib.sha256(batch_cpu['point_clouds'][bid].numpy().tobytes()).hexdigest() == row['point_sha256']
                assert torch.cat([batch_cpu['center_label'][bid, 0, :3],
                    batch_cpu['size_gts'][bid, 0]]).tolist() == row['noisy_root_box']
            witnesses = [self.step(batch_cpu, criterion, True) for _ in range(2)]
            restore = self.restore_and_integration_witness(batch_cpu)
            self.write_json(self.output / 'preflight.json', dict(status='pass',
                optimizer_steps_per_arm=2, weight_files_created=0, accuracy_result=False,
                trainable_parameters_per_arm=27841, witnesses=witnesses, restore=restore,
                separate_optimizers_and_gradients=True, parent_and_box_head_frozen=True,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                elapsed_seconds=time.perf_counter() - started, spec_sha256=self.sha(self.args.spec)))
            print('PAIRED_SUPPORT_PREFLIGHT_COMPLETE', flush=True)
            return
        initial_rows, initial_receipt = self.evaluate('initial', False)
        self.dataset.augment = self.dataset.augment_det = True
        self.reset_rng()
        total = math.ceil(len(self.partitions['fit']) / 8)
        assert total == 3723
        with (self.output / 'train.jsonl').open('w') as stream:
            for index, batch_cpu in enumerate(self.loader('fit', True), 1):
                record = self.step(batch_cpu, criterion)
                assert len(record['rows']) == (2 if index == total else 8)
                self.seen.extend(record['rows'])
                stream.write(json.dumps(record) + '\n')
                if index == 1 or index % 128 == 0 or index == total:
                    stream.flush()
                    print('PAIRED_SUPPORT_TRAIN_PROGRESS ' + json.dumps(record), flush=True)
        assert self.steps == total and Counter(self.seen) == Counter(self.partitions['fit'])
        self.save()
        final_rows, final_receipt = self.evaluate('terminal', False)
        assert all(old['row_id'] == new['row_id'] and old['point_sha256'] == new['point_sha256']
                   and old['root_box'] == new['root_box']
                   for old, new in zip(initial_rows, final_rows))
        parent_drift = [dict(row_id=old['row_id'], old_query=old['query'], new_query=new['query'],
            query_changed=old['query'] != new['query'],
            box_max_abs_difference=max(abs(a-b) for a,b in zip(old['parent_box'],new['parent_box'])))
            for old,new in zip(initial_rows,final_rows)]
        self.write_json(self.output / 'frozen_parent_repeat_drift.json', dict(
            comparison_scope='independent full passes, unchanged parent states', rows=parent_drift,
            query_changes=sum(value['query_changed'] for value in parent_drift),
            box_changes=sum(value['box_max_abs_difference'] > 0 for value in parent_drift),
            numerical_drift_is_not_support_head_gain=True))
        self.frozen_state_exact()
        self.write_json(self.output / 'receipt.json', dict(status='complete',
            training_steps_per_arm=3723, fit_rows_per_arm=len(self.seen), holdout_rows=len(final_rows),
            initial=initial_receipt['metrics'], terminal=final_receipt['metrics'], formal_rows=0,
            fit_seen_exactly_once_per_arm=True, parent_and_box_head_states_exact=True,
            elapsed_seconds=time.perf_counter() - started, spec_sha256=self.sha(self.args.spec)))
