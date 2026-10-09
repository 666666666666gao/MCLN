"""Original matched box supervision for equal-capacity support-source controls."""
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from mask_support_model_factory import build_support_model
from matched_mask_objective import frozen_parent_assignments
from matched_span_objective import matched_span_loss
from native_root_bbs import native_root_bbs
from readback_preflight_checks import observed_readback_forward
from span_refinement_model import (SpanRefinementModel, apply_span_mixer,
                                   span_checkpoint_payload, restore_span_checkpoint)
from whole_model_preflight_checks import optimizer_restore_exact


ARMS = ('whole_support', 'extremal_support')


class PairedSpanRun:
    def __init__(self, model, config, spec, output, args, initial, prepare, loader,
                 dataset, partitions, reset_rng, box_iou, evaluator, sha, write_json,
                 cfg, official, original_g, selected, data_root, parent_support):
        self.parent, self.config, self.spec, self.output = model, config, spec, output
        self.args, self.initial, self.prepare, self.loader = args, initial, prepare, loader
        self.dataset, self.partitions, self.reset_rng = dataset, partitions, reset_rng
        self.box_iou, self.evaluator, self.sha, self.write_json = box_iou, evaluator, sha, write_json
        self.cfg, self.official, self.original_g = cfg, official, original_g
        self.selected, self.data_root, self.parent_support = selected, data_root, parent_support
        self.parent_identity = {key: spec[key] for key in (
            'checkpoint_sha256', 'base_terminal_sha256', 'selected_terminal_sha256',
            'parent_support_terminal_sha256', 'source_port_sha256', 'env_spec_sha256')}
        self.models = {arm: SpanRefinementModel(model, arm).cuda() for arm in ARMS}
        self.models['extremal_support'].mixer.load_state_dict(
            self.models['whole_support'].mixer.state_dict(), strict=True)
        self.heads = {arm: value.mixer for arm, value in self.models.items()}
        assert all(torch.equal(value, self.heads[ARMS[1]].state_dict()[name])
                   for name, value in self.heads[ARMS[0]].state_dict().items())
        assert all(a.data_ptr() != b.data_ptr() for a, b in zip(
            self.heads[ARMS[0]].parameters(), self.heads[ARMS[1]].parameters()))
        self.optimizers = {arm: torch.optim.AdamW(head.parameters(), lr=spec['lr'],
            weight_decay=spec['weight_decay']) for arm, head in self.heads.items()}
        for arm in ARMS:
            assert len(self.models[arm].state_dict()) == 1328
            assert len(self.heads[arm].state_dict()) == 14
            assert sum(p.numel() for p in self.heads[arm].parameters()) == 29793
            assert not self.optimizers[arm].state
            assert not set(map(id, self.parent.parameters())).intersection(
                set(map(id, self.heads[arm].parameters())))
        self.steps, self.seen = 0, []
        self.parent.eval()

    def frozen_state_exact(self):
        current = self.parent.state_dict()
        assert set(current) == set(self.initial)
        assert all(torch.equal(value.detach().cpu(), self.initial[name]) for name, value in current.items())
        assert not any(p.requires_grad or p.grad is not None for p in self.parent.parameters())
        return True

    def forward_pair(self, inputs):
        with torch.no_grad():
            parent, call = observed_readback_forward(self.parent, inputs)
        pair = {arm: apply_span_mixer(head, parent, parent['support_member_geometry'])
                for arm, head in self.heads.items()}
        for value in pair.values():
            assert value['last_sem_cls_scores'] is parent['last_sem_cls_scores']
            assert value['sp_last_pred_masks'] is parent['sp_last_pred_masks']
            assert value['last_pred_masks'] is parent['last_pred_masks']
            assert (value['last_pred_size'] > 0).all()
        if self.steps == 0:
            for value in pair.values():
                assert torch.equal(value['last_center'], parent['last_center'])
                assert torch.equal(value['last_pred_size'], parent['last_pred_size'])
                assert torch.count_nonzero(value['span_axis_gate']) == 0
        return parent, pair, call

    def support_input_witness(self, parent, pair, inputs, batch, indices, targets):
        """Record real input/member provenance; no prediction or target is changed."""
        directory = self.output / ('preflight_support_step_%d' % (self.steps + 1))
        directory.mkdir()
        raw = inputs['points'][:, 1:].reshape(inputs['batch_size'], 50000, 6)
        scores = native_root_bbs(parent['last_sem_cls_scores'], batch)
        records = []
        for bid, (queries, gt_ids) in enumerate(indices):
            selected = int(scores[bid].argsort(descending=True)[0])
            observed_queries = torch.unique(torch.cat([queries, queries.new_tensor([selected])]), sorted=True)
            text, own, alpha = (parent['last_pred_masks'][bid][0],
                                parent['sp_last_pred_masks'][bid], parent['adaptive_weights'][bid])
            foreground = (alpha * text + (1 - alpha) * own).sigmoid() > .5
            geometry = parent['support_member_geometry'][bid]
            path = directory / ('row_%02d.npz' % bid)
            actual_points = raw[bid].detach().cpu().numpy()
            point_sha = hashlib.sha256(actual_points.tobytes()).hexdigest()
            assert point_sha == hashlib.sha256(batch['point_clouds'][bid].cpu().numpy().tobytes()).hexdigest()
            arrays = dict(raw_points=actual_points, superpoints=parent['superpoints'][bid].cpu().numpy(),
                scores=scores[bid].cpu().numpy(), foreground_sp=foreground.cpu().numpy(),
                reference_valid=parent['mask_reference_valid'][bid].cpu().numpy(),
                observed_queries=observed_queries.cpu().numpy(),
                observed_text_logits=text[observed_queries].cpu().numpy(),
                observed_query_logits=own[observed_queries].cpu().numpy(), alpha=alpha.cpu().numpy(),
                mask_center=parent['last_center'][bid].cpu().numpy(), mask_size=parent['last_pred_size'][bid].cpu().numpy(),
                native_center=parent['native_coarse_center'][bid].cpu().numpy(), native_size=parent['native_coarse_size'][bid].cpu().numpy(),
                matched_queries=queries.cpu().numpy(), matched_gt_ids=gt_ids.cpu().numpy(),
                actual_gt_boxes=targets[bid]['boxes'].cpu().numpy(), actual_gt_masks=targets[bid]['masks'].cpu().numpy())
            arrays.update({'member_' + name: np.asarray(value) for name, value in geometry.items()})
            arrays.update({arm + '_source_fraction': pair[arm]['span_source_fraction'][bid].detach().cpu().numpy() for arm in ARMS})
            np.savez_compressed(path, **arrays)
            records.append(dict(row_id=int(batch['local_training_id'][bid]), bid=bid,
                selected_query=selected, observed_queries=observed_queries.cpu().tolist(),
                raw_points_sha256=point_sha, scene=batch['scan_ids'][bid], path=str(path),
                bytes=path.stat().st_size, sha256=self.sha(path), candidate_count=256,
                raw_logits_scope='native winner plus every original matched query; foreground/members cover all256',
                gt_scope='actual dataset boxes/masks for provenance only; not an inference input'))
        self.write_json(directory / 'receipt.json', dict(status='ACTUAL_SUPPORT_INPUTS_CAPTURED',
            step=self.steps + 1, rows=records, neural_replays=0, targets_or_predictions_changed=False))
        return records

    def step(self, batch_cpu, criterion, preflight=False):
        inputs, batch = self.prepare(batch_cpu, 'train')
        parent, pair, call = self.forward_pair(inputs)
        indices, targets = frozen_parent_assignments(parent, batch, criterion.matcher)
        support_witness = self.support_input_witness(parent, pair, inputs, batch, indices, targets) if preflight else None
        records = {}
        for arm in ARMS:
            other = ARMS[1 - ARMS.index(arm)]
            head, optimizer = self.heads[arm], self.optimizers[arm]
            prior_other = {name: value.detach().clone() for name, value in self.heads[other].state_dict().items()} if preflight else None
            loss, record = matched_span_loss(pair[arm], indices, targets, criterion)
            if preflight:
                gradients = torch.autograd.grad(loss, (pair[arm]['last_center'], pair[arm]['last_pred_size']), retain_graph=True)
                matched = torch.zeros_like(pair[arm]['last_center'][..., 0], dtype=torch.bool)
                for bid, (queries, _) in enumerate(indices):
                    matched[bid, queries] = True
                assert all(torch.count_nonzero(value[~matched]) == 0 for value in gradients)
                record['direct_output_gradient'] = dict(
                    unmatched_nonzero=0, matched_queries=int(matched.sum()),
                    matched_norms=[float(value[matched].norm()) for value in gradients])
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
            assert all(p.grad is None for p in self.heads[other].parameters())
            assert all(p.grad is None for p in self.parent.parameters())
            if preflight:
                norms = {name: float(p.grad.norm()) for name, p in head.named_parameters()}
                assert norms['output.weight'] > 0
                upstream = sum(value for name, value in norms.items() if not name.startswith('output.'))
                assert upstream > 0 if self.steps else upstream == 0
                record.update(raw_parameter_gradient_norms=norms,
                    upstream_group_grad_norms={group: sum(value for name, value in norms.items()
                        if name.startswith(group + '.')) for group in
                        ('query_projection', 'support_projection', 'face_encoder', 'axis_decoder')},
                    neutral_same_forward_exact=self.steps == 0,
                    axis_gate_min=float(pair[arm]['span_axis_gate'].min()),
                    axis_gate_max=float(pair[arm]['span_axis_gate'].max()),
                    invalid_reference_queries=int((~parent['mask_reference_valid']).sum()),
                    source_fraction_min=float(pair[arm]['span_source_fraction'].min()),
                    source_fraction_max=float(pair[arm]['span_source_fraction'].max()))
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), self.spec['clip_norm'])
            assert torch.isfinite(norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if preflight:
                assert all(torch.equal(value, self.heads[other].state_dict()[name]) for name, value in prior_other.items())
            record.update(loss=float(loss), gradient_norm=float(norm), independent_head_gradients=True)
            records[arm] = record
        self.steps += 1
        if preflight:
            self.frozen_state_exact()
        return dict(step=self.steps, rows=batch['local_training_id'].cpu().tolist(), arms=records,
            frozen_parent_forwards_per_batch=1, final_semantic_head_calls=call['final_semantic_head_calls'],
            original_parent_assignment=[dict(queries=q.tolist(), actual_gt_ids=t.tolist()) for q, t in indices],
            support_input_witness=support_witness,
            expanded_positive_queries=0)

    def payload(self, arm):
        value = span_checkpoint_payload(self.models[arm], self.optimizers[arm], self.steps, self.parent_identity)
        value['mixer_state'] = {name: tensor.detach().cpu().clone() for name, tensor in value['mixer_state'].items()}
        value.update(spec_sha256=self.sha(self.args.spec), row_ids=list(self.seen),
            trainable_parameters=29793, trainable_state_tensors=14,
            parent_and_mask_and_score_frozen=True, optimizer_reinitialized=True,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            numpy_rng=np.random.get_state(), python_rng=random.getstate())
        return value

    def save(self):
        for arm in ARMS:
            temporary = self.output / arm / 'terminal.pth.tmp'
            torch.save(self.payload(arm), str(temporary))
            os.replace(str(temporary), str(self.output / arm / 'terminal.pth'))

    def restore_terminals(self):
        for arm in ARMS:
            path = self.output / arm / 'terminal.pth'
            payload = torch.load(str(path), map_location='cpu')
            assert payload['step'] == 3723 and payload['source_mode'] == arm
            assert payload['spec_sha256'] == self.sha(self.args.spec)
            assert Counter(payload['row_ids']) == Counter(self.partitions['fit_saved'])
            assert restore_span_checkpoint(self.models[arm], self.optimizers[arm], payload, self.parent_identity) == 3723
            assert all(int(value['step']) == 3723 for value in self.optimizers[arm].state.values())
            self.write_json(self.output / arm / 'formal_restore.json', dict(status='pass',
                terminal_sha256=self.sha(path), source_mode=arm, parent_identity=self.parent_identity,
                optimizer=optimizer_restore_exact(self.optimizers[arm], payload['optimizer'])))
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
            rebuilt_parent, _, receipt = build_support_model(self.cfg, self.official,
                self.original_g, self.selected, self.data_root, self.parent_support)
            rebuilt = SpanRefinementModel(rebuilt_parent, arm)
            expected = {'parent.' + name: value for name, value in self.initial.items()}
            expected.update({'mixer.' + name: value for name, value in payload['mixer_state'].items()})
            cpu_optimizer = torch.optim.AdamW(rebuilt.mixer.parameters(), lr=self.spec['lr'], weight_decay=self.spec['weight_decay'])
            assert restore_span_checkpoint(rebuilt, cpu_optimizer, payload, self.parent_identity) == 2
            assert set(rebuilt.state_dict()) == set(expected) and len(expected) == 1328
            assert all(torch.equal(value.detach().cpu(), expected[name]) for name, value in rebuilt.state_dict().items())
            optimizer_restore_exact(cpu_optimizer, payload['optimizer'])
            restored_head = rebuilt.mixer.cuda()
            deployed = SpanRefinementModel(self.parent, arm)
            deployed.mixer = restored_head
            gpu_optimizer = torch.optim.AdamW(deployed.mixer.parameters(), lr=self.spec['lr'], weight_decay=self.spec['weight_decay'])
            assert restore_span_checkpoint(deployed, gpu_optimizer, payload, self.parent_identity) == 2
            optimizer = optimizer_restore_exact(gpu_optimizer, payload['optimizer'])
            torch.set_rng_state(payload['torch_rng'])
            torch.cuda.set_rng_state_all(payload['cuda_rng'])
            np.random.set_state(payload['numpy_rng'])
            random.setstate(payload['python_rng'])
            assert torch.equal(torch.get_rng_state(), payload['torch_rng'])
            assert all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), payload['cuda_rng']))
            current_numpy = np.random.get_state()
            assert current_numpy[0] == payload['numpy_rng'][0] and np.array_equal(current_numpy[1], payload['numpy_rng'][1])
            assert current_numpy[2:] == payload['numpy_rng'][2:] and random.getstate() == payload['python_rng']
            inputs, _ = self.prepare(batch_cpu, 'train')
            captured, events = [], []
            def capture_parent(module, arguments, result):
                captured.append(result)
                events.append('parent_complete')
            handles = [self.parent.register_forward_hook(capture_parent),
                self.parent.prediction_heads[-1].sem_cls_scores_head.register_forward_pre_hook(
                    lambda module, arguments: events.append('native_semantic_head')),
                restored_head.register_forward_pre_hook(lambda module, arguments: events.append('span_mixer'))]
            with torch.no_grad():
                integrated = deployed(dict(inputs))
            for handle in handles:
                handle.remove()
            assert events == ['native_semantic_head', 'parent_complete', 'span_mixer'] and len(captured) == 1
            with torch.no_grad():
                reference = apply_span_mixer(self.heads[arm], captured[0], captured[0]['support_member_geometry'])
            assert all(torch.equal(integrated[key], reference[key]) for key in
                       ('last_center', 'last_pred_size', 'last_sem_cls_scores', 'span_axis_gate'))
            assert integrated['sp_last_pred_masks'] is captured[0]['sp_last_pred_masks']
            assert integrated['last_pred_masks'] is captured[0]['last_pred_masks']
            self.frozen_state_exact()
            checks[arm] = dict(full_cpu_state_exact=True, full_state_tensors=1328,
                parent_cpu_state_tensors=receipt['full_state_tensors'], source_mode_exact=True,
                actual_native_dictionary_model_call=True, actual_native_gpu_integration=True,
                original_semantic_calls=1, original_parent_calls=1, deployed_mixer_calls=1,
                updated_same_cache_output_exact=True, gpu_head_and_optimizer_restore=optimizer,
                cpu_and_cuda_numpy_python_rng_restored=True, serialization_bytes=length,
                comparison_scope='actual wrapper versus restored same-parent-cache head; not repeated-backbone identity',
                gpu_parent_cold_reconstruction_executed=False, weight_files_created=0)
            del rebuilt_parent, rebuilt, cpu_optimizer, gpu_optimizer, deployed, restored_head, integrated, reference, captured
        return checks

    def evaluate(self):
        self.dataset.augment = self.dataset.augment_det = False
        self.parent.eval()
        self.reset_rng()
        directory = self.output / 'formal'
        directory.mkdir()
        evaluators = {arm: self.evaluator(only_root=True, thresholds=[.25, .5], topks=[1, 5, 10],
            prefixes=['last_'], filter_non_gt_boxes=False, model='PVGround') for arm in ARMS}
        rows, started = [], time.time()
        with torch.no_grad(), (directory / 'rows.jsonl').open('w') as stream:
            for batch_cpu in self.loader('holdout', False):
                inputs, batch = self.prepare(batch_cpu, 'eval')
                parent, pair, call = self.forward_pair(inputs)
                scores = native_root_bbs(parent['last_sem_cls_scores'], batch)
                boxes = dict(native=torch.cat([parent['native_coarse_center'], parent['native_coarse_size'].clamp_min(1e-6)], -1),
                    mask=torch.cat([parent['last_center'], parent['last_pred_size']], -1))
                boxes['fixed_half'] = (boxes['native'] + boxes['mask']) * .5
                boxes.update({arm: torch.cat([value['last_center'], value['last_pred_size']], -1) for arm, value in pair.items()})
                truth = torch.cat([batch['center_label'][:, 0, :3], batch['size_gts'][:, 0]], -1)
                np.savez_compressed(directory / ('batch_%05d.npz' % len(rows)),
                    row_ids=batch['local_training_id'].cpu().numpy(), root_gt=truth.cpu().numpy(),
                    scores=scores.cpu().numpy(), **{name: value.cpu().numpy() for name, value in boxes.items()})
                for arm in ARMS:
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
                    mask = ((alpha * parent['last_pred_masks'][bid][0, query]
                        + (1 - alpha) * parent['sp_last_pred_masks'][bid][query]).sigmoid() > .5)[parent['superpoints'][bid]]
                    target = batch['gt_masks'][bid, 0].bool()
                    mask_iou = float((mask & target).sum().float() / (mask | target).sum())
                    record = dict(row_id=row_id, scan_id=batch['scan_ids'][bid], target_id=int(batch['target_id'][bid]),
                        query=query, root_box=truth[bid].cpu().tolist(), mask_iou=mask_iou,
                        point_sha256=hashlib.sha256(batch['point_clouds'][bid].cpu().numpy().tobytes()).hexdigest(),
                        parent_forwards=1, final_semantic_head_calls=call['final_semantic_head_calls'], controls={}, arms={})
                    for name, value in boxes.items():
                        iou = self.box_iou(value[bid], truth[bid])
                        current = dict(box=value[bid, query].cpu().tolist(), iou=float(iou[query]),
                            oracle25=[int((iou[ranked[:count]] > .25).any()) for count in (16, 32, 64, 256)],
                            oracle50=[int((iou[ranked[:count]] > .5).any()) for count in (16, 32, 64, 256)])
                        if name in ARMS:
                            current.update(axis_gate=pair[name]['span_axis_gate'][bid, query].cpu().tolist(),
                                source_fraction=pair[name]['span_source_fraction'][bid, query].cpu().tolist())
                            record['arms'][name] = current
                        else:
                            record['controls'][name] = current
                    rows.append(record)
                    stream.write(json.dumps(record) + '\n')
                if len(rows) % 512 < 8:
                    stream.flush()
                    print('SPAN_EVAL_PROGRESS ' + json.dumps(dict(rows=len(rows), total=9508,
                        seconds=time.time() - started)), flush=True)
        assert len(rows) == 9508
        metrics = {}
        for arm in ARMS:
            hits = [sum(row['arms'][arm]['iou'] > threshold for row in rows) for threshold in (.25, .5)]
            assert hits[0] == evaluators[arm].dets[('last_', .25, 1, 'bbs')]
            assert hits[1] == evaluators[arm].dets[('last_', .5, 1, 'bbs')]
            assert abs(sum(row['mask_iou'] for row in rows) - float(evaluators[arm].dets['mask_pos'])) < 1e-3
            metrics[arm] = dict(rec_hits25=hits[0], rec_hits50=hits[1],
                repairs25=sum(row['controls']['mask']['iou'] <= .25 < row['arms'][arm]['iou'] for row in rows),
                damages25=sum(row['arms'][arm]['iou'] <= .25 < row['controls']['mask']['iou'] for row in rows),
                repairs50=sum(row['controls']['mask']['iou'] <= .5 < row['arms'][arm]['iou'] for row in rows),
                damages50=sum(row['arms'][arm]['iou'] <= .5 < row['controls']['mask']['iou'] for row in rows))
        receipt = dict(status='pass', rows=9508, formal_rows=9508, metrics=metrics,
            same_forward_controls={name: [sum(row['controls'][name]['iou'] > threshold for row in rows)
                                         for threshold in (.25, .5)] for name in ('native', 'mask', 'fixed_half')},
            mask_hits25=sum(row['mask_iou'] > .25 for row in rows),
            mask_hits50=sum(row['mask_iou'] > .5 for row in rows),
            mask_miou=sum(row['mask_iou'] for row in rows) / 9508 * 100,
            primary_mode='bbs', all256_retained=True, same_selected_query_box_and_mask=True,
            frozen_parent_forward_per_batch=1, elapsed_seconds=time.time() - started,
            rows_sha256=self.sha(directory / 'rows.jsonl'))
        self.write_json(directory / 'receipt.json', receipt)
        print('SPAN_EVAL_COMPLETE ' + json.dumps(receipt), flush=True)

    def run(self, criterion, manifest, started):
        if self.args.mode == 'formal':
            self.restore_terminals()
            self.evaluate()
            self.frozen_state_exact()
            return
        self.dataset.augment = self.dataset.augment_det = True
        self.reset_rng()
        if self.args.mode == 'preflight':
            for index, batch_cpu in enumerate(self.loader('fit', True)):
                if index == self.spec['preflight_batch_index']:
                    break
            assert index == self.spec['preflight_batch_index']
            reference = [json.loads(line) for line in Path(self.spec['preflight_reference_rows']).read_text().splitlines()]
            for bid, row_id in enumerate(batch_cpu['local_training_id'].tolist()):
                row = reference[index * 8 + bid]
                assert row_id == row['row_id']
                assert hashlib.sha256(batch_cpu['point_clouds'][bid].numpy().tobytes()).hexdigest() == row['point_sha256']
                assert torch.cat([batch_cpu['center_label'][bid, 0, :3], batch_cpu['size_gts'][bid, 0]]).tolist() == row['noisy_root_box']
            witnesses = [self.step(batch_cpu, criterion, True) for _ in range(2)]
            restore = self.restore_and_integration_witness(batch_cpu)
            self.write_json(self.output / 'preflight.json', dict(status='pass', optimizer_steps_per_arm=2,
                weight_files_created=0, accuracy_result=False, trainable_parameters_per_arm=29793,
                witnesses=witnesses, restore=restore, separate_optimizers_and_gradients=True,
                parent_and_mask_and_score_frozen=True, torch_version=torch.__version__,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                elapsed_seconds=time.perf_counter() - started, spec_sha256=self.sha(self.args.spec)))
            return
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
                    print('SPAN_TRAIN_PROGRESS ' + json.dumps(record), flush=True)
        assert self.steps == total and Counter(self.seen) == Counter(self.partitions['fit'])
        self.save()
        self.frozen_state_exact()
        self.write_json(self.output / 'receipt.json', dict(status='complete', training_steps_per_arm=3723,
            fit_rows_per_arm=len(self.seen), fit_seen_exactly_once_per_arm=True,
            parent_and_mask_and_score_states_exact=True, new_formal_rows=0,
            no_new_initial_or_holdout_accuracy_claim=True,
            elapsed_seconds=time.perf_counter() - started, spec_sha256=self.sha(self.args.spec)))
