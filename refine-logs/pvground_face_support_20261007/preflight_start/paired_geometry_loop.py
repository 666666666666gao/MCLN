"""Two independent geometry heads, one actual frozen PV/G forward per batch."""
import copy
import datetime
import hashlib
import io
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from face_support_model_factory import build_face_support_model
from mask_reference import reference_bounds_witness
from native_root_bbs import native_root_bbs
from pvground_boundary_box_refiner import distribution_loss
from pvground_semantic_assignment import semantic_assignment_correction
from query_supported_geometry import query_supported_geometry_loss
from readback_preflight_checks import observed_readback_forward
from whole_model_preflight_checks import optimizer_restore_exact


ARMS = ('face_center', 'face_region')
COMMON_TENSORS = ('native_coarse_center', 'native_coarse_size', 'mask_reference_center',
                 'mask_reference_size', 'mask_reference_valid', 'geometry_reference_center',
                 'geometry_reference_size', 'whole_mask_range_evidence', 'last_proj_queries',
                 'last_sem_cls_scores')
COMMON_LISTS = ('last_pred_masks', 'sp_last_pred_masks', 'adaptive_weights')


class PairedGeometryRun:
    def __init__(self, model, config, spec, output, args, initial, core_names,
                 prepare, loader, dataset, partitions, native_loss, reset_rng,
                 box_iou, evaluator_class, sha, write_json, cfg, official_payload,
                 g_payload, reference_payload, data_root):
        self.model, self.config, self.spec = model, config, spec
        self.output, self.args = output, args
        self.initial, self.core_names = initial, core_names
        self.prepare, self.loader = prepare, loader
        self.dataset, self.partitions = dataset, partitions
        self.native_loss, self.reset_rng, self.box_iou = native_loss, reset_rng, box_iou
        self.evaluator_class, self.sha, self.write_json = evaluator_class, sha, write_json
        self.cfg, self.official_payload, self.g_payload = cfg, official_payload, g_payload
        self.reference_payload, self.data_root = reference_payload, data_root
        self.heads = {'face_center': model.candidate_box_refiner}
        self.heads['face_region'] = copy.deepcopy(model.candidate_box_refiner)
        self.heads['face_region'].sampling_mode = 'face_region'
        assert all(torch.equal(value, self.heads['face_region'].state_dict()[name])
                   for name, value in self.heads['face_center'].state_dict().items())
        assert all(first.data_ptr() != second.data_ptr() for first, second in zip(
            self.heads['face_center'].parameters(), self.heads['face_region'].parameters()))
        self.optimizers = {arm: torch.optim.AdamW(head.parameters(), lr=spec['lr'],
            weight_decay=spec['weight_decay']) for arm, head in self.heads.items()}
        self.selected = {'candidate_box_refiner.' + name for name in model.candidate_box_refiner.state_dict()}
        self.steps = 0
        self.seen = []
        self.forward_calls = 0
        readback = self.model.boundary_evidence_readback
        assert torch.count_nonzero(readback.output.weight) == 0
        assert torch.count_nonzero(readback.output.bias) == 0
        assert not any(parameter.requires_grad for parameter in readback.parameters())
        for arm in ARMS:
            assert len(tuple(self.heads[arm].parameters())) == 10
            assert sum(parameter.numel() for parameter in self.heads[arm].parameters()) == 456102

    def forward_pair(self, inputs):
        captured = []
        handle = self.heads['face_center'].register_forward_pre_hook(
            lambda module, arguments: captured.append(arguments))
        first, call = observed_readback_forward(self.model, inputs)
        handle.remove()
        self.forward_calls += 1
        assert len(captured) == 1
        query, points, coarse_center, coarse_size, endpoint_argument = captured[0]
        assert endpoint_argument is first
        assert all(not value.requires_grad for value in (query, points, coarse_center, coarse_size))
        assert torch.equal(first['last_semantic_query_before_readback'],
                           first['last_semantic_query_after_readback'])
        # The frozen zero-output R retains an autograd edge to head A. Native
        # scoring is provably unchanged, so detach this common frozen score;
        # otherwise B's criterion could backward through A's already-used graph.
        first['last_sem_cls_scores'] = first['last_sem_cls_scores'].detach()
        second = dict(first)
        center, size = self.heads['face_region'](query, points, coarse_center, coarse_size, second)
        second['last_center'], second['last_pred_size'] = center, size
        for key in COMMON_TENSORS:
            assert torch.equal(first[key], second[key]), key
            assert not first[key].requires_grad and not second[key].requires_grad, key
        for key in COMMON_LISTS:
            assert len(first[key]) == len(second[key])
            assert all(torch.equal(a, b) and not a.requires_grad and not b.requires_grad
                       for a, b in zip(first[key], second[key])), key
        return dict(face_center=first, face_region=second), call, captured[0]

    def loss(self, predictions, batch, set_criterion):
        matches = []
        hook = set_criterion.matcher.register_forward_hook(lambda module, arguments, result:
            matches.append([(queries.clone(), targets.clone()) for queries, targets in result]))
        native, predictions = self.native_loss(predictions, batch)
        hook.remove()
        assert len(matches) == 7
        correction, assignment = semantic_assignment_correction(
            predictions, batch, matches[1], set_criterion.eos_coef)
        edge, edge_counts = distribution_loss(predictions, batch, matches[1])
        roots = torch.cat([batch['center_label'][:, 0, :3], batch['size_gts'][:, 0]], -1)
        extra, extra_counts, qualified = query_supported_geometry_loss(
            predictions, batch, matches[1], set_criterion, roots)
        loss = native + correction + edge / 7 + self.spec['extra_geometry_weight'] * extra
        assert torch.isfinite(loss)
        record = dict(loss=float(loss), native_loss=float(native), G_correction=float(correction),
            matched_boundary_loss=float(edge), extra_geometry_loss=float(extra),
            extra_counts=extra_counts, matched_boundary_counts=edge_counts,
            assignment_counts=assignment, reference_keep_weight=0.0)
        return loss, record, qualified, extra

    def member_witness(self, cached):
        """Independent rectangle distances on real input, not a toy replacement."""
        query, points, _, _, predictions = cached
        center = predictions['geometry_reference_center'].detach()
        size = predictions['geometry_reference_size'].detach().clamp(min=1e-6)
        layout = center[:, :, None] + .5 * size[:, :, None] * self.heads['face_center'].locations
        flat = layout.reshape(len(query), 256 * 7, 3)
        old_indices, old_distances = self.heads['face_center'].nearest_members(points[..., :3], flat)
        new_indices, new_distances = self.heads['face_region'].nearest_members(points[..., :3], flat)
        old_indices, old_distances = old_indices.reshape(len(query), 256, 7, 16), old_distances.reshape(len(query), 256, 7, 16)
        new_indices, new_distances = new_indices.reshape(len(query), 256, 7, 16), new_distances.reshape(len(query), 256, 7, 16)
        assert torch.equal(old_indices[:, :, 0], new_indices[:, :, 0])
        assert torch.equal(old_distances[:, :, 0], new_distances[:, :, 0])
        xyz = points[..., :3].detach().cpu().numpy().astype(np.float64)
        locations = layout.detach().cpu().numpy().astype(np.float64)
        chosen = new_indices.cpu().numpy()
        sizes = size.cpu().numpy().astype(np.float64)
        distance_error = 0.0
        checked = 0
        for bid in range(len(query)):
            for candidate in range(0, 256, 32):
                for face, axis in ((1, 0), (2, 0), (3, 1), (4, 1), (5, 2), (6, 2)):
                    # Project every raw point onto the bounded rectangle.
                    projected = xyz[bid].copy()
                    projected[:, axis] = locations[bid, candidate, face, axis]
                    for tangent in range(3):
                        if tangent != axis:
                            low = locations[bid, candidate, 0, tangent] - .5 * sizes[bid, candidate, tangent]
                            high = locations[bid, candidate, 0, tangent] + .5 * sizes[bid, candidate, tangent]
                            projected[:, tangent] = np.clip(projected[:, tangent], low, high)
                    squared = np.square(xyz[bid] - projected).sum(-1)
                    selected = chosen[bid, candidate, face]
                    # Float64 projection versus actual float32 CUDA arithmetic.
                    assert squared[selected].max() <= np.partition(squared, 15)[15] + 1e-6
                    expected = np.linalg.norm(xyz[bid, selected] - locations[bid, candidate, face], axis=-1)
                    observed = new_distances[bid, candidate, face].cpu().numpy()
                    error = float(np.abs(expected - observed).max())
                    assert error <= 1e-5
                    distance_error = max(distance_error, error)
                    checked += 1
        return dict(center_members_exact=True, independent_real_rectangle_checks=checked,
            encoded_euclidean_distance_max_error=distance_error,
            face_member_indices_different=int((old_indices[:, :, 1:] != new_indices[:, :, 1:]).sum()),
            no_gt_sampling=True, sampling_does_not_guarantee_uniform_face_coverage=True)

    def step(self, batch_cpu, set_criterion, preflight):
        inputs, batch = self.prepare(batch_cpu, 'train')
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        predictions, call, cached = self.forward_pair(inputs)
        witnesses = {}
        if preflight:
            witnesses.update(self.member_witness(cached))
            witnesses.update(reference_bounds_witness(predictions['face_center'], batch))
        records = {}
        for arm in ARMS:
            other = ARMS[1 - ARMS.index(arm)]
            head, optimizer = self.heads[arm], self.optimizers[arm]
            if preflight:
                other_before = {key: value.detach().clone() for key, value in self.heads[other].state_dict().items()}
            loss, record, qualified, extra = self.loss(predictions[arm], batch, set_criterion)
            if preflight and self.steps == 0:
                assert torch.equal(predictions[arm]['last_center'], predictions[arm]['geometry_reference_center'])
                assert torch.equal(predictions[arm]['last_pred_size'], predictions[arm]['geometry_reference_size'].clamp(min=1e-6))
                record['neutral_decode_equals_reference'] = True
            if preflight:
                gradients = torch.autograd.grad(extra + predictions[arm]['boundary_logits'].sum() * 0,
                    (predictions[arm]['last_center'], predictions[arm]['last_pred_size'], predictions[arm]['boundary_logits']),
                    retain_graph=True)
                for bid, queries in enumerate(qualified):
                    outside = torch.ones(256, dtype=torch.bool, device=queries.device)
                    outside[queries] = False
                    assert all((gradient[bid, outside] == 0).all() for gradient in gradients)
                record['extra_direct_output_gradients_only_qualified'] = True
            loss.backward()
            assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in head.parameters())
            assert all(parameter.grad is None for parameter in self.heads[other].parameters())
            assert all(parameter.grad is None for name, parameter in self.model.named_parameters()
                       if name not in self.selected)
            if preflight:
                record['raw_parameter_gradient_norms'] = {name: float(parameter.grad.norm()) for name, parameter in head.named_parameters()}
            norm = torch.nn.utils.clip_grad_norm_(tuple(head.parameters()), self.spec['clip_norm'])
            assert torch.isfinite(norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if preflight:
                assert all(torch.equal(value, self.heads[other].state_dict()[key]) for key, value in other_before.items())
            record.update(sampling_mode=arm, gradient_norm=float(norm), cross_head_gradients_absent=True)
            records[arm] = record
        self.steps += 1
        if preflight:
            score = native_root_bbs(predictions['face_center']['last_sem_cls_scores'], batch)
            assert torch.equal(score, native_root_bbs(predictions['face_region']['last_sem_cls_scores'], batch))
            assert all(torch.equal(self.model.state_dict()[name].detach().cpu(), self.initial[name]) for name in self.core_names)
        record = dict(step=self.steps, rows=batch['local_training_id'].cpu().tolist(), arms=records,
            frozen_parent_forwards_per_batch=1, native_final_semantic_head_calls=call['final_semantic_head_calls'],
            zero_R_semantic_exact=True, shared_native_score_and_masks_exact=True, **witnesses)
        del predictions, cached, inputs, batch
        return record

    def payload(self, arm):
        return dict(state_delta={'candidate_box_refiner.' + name: value.detach().cpu().clone()
                                for name, value in self.heads[arm].state_dict().items()},
            optimizer=self.optimizers[arm].state_dict(), sampling_mode=arm, step=self.steps,
            row_ids=list(self.seen), head_only=True, reference_mode='fused_mask',
            reference_keep_weight=0.0, extra_geometry_weight=1.0, boundary_loss_weight=1.0 / 7,
            common_output_reset=['output.weight', 'output.bias'], retained_hidden_prior_updates=11169,
            retained_hidden_total_updates=11169 + self.steps, reset_output_total_updates=self.steps,
            spec_sha256=self.sha(self.args.spec), sampler_sha256=self.spec['sampler_sha256'],
            checkpoint_sha256=self.spec['checkpoint_sha256'], base_terminal_sha256=self.spec['base_terminal_sha256'],
            selected_terminal_sha256=self.spec['selected_terminal_sha256'], source_port_sha256=self.spec['source_port_sha256'],
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            numpy_rng=np.random.get_state(), python_rng=__import__('random').getstate())

    def save(self, name):
        for arm in ARMS:
            temporary = self.output / arm / (name + '.tmp')
            torch.save(self.payload(arm), str(temporary))
            os.replace(str(temporary), str(self.output / arm / name))

    def restore_terminals(self):
        for arm in ARMS:
            path = self.output / arm / 'terminal.pth'
            payload = torch.load(str(path), map_location='cpu')
            assert payload['step'] == 3723 and payload['sampling_mode'] == arm
            assert payload['spec_sha256'] == self.sha(self.args.spec)
            assert payload['sampler_sha256'] == self.spec['sampler_sha256']
            assert payload['reference_keep_weight'] == 0.0 and payload['extra_geometry_weight'] == 1.0
            for key in ('checkpoint_sha256', 'base_terminal_sha256', 'selected_terminal_sha256', 'source_port_sha256'):
                assert payload[key] == self.spec[key]
            assert Counter(payload['row_ids']) == Counter(self.partitions['fit_saved'])
            assert set(payload['state_delta']) == self.selected
            delta = {name[len('candidate_box_refiner.'):]: value for name, value in payload['state_delta'].items()}
            self.heads[arm].load_state_dict(delta, strict=True)
            self.optimizers[arm].load_state_dict(payload['optimizer'])
            assert all(int(state['step']) == 3723 for state in self.optimizers[arm].state.values())
            self.write_json(self.output / arm / 'formal_restore.json', dict(status='pass',
                terminal_sha256=self.sha(path), sampling_mode=arm,
                optimizer=optimizer_restore_exact(self.optimizers[arm], payload['optimizer'])))
        self.steps = 3723

    def cpu_restore_witness(self):
        checks = {}
        for arm in ARMS:
            memory = io.BytesIO()
            torch.save(self.payload(arm), memory)
            size = memory.tell()
            memory.seek(0)
            restored = torch.load(memory, map_location='cpu')
            # Reconstruct the unchanged frozen zero-R hidden initialization in
            # the exact original constructor RNG sequence; no probe state carries
            # into the separate formal process.
            self.reset_rng()
            rebuilt, _, receipt = build_face_support_model(self.cfg, self.official_payload,
                self.g_payload, self.reference_payload, restored, self.data_root)
            expected = dict(self.initial, **restored['state_delta'])
            assert set(rebuilt.state_dict()) == set(expected)
            assert all(torch.equal(value.detach().cpu(), expected[name]) for name, value in rebuilt.state_dict().items())
            assert rebuilt.candidate_box_refiner.sampling_mode == arm
            cpu_optimizer = torch.optim.AdamW(rebuilt.candidate_box_refiner.parameters(),
                lr=self.spec['lr'], weight_decay=self.spec['weight_decay'])
            cpu_optimizer.load_state_dict(restored['optimizer'])
            optimizer = optimizer_restore_exact(cpu_optimizer, restored['optimizer'])
            assert all(int(state['step']) == 2 for state in cpu_optimizer.state.values())
            checks[arm] = dict(full_cpu_model_exact=True, declared_sampler_restored=True,
                optimizer=optimizer, serialization_bytes=size, weight_files_created=0,
                full_state_tensors=receipt['full_state_tensors'], deployed_geometry_heads=1)
            del rebuilt, cpu_optimizer, restored, memory
        return checks

    @torch.no_grad()
    def evaluate(self, stage, formal):
        self.dataset.augment = self.dataset.augment_det = False
        self.model.eval()
        for head in self.heads.values():
            head.eval()
        self.reset_rng()
        directory = self.output / stage
        directory.mkdir()
        evaluators = {arm: self.evaluator_class(only_root=True, thresholds=[.25, .5], topks=[1, 5, 10],
            prefixes=['last_'], filter_non_gt_boxes=False, model='PVGround') for arm in ARMS}
        rows = []
        start = time.time()
        with (directory / 'rows.jsonl').open('w') as stream:
            for batch in self.loader('holdout', False):
                inputs, batch = self.prepare(batch, 'eval')
                pair, call, cached = self.forward_pair(inputs)
                first = pair['face_center']
                scores = native_root_bbs(first['last_sem_cls_scores'], batch)
                finals = {arm: torch.cat([value['last_center'], value['last_pred_size']], -1) for arm, value in pair.items()}
                reference = torch.cat([first['geometry_reference_center'], first['geometry_reference_size'].clamp(min=1e-6)], -1)
                coarse = torch.cat([first['native_coarse_center'], first['native_coarse_size'].clamp(min=1e-6)], -1)
                truth = torch.cat([batch['center_label'][:, 0, :3], batch['size_gts'][:, 0]], -1)
                if formal:
                    np.savez_compressed(directory / ('batch_%05d.npz' % len(rows)),
                        row_ids=batch['local_training_id'].cpu().numpy(), root_gt=truth.cpu().numpy(),
                        original_prior=coarse.cpu().numpy(), reference=reference.cpu().numpy(),
                        reference_valid=first['mask_reference_valid'].cpu().numpy(), scores=scores.cpu().numpy(),
                        final_face_center=finals['face_center'].cpu().numpy(),
                        final_face_region=finals['face_region'].cpu().numpy())
                for arm in ARMS:
                    assert (finals[arm][..., 3:] > 0).all()
                    _, pair[arm] = self.native_loss(pair[arm], batch)
                    for key in pair[arm]:
                        if 'pred_size' in key:
                            pair[arm][key] = pair[arm][key].clamp(min=1e-6)
                    evaluators[arm].evaluate(pair[arm], 'last_')
                for bid in range(len(batch['utterances'])):
                    row_id = int(batch['local_training_id'][bid])
                    assert row_id == self.partitions['holdout'][len(rows)]
                    ranked = scores[bid].argsort(descending=True)
                    query = int(ranked[0])
                    alpha = first['adaptive_weights'][bid]
                    mask = ((alpha * first['last_pred_masks'][bid][0, query]
                        + (1 - alpha) * first['sp_last_pred_masks'][bid][query]).sigmoid() > .5)[first['superpoints'][bid]]
                    target_mask = batch['gt_masks'][bid, 0].bool()
                    mask_iou = float((mask & target_mask).sum().float() / (mask | target_mask).sum())
                    original_iou, reference_iou = self.box_iou(coarse[bid], truth[bid]), self.box_iou(reference[bid], truth[bid])
                    record = dict(row_id=row_id, scan_id=batch['scan_ids'][bid], target_id=int(batch['target_id'][bid]),
                        root_box=truth[bid].cpu().tolist(), query=query, mask_iou=mask_iou,
                        point_sha256=hashlib.sha256(batch['point_clouds'][bid].cpu().numpy().tobytes()).hexdigest(),
                        coarse_box=coarse[bid, query].cpu().tolist(), coarse_iou=float(original_iou[query]),
                        reference_box=reference[bid, query].cpu().tolist(), reference_iou=float(reference_iou[query]),
                        reference_valid=bool(first['mask_reference_valid'][bid, query]),
                        frozen_parent_forwards=1, native_head_calls=call['final_semantic_head_calls'], arms={})
                    for arm in ARMS:
                        iou = self.box_iou(finals[arm][bid], truth[bid])
                        distances = pair[arm]['p3_neighbor_distances'][bid].flatten().float()
                        record['arms'][arm] = dict(box=finals[arm][bid, query].cpu().tolist(), iou=float(iou[query]),
                            oracle25=[int((iou[ranked[:count]] > .25).any()) for count in (16, 32, 64, 256)],
                            oracle50=[int((iou[ranked[:count]] > .5).any()) for count in (16, 32, 64, 256)],
                            neighbor_distance_mean=float(distances.mean()), neighbor_distance_max=float(distances.max()))
                    rows.append(record)
                    stream.write(json.dumps(record) + '\n')
                if len(rows) % 512 < 8:
                    stream.flush()
                    print('PAIRED_FACE_EVAL_PROGRESS ' + json.dumps(dict(stage=stage, rows=len(rows),
                        total=len(self.partitions['holdout']), seconds=time.time() - start)), flush=True)
                del pair, first, cached, inputs, batch
        assert len(rows) == len(self.partitions['holdout'])
        metrics = {}
        for arm in ARMS:
            hits25, hits50 = (sum(row['arms'][arm]['iou'] > threshold for row in rows) for threshold in (.25, .5))
            assert hits25 == evaluators[arm].dets[('last_', .25, 1, 'bbs')]
            assert hits50 == evaluators[arm].dets[('last_', .5, 1, 'bbs')]
            mask_sum = sum(row['mask_iou'] for row in rows)
            assert abs(mask_sum - float(evaluators[arm].dets['mask_pos'])) < 1e-3
            metrics[arm] = dict(rec_hits25=hits25, rec_hits50=hits50,
                mask_hits25=sum(row['mask_iou'] > .25 for row in rows),
                mask_hits50=sum(row['mask_iou'] > .5 for row in rows), mask_miou=mask_sum / len(rows) * 100)
        receipt = dict(status='pass', stage=stage, rows=len(rows), metrics=metrics,
            formal_rows=len(rows) if formal else 0, shared_parent_forward_per_batch=1,
            native_selected_query_and_masks_shared=True, primary_mode='bbs',
            same_forward_reference_hits25=sum(row['reference_iou'] > .25 for row in rows),
            same_forward_reference_hits50=sum(row['reference_iou'] > .5 for row in rows),
            elapsed_seconds=time.time() - start, rows_sha256=self.sha(directory / 'rows.jsonl'))
        self.write_json(directory / 'receipt.json', receipt)
        print('PAIRED_FACE_EVAL_COMPLETE ' + json.dumps(receipt), flush=True)
        return rows, receipt

    def run(self, set_criterion, manifest, started):
        if self.args.mode in ('initial_formal', 'formal'):
            if self.args.mode == 'formal':
                self.restore_terminals()
            self.evaluate(self.args.mode, True)
            # Both neutral head states already exist in the protected parent;
            # do not create duplicate zero-update weight files.
            return
        if self.args.mode == 'preflight':
            self.dataset.augment = self.dataset.augment_det = True
            self.model.eval()
            for head in self.heads.values():
                head.train()
            self.reset_rng()
            for batch_index, batch_cpu in enumerate(self.loader('fit', True)):
                if batch_index == self.spec['preflight_batch_index']:
                    break
            assert batch_index == self.spec['preflight_batch_index']
            reference_rows = [json.loads(line) for line in Path(self.spec['preflight_reference_rows']).read_text().splitlines()]
            for bid, row_id in enumerate(batch_cpu['local_training_id'].tolist()):
                row = reference_rows[batch_index * 8 + bid]
                assert row_id == row['row_id']
                assert hashlib.sha256(batch_cpu['point_clouds'][bid].numpy().tobytes()).hexdigest() == row['point_sha256']
                assert torch.cat([batch_cpu['center_label'][bid, 0, :3], batch_cpu['size_gts'][bid, 0]]).tolist() == row['noisy_root_box']
            witness = [self.step(batch_cpu, set_criterion, True) for _ in range(2)]
            for arm in ARMS:
                assert witness[0]['arms'][arm]['neutral_decode_equals_reference']
                assert all(value > 0 for value in witness[1]['arms'][arm]['raw_parameter_gradient_norms'].values())
            restore = self.cpu_restore_witness()
            receipt = dict(status='pass', optimizer_steps_per_arm=2, frozen_parent_forward_calls=self.forward_calls,
                weight_files_created=0, accuracy_result=False, witnesses=witness, cpu_restore=restore,
                same_forward_native_score_mask_reference_exact=True, separate_optimizers_and_gradients=True,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                elapsed_seconds=time.perf_counter() - started, spec_sha256=self.sha(self.args.spec))
            self.write_json(self.output / 'preflight.json', receipt)
            print('PAIRED_FACE_PREFLIGHT_COMPLETE ' + json.dumps(receipt), flush=True)
            return
        initial_rows, initial_receipt = self.evaluate('initial', False)
        self.dataset.augment = self.dataset.augment_det = True
        self.model.eval()
        for head in self.heads.values():
            head.train()
        self.reset_rng()
        start = time.time()
        total = math.ceil(len(self.partitions['fit']) / 8)
        assert total == 3723
        with (self.output / 'train.jsonl').open('w') as stream:
            for index, batch_cpu in enumerate(self.loader('fit', True), 1):
                step_begin = time.time()
                record = self.step(batch_cpu, set_criterion, False)
                assert len(record['rows']) == (2 if index == total else 8)
                self.seen.extend(record['rows'])
                record.update(total_steps=total, seconds=time.time() - step_begin, cumulative_seconds=time.time() - start)
                stream.write(json.dumps(record) + '\n')
                if index == 1 or index % 64 == 0:
                    stream.flush()
                    print('PAIRED_FACE_TRAIN_PROGRESS ' + json.dumps(record), flush=True)
                if index % 512 == 0:
                    self.save('latest.pth')
        assert self.steps == 3723 and Counter(self.seen) == Counter(self.partitions['fit'])
        assert all(torch.equal(self.model.state_dict()[name].detach().cpu(), self.initial[name]) for name in self.core_names)
        self.save('latest.pth')
        for arm in ARMS:
            os.replace(str(self.output / arm / 'latest.pth'), str(self.output / arm / 'terminal.pth'))
        final_rows, final_receipt = self.evaluate('terminal', False)
        transitions = {}
        for arm in ARMS:
            transitions[arm] = {}
            for threshold in (.25, .5):
                repairs = damages = 0
                for old, new in zip(initial_rows, final_rows):
                    assert old['row_id'] == new['row_id'] and old['point_sha256'] == new['point_sha256'] and old['root_box'] == new['root_box']
                    repairs += old['arms'][arm]['iou'] <= threshold < new['arms'][arm]['iou']
                    damages += new['arms'][arm]['iou'] <= threshold < old['arms'][arm]['iou']
                transitions[arm][str(threshold)] = dict(repairs=repairs, damages=damages, net=repairs - damages)
        receipt = dict(status='complete', training_steps_per_arm=3723, fit_rows_per_arm=len(self.seen),
            holdout_rows=len(final_rows), formal_rows=0, initial=initial_receipt['metrics'], terminal=final_receipt['metrics'],
            transitions=transitions, physical_batch=8, effective_batch=8, accumulation=1,
            last_batch_rows=2, fit_seen_exactly_once_per_arm=True, parent_and_zero_R_states_exact=True,
            head_parameters_per_arm=456102, trainable_heads_in_experiment=2, deployed_heads_per_model=1,
            reference_keep_weight=0.0, extra_geometry_weight=1.0,
            terminal_sha256={arm: self.sha(self.output / arm / 'terminal.pth') for arm in ARMS},
            train_log_sha256=self.sha(self.output / 'train.jsonl'), spec_sha256=self.sha(self.args.spec))
        self.write_json(self.output / 'receipt.json', receipt)
        print('PAIRED_FACE_FIT_COMPLETE ' + json.dumps(receipt), flush=True)
