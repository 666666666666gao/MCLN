    # Fixed first eight fit batches, chosen before observing their outcomes.
    assert spec['probe_batches'] == 8 and spec['probe_rows'] == 64
    dataset.augment = True
    dataset.augment_det = True
    model.eval()
    reset_rng()
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    reference = [json.loads(line)['rows'] for line in Path(spec['reference_fit_log']).read_text().splitlines()[:8]]
    assert len(reference) == 8 and all(len(ids) == 8 for ids in reference)
    records = []
    summary = Counter()
    shapes = []
    start = time.time()
    with (output / 'rows.jsonl').open('x') as stream:
        for index, batch_cpu in enumerate(loader('fit', True)):
            row_ids = batch_cpu['local_training_id'].tolist()
            assert row_ids == reference[index]
            inputs, batch = prepare(batch_cpu, 'train')
            captured = []
            hook = set_criterion.matcher.register_forward_hook(
                lambda module, arguments, result: captured.append((arguments, result)))
            with torch.no_grad():
                predictions, call = observed_readback_forward(model, inputs)
                assert torch.equal(predictions['last_semantic_query_before_readback'],
                                   predictions['last_semantic_query_after_readback'])
                native, predictions = native_loss(predictions, batch)
            hook.remove()
            assert len(captured) == 7
            arguments, indices = captured[1]  # Native order: proposal, last, 0head...4head.
            native_outputs, native_targets = arguments
            boxes = torch.cat([predictions['last_center'], predictions['last_pred_size']], -1)
            assert torch.equal(native_outputs['pred_boxes'], boxes)
            assert (boxes[..., 3:] > 0).all()
            # Probe detached output tensors only. No model gradient or update.
            box_leaf = boxes.detach().clone().requires_grad_(True)
            num_boxes = float(sum(len(pair[1]) for pair in indices))
            native_box_losses = set_criterion.loss_boxes(
                dict(native_outputs, pred_boxes=box_leaf), native_targets, indices, num_boxes, None)
            box_gradient = torch.autograd.grad(
                (10*native_box_losses['loss_bbox'] + 2*native_box_losses['loss_giou'])/7, box_leaf)[0]
            boundary_leaf = predictions['boundary_logits'].detach().clone().requires_grad_(True)
            boundary_loss, boundary_counts = distribution_loss(
                dict(predictions, boundary_logits=boundary_leaf), batch, indices)
            boundary_gradient = torch.autograd.grad(boundary_loss/7, boundary_leaf)[0]
            assert torch.isfinite(box_gradient).all() and torch.isfinite(boundary_gradient).all()
            scores = native_root_bbs(predictions['last_sem_cls_scores'], batch)
            arrays = {name: [] for name in ('row_id', 'boxes', 'coarse_boxes', 'root_box', 'bbs',
                'box_iou', 'mask_iou', 'mask_intersection', 'mask_union', 'matched_slot',
                'box_gradient_max', 'boundary_gradient_max', 'max_face_error', 'max_face_change')}
            for bid, row_id in enumerate(row_ids):
                valid = batch['box_label_mask'][bid].bool().nonzero().flatten()
                queries, targets = indices[bid]
                assert int(valid[0]) == 0 and int((targets == 0).sum()) == 1
                truth = torch.cat([batch['center_label'][bid, 0, :3], batch['size_gts'][bid, 0]])
                assert torch.equal(native_targets[bid]['boxes'],
                    torch.cat([batch['center_label'][bid, valid, :3], batch['size_gts'][bid, valid]], -1))
                matched = torch.full((256,), -1, dtype=torch.int16, device=boxes.device)
                matched[queries] = valid[targets].to(torch.int16)
                direct = torch.zeros(256, dtype=torch.bool, device=boxes.device)
                direct[queries] = True
                box_grad = box_gradient[bid].abs().amax(-1)
                edge_grad = boundary_gradient[bid].abs().flatten(1).amax(-1)
                assert (box_grad[~direct] == 0).all() and (edge_grad[~direct] == 0).all()
                iou = box_iou(boxes[bid], truth)
                point_sp = predictions['superpoints'][bid]
                text = predictions['last_pred_masks'][bid][0]
                query = predictions['sp_last_pred_masks'][bid]
                assert text.shape == query.shape and text.shape[0] == 256
                mask_truth = batch['gt_masks'][bid, 0].bool()
                assert mask_truth.any()
                members = torch.bincount(point_sp, minlength=text.shape[-1]).float()
                root_members = torch.bincount(point_sp[mask_truth], minlength=text.shape[-1]).float()
                alpha = predictions['adaptive_weights'][bid]
                support = (alpha*text + (1-alpha)*query).sigmoid() > .5
                intersection = support.float() @ root_members
                union = support.float() @ members + mask_truth.sum() - intersection
                mask_iou = intersection/union
                assert torch.isfinite(mask_iou).all()
                ranked = scores[bid].argsort(descending=True)
                selected = int(ranked[0])
                selected_mask = support[selected][point_sp]
                exact = (selected_mask & mask_truth).sum().float()/(selected_mask | mask_truth).sum()
                assert float(mask_iou[selected]) == float(exact)
                mask_only = (mask_iou > .5) & (iou <= .5)
                both_good = (mask_iou > .5) & (iou > .5)
                coarse = torch.cat([predictions['p3_coarse_center'][bid],
                    predictions['p3_coarse_size'][bid].clamp(min=1e-6)], -1)
                faces = torch.cat([boxes[bid, :, :3]-boxes[bid, :, 3:]/2,
                                   boxes[bid, :, :3]+boxes[bid, :, 3:]/2], -1)
                coarse_faces = torch.cat([coarse[:, :3]-coarse[:, 3:]/2,
                                          coarse[:, :3]+coarse[:, 3:]/2], -1)
                gt_faces = torch.cat([truth[:3]-truth[3:]/2, truth[:3]+truth[3:]/2])
                counts = dict(mask_only=int(mask_only.sum()), both_good=int(both_good.sum()),
                    mask_only_unmatched=int((mask_only & (matched < 0)).sum()),
                    mask_only_matched_root=int((mask_only & (matched == 0)).sum()),
                    mask_only_matched_other=int((mask_only & (matched > 0)).sum()),
                    mask_only_box_gradient_nonzero=int((mask_only & (box_grad > 0)).sum()),
                    mask_only_boundary_gradient_nonzero=int((mask_only & (edge_grad > 0)).sum()),
                    unmatched_good_box=int(((iou > .5) & (matched < 0)).sum()))
                record = dict(row_id=row_id, batch_index=index, scan_id=batch['scan_ids'][bid],
                    target_id=int(batch['target_id'][bid]), utterance=batch['utterances'][bid],
                    language_dataset=batch['language_dataset'][bid], sample_dataset=batch['sample_dataset'][bid],
                    valid_native_GT_slots=valid.cpu().tolist(), native_matched_queries=queries.cpu().tolist(),
                    native_matched_slots=valid[targets].cpu().tolist(),
                    root_box=truth.cpu().tolist(), point_sha256=hashlib.sha256(batch_cpu['point_clouds'][bid].numpy().tobytes()).hexdigest(),
                    selected_query=selected, selected_box_iou=float(iou[selected]),
                    selected_mask_iou=float(mask_iou[selected]), selected_matched_slot=int(matched[selected]),
                    counts=counts, candidates=256,
                    no_direct_box_target_for_unmatched=True, no_direct_boundary_target_for_unmatched=True)
                records.append(record)
                stream.write(json.dumps(record)+'\n')
                summary.update(counts)
                summary['rows_with_mask_only_unmatched'] += counts['mask_only_unmatched'] > 0
                summary['rows_with_both_good'] += counts['both_good'] > 0
                summary['selected_mask_only'] += bool(mask_only[selected])
                summary['selected_mask_only_unmatched'] += bool(mask_only[selected] & (matched[selected] < 0))
                summary['valid_native_GT_slots_total'] += len(valid)
                values = dict(row_id=np.asarray(row_id), boxes=boxes[bid], coarse_boxes=coarse,
                    root_box=truth, bbs=scores[bid], box_iou=iou, mask_iou=mask_iou,
                    mask_intersection=intersection.to(torch.int64), mask_union=union.to(torch.int64),
                    matched_slot=matched, box_gradient_max=box_grad, boundary_gradient_max=edge_grad,
                    max_face_error=(faces-gt_faces).abs().amax(-1),
                    max_face_change=(faces-coarse_faces).abs().amax(-1))
                for key, value in values.items():
                    arrays[key].append(value.cpu().numpy() if torch.is_tensor(value) else value)
            np.savez_compressed(str(output / ('batch_%02d.npz' % index)),
                               **{key: np.stack(values) for key, values in arrays.items()})
            shapes.append(dict(index=index, rows=row_ids, matching_calls=7, native_box_count=num_boxes,
                boundary_counts=boundary_counts, native_head_calls=call['final_semantic_head_calls']))
            stream.flush()
            print('FIT_ROLE_PROBE_BATCH '+json.dumps(dict(index=index, rows=row_ids,
                  totals=dict(summary), elapsed_seconds=time.time()-start)), flush=True)
            del predictions, inputs, batch, captured, box_leaf, boundary_leaf
            if index + 1 == spec['probe_batches']:
                break
    assert len(records) == 64
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(value.detach().cpu(), before[name]) for name, value in model.state_dict().items())
    receipt = dict(status='pass', time_cst=datetime.datetime.now().astimezone().isoformat(),
        rows=len(records), batches=8, candidates=256, totals=dict(summary), batches_metadata=shapes,
        model_state_unchanged=True, model_gradients_absent=True, optimizer_constructed=False,
        optimizer_steps=0, weight_files_created=0, sample_order_matches_completed_fit_first64=True,
        augmentation=dict(points=True, detected_boxes=True), model_mode='eval', geometry_parent_hits=[5616,4506],
        zero_R=True, helper_root=spec['helper_root'], elapsed_seconds=time.perf_counter()-begin,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        spec_sha256=sha(args.spec), runner_sha256=sha(__file__), rows_sha256=sha(output/'rows.jsonl'),
        accuracy_result=False, inference_GT_added=False,
        limitations=('Fixed 64 augmented fit rows and current protected checkpoint; not historical per-step assignments or validation prevalence. '
            'Detached-output gradients measure only final native L1/GIoU and matched boundary-distribution losses. '
            'An unmatched candidate may change through shared parameters or other losses. '
            'Mask and Box IoUs are training-GT diagnostics, not new deployment gates or physical-identity proof.'))
    write_json(output / 'receipt.json', receipt)
    print('FIT_ROLE_PROBE_COMPLETE '+json.dumps(receipt), flush=True)
