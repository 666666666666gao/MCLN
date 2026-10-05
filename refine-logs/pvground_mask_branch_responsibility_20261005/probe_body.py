    assert spec['probe_batches']==8 and spec['probe_rows']==64
    dataset.augment=True
    dataset.augment_det=True
    model.eval()
    reset_rng()
    before={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    reference=[json.loads(line) for line in Path(spec['reference_rows']).read_text().splitlines()]
    assert len(reference)==64
    records=[]
    totals=Counter()
    start=time.time()
    with (output/'rows.jsonl').open('x') as stream:
        for index,batch_cpu in enumerate(loader('fit',True)):
            row_ids=batch_cpu['local_training_id'].tolist()
            assert row_ids==[row['row_id'] for row in reference[index*8:(index+1)*8]]
            inputs,batch=prepare(batch_cpu,'train')
            captured=[]
            hook=set_criterion.matcher.register_forward_hook(
                lambda module,arguments,result:captured.append((arguments,result)))
            with torch.no_grad():
                predictions,call=observed_readback_forward(model,inputs)
                assert torch.equal(predictions['last_semantic_query_before_readback'],
                    predictions['last_semantic_query_after_readback'])
                native,predictions=native_loss(predictions,batch)
            hook.remove()
            assert len(captured)==7
            arguments,indices=captured[1]
            native_outputs,native_targets=arguments
            boxes=torch.cat([predictions['last_center'],predictions['last_pred_size']],-1)
            assert torch.equal(native_outputs['pred_boxes'],boxes)
            scores=native_root_bbs(predictions['last_sem_cls_scores'],batch)
            arrays={name:[] for name in ('row_id','boxes','root_box','bbs','box_iou',
                'text_iou','query_iou','fused_iou','text_intersection','text_union',
                'query_intersection','query_union','fused_intersection','fused_union',
                'matched_slot','alpha')}
            for bid,row_id in enumerate(row_ids):
                prior=reference[index*8+bid]
                point_digest=hashlib.sha256(batch_cpu['point_clouds'][bid].numpy().tobytes()).hexdigest()
                assert point_digest==prior['point_sha256']
                valid=batch['box_label_mask'][bid].bool().nonzero().flatten()
                queries,targets=indices[bid]
                assert int(valid[0])==0
                truth=torch.cat([batch['center_label'][bid,0,:3],batch['size_gts'][bid,0]])
                assert truth.cpu().tolist()==prior['root_box']
                assert batch['scan_ids'][bid]==prior['scan_id']
                assert torch.equal(native_targets[bid]['boxes'],
                    torch.cat([batch['center_label'][bid,valid,:3],batch['size_gts'][bid,valid]],-1))
                matched=torch.full((256,),-1,dtype=torch.int16,device=boxes.device)
                matched[queries]=valid[targets].to(torch.int16)
                point_sp=predictions['superpoints'][bid]
                text=predictions['last_pred_masks'][bid][0]
                query=predictions['sp_last_pred_masks'][bid]
                assert text.shape==query.shape and text.shape[0]==256
                assert torch.equal(text,text[:1].expand_as(text))
                alpha=predictions['adaptive_weights'][bid]
                mask_truth=batch['gt_masks'][bid,0].bool()
                members=torch.bincount(point_sp,minlength=text.shape[-1]).float()
                root_members=torch.bincount(point_sp[mask_truth],minlength=text.shape[-1]).float()
                supports={'text':text.sigmoid()>.5,'query':query.sigmoid()>.5,
                    'fused':(alpha*text+(1-alpha)*query).sigmoid()>.5}
                selected=int(scores[bid].argmax())
                values=dict(row_id=np.asarray(row_id),boxes=boxes[bid],root_box=truth,
                    bbs=scores[bid],box_iou=box_iou(boxes[bid],truth),matched_slot=matched,
                    alpha=alpha)
                for branch,support in supports.items():
                    intersection=support.float()@root_members
                    union=support.float()@members+mask_truth.sum()-intersection
                    iou=intersection/union
                    assert torch.isfinite(iou).all()
                    selected_mask=support[selected][point_sp]
                    exact=(selected_mask&mask_truth).sum().float()/(selected_mask|mask_truth).sum()
                    assert float(iou[selected])==float(exact)
                    values[branch+'_intersection']=intersection.to(torch.int64)
                    values[branch+'_union']=union.to(torch.int64)
                    values[branch+'_iou']=iou
                fused_only=(values['fused_iou']>.5)&(values['box_iou']<=.5)&(matched<0)
                own_query=values['query_iou']>.5
                text_good=values['text_iou']>.5
                counts=dict(fused_mask_only_unmatched=int(fused_only.sum()),
                    own_query_confirmed=int((fused_only&own_query).sum()),
                    query_not_qualified_text_qualified=int((fused_only&~own_query&text_good).sum()),
                    neither_branch_qualified=int((fused_only&~own_query&~text_good).sum()),
                    own_query_mask_only_unmatched=int((own_query&(values['box_iou']<=.5)&(matched<0)).sum()))
                record=dict(row_id=row_id,batch_index=index,scan_id=prior['scan_id'],
                    point_sha256=point_digest,root_box=truth.cpu().tolist(),
                    valid_native_GT_slots=valid.cpu().tolist(),selected_query=selected,
                    selected_box_iou=float(values['box_iou'][selected]),
                    selected_branch_ious={branch:float(values[branch+'_iou'][selected]) for branch in supports},
                    native_matched_queries=queries.cpu().tolist(),native_matched_slots=valid[targets].cpu().tolist(),
                    alpha=alpha.cpu().tolist(),counts=counts,candidates=256,text_repeated_all_queries=True)
                records.append(record)
                stream.write(json.dumps(record)+'\n')
                totals.update(counts)
                totals['rows_with_own_query_confirmed']+=counts['own_query_confirmed']>0
                for key,value in values.items():
                    arrays[key].append(value.cpu().numpy() if torch.is_tensor(value) else value)
            np.savez_compressed(str(output/('batch_%02d.npz'%index)),
                **{name:np.stack(values) for name,values in arrays.items()})
            stream.flush()
            print('MASK_BRANCH_BATCH '+json.dumps(dict(index=index,totals=dict(totals),
                elapsed_seconds=time.time()-start)),flush=True)
            del predictions,inputs,batch,captured
            if index+1==spec['probe_batches']:
                break
    assert len(records)==64
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(value.detach().cpu(),before[name]) for name,value in model.state_dict().items())
    receipt=dict(status='pass',time_cst=datetime.datetime.now().astimezone().isoformat(),rows=64,batches=8,
        candidates=256,totals=dict(totals),optimizer_steps=0,optimizer_constructed=False,
        weight_files_created=0,model_state_unchanged=True,model_gradients_absent=True,
        accuracy_result=False,geometry_parent_hits=[5616,4506],all_text_logits_repeated=True,
        point_GT_row_identity_matches_closed_role_probe=True,model_mode='eval',
        augmentation=dict(points=True,detected_boxes=True),elapsed_seconds=time.perf_counter()-begin,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        spec_sha256=sha(args.spec),runner_sha256=sha(__file__),rows_sha256=sha(output/'rows.jsonl'),
        limitations='Fixed augmentedfit64, training-GT support proxy only. Separate Query support is not physical identity proof. Independent CUDA forwards can differ from the prior probe. No new geometry loss or inference GT gate.')
    write_json(output/'receipt.json',receipt)
    print('MASK_BRANCH_COMPLETE '+json.dumps(receipt),flush=True)
