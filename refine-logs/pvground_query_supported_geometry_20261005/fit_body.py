    def step(batch_cpu, update):
        inputs,batch=prepare(batch_cpu,'train')
        head_inputs=[]
        readback_inputs=[]
        if args.mode=='preflight':
            head_hook=geometry_head.register_forward_pre_hook(lambda module,arguments:head_inputs.append(arguments))
            readback_hook=model.boundary_evidence_readback.register_forward_pre_hook(lambda module,arguments:readback_inputs.append(arguments))
        predictions,call=observed_readback_forward(model,inputs)
        if args.mode=='preflight':
            head_hook.remove();readback_hook.remove()
            assert len(head_inputs)==len(readback_inputs)==1
            before_bbs=native_root_bbs(predictions['last_sem_cls_scores'],batch).detach().clone()
            before_masks=[value.detach().clone() for value in (predictions['last_pred_masks']+predictions['sp_last_pred_masks']+predictions['adaptive_weights'])]
        assert torch.equal(predictions['last_semantic_query_before_readback'],predictions['last_semantic_query_after_readback'])
        matches=[]
        hook=set_criterion.matcher.register_forward_hook(
            lambda module,arguments,result:matches.append([(q.clone(),t.clone()) for q,t in result]))
        native,predictions=native_loss(predictions,batch)
        hook.remove()
        assert len(matches)==7
        correction,assignment=semantic_assignment_correction(predictions,batch,matches[1],set_criterion.eos_coef)
        edge,edge_counts=distribution_loss(predictions,batch,matches[1])
        extra,extra_counts,qualified=query_supported_geometry_loss(predictions,batch,matches[1],set_criterion)
        loss=native+correction+(1.0/7)*edge+spec['extra_geometry_weight']*extra
        assert torch.isfinite(loss)
        witness={}
        if args.mode=='preflight':
            assert extra_counts['extra_candidates']>0
            assert 0 in extra_counts['extra_row_counts'] and extra_counts['extra_boundary_outside']>0
            output_gradients=torch.autograd.grad(extra,
                (predictions['last_center'],predictions['last_pred_size'],predictions['boundary_logits']),retain_graph=True)
            for bid,queries in enumerate(qualified):
                exclude=torch.ones(256,dtype=torch.bool,device=queries.device)
                exclude[queries]=False
                for gradient in output_gradients:
                    assert (gradient[bid,exclude]==0).all()
            isolated=torch.autograd.grad(extra,tuple(trainable.values()),retain_graph=True)
            witness=dict(extra_alone_output_gradient=float(isolated[-2].norm()),
                extra_direct_output_gradients_only_qualified=True)
            assert witness['extra_alone_output_gradient']>0
        optimizer.zero_grad()
        loss.backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trainable.values())
        assert all(parameter.grad is None for name,parameter in model.named_parameters() if name not in selected)
        norm=torch.nn.utils.clip_grad_norm_(tuple(trainable.values()),spec['clip_norm'])
        assert torch.isfinite(norm)
        if update:
            optimizer.step()
        if args.mode=='preflight':
            with torch.no_grad():
                center,size=geometry_head(*head_inputs[0])
                predictions['last_center']=center
                predictions['last_pred_size']=size
                replay_query=model.boundary_evidence_readback(*readback_inputs[0])
                assert torch.equal(replay_query,predictions['last_semantic_query_before_readback'])
                replay_logits=model.prediction_heads[-1].sem_cls_scores_head(replay_query.transpose(1,2).contiguous()).transpose(2,1)
                assert torch.equal(native_root_bbs(replay_logits,batch),before_bbs)
                current_masks=predictions['last_pred_masks']+predictions['sp_last_pred_masks']+predictions['adaptive_weights']
                assert all(torch.equal(value,old) for value,old in zip(current_masks,before_masks))
                witness.update(cached_upstream_after_head_update_native_bbs_exact=True,
                    cached_upstream_after_head_update_masks_exact=True,
                    actual_empty_row_and_clipped_target_exercised=True,diagnostic_native_head_replay_calls=1)
        record=dict(loss=float(loss),native_loss=float(native),G_correction=float(correction),
            matched_boundary_loss=float(edge),extra_geometry_loss=float(extra),
            extra_geometry_weight=spec['extra_geometry_weight'],extra_counts=extra_counts,
            matched_boundary_counts=edge_counts,assignment_counts=assignment,gradient_norm=float(norm),
            native_head_calls=call['final_semantic_head_calls'],rows=batch['local_training_id'].cpu().tolist(),
            zero_R_semantic_exact=True,**witness)
        del predictions,inputs,batch
        return record

    if formal:
        evaluate('formal')
        return
    dataset.augment=True
    dataset.augment_det=True
    model.eval()
    geometry_head.train()
    if args.mode=='preflight':
        reset_rng()
        for batch_index,batch_cpu in enumerate(loader('fit',True)):
            if batch_index==spec['preflight_batch_index']:
                break
        assert batch_index==spec['preflight_batch_index']
        witnesses=[step(batch_cpu,True) for _ in range(2)]
        assert all(torch.equal(model.state_dict()[name].detach().cpu(),initial[name]) for name in core_names)
        memory=io.BytesIO()
        torch.save(dict(state_delta={name:value.detach().cpu().clone() for name,value in model.state_dict().items() if name in selected},
            optimizer=optimizer.state_dict()),memory)
        size=memory.tell()
        memory.seek(0)
        restored=torch.load(memory,map_location='cpu')
        model.load_state_dict(dict(initial,**restored['state_delta']),strict=True)
        optimizer.load_state_dict(restored['optimizer'])
        optimizer_check=optimizer_restore_exact(optimizer,restored['optimizer'])
        assert all(int(state['step'])==2 for state in optimizer.state.values())
        assert all(torch.equal(value.detach().cpu(),dict(initial,**restored['state_delta'])[name]) for name,value in model.state_dict().items())
        receipt=dict(status='pass',optimizer_steps=2,weight_files_created=0,accuracy_result=False,
            head_parameters=456102,head_state_tensors=10,all_parent_and_R_states_exact=True,
            isolated_extra_geometry_gradient_verified=True,extra_geometry_weight=spec['extra_geometry_weight'],
            optimizer_exact_check=optimizer_check,serialization_bytes=size,witnesses=witnesses,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            elapsed_seconds=time.perf_counter()-begin,spec_sha256=sha(args.spec),runner_sha256=sha(__file__))
        write_json(output/'preflight.json',receipt)
        print('QUERY_SUPPORTED_GEOMETRY_PREFLIGHT_COMPLETE '+json.dumps(receipt),flush=True)
        return

    initial_rows,initial_receipt=evaluate('initial')
    dataset.augment=True
    dataset.augment_det=True
    model.eval()
    geometry_head.train()
    reset_rng()
    seen=[]
    start=time.time()
    total=math.ceil(len(partitions['fit'])/8)
    assert total==3723

    def save_checkpoint(name,step_number):
        payload=dict(state_delta={name:value.detach().cpu() for name,value in model.state_dict().items() if name in selected},
            optimizer=optimizer.state_dict(),step=step_number,row_ids=seen,head_only=True,
            boundary_mode='distribution',support_arm='whole_range',use_whole_range=True,boundary_loss_weight=1.0/7,
            geometry_parent_fit_updates=3723,total_geometry_fit_updates=3723+step_number,
            checkpoint_sha256=spec['checkpoint_sha256'],base_terminal_sha256=spec['base_terminal_sha256'],
            geometry_terminal_sha256=spec['geometry_terminal_sha256'],source_port_sha256=spec['source_port_sha256'],
            spec_sha256=sha(args.spec),extra_geometry_weight=spec['extra_geometry_weight'],
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            numpy_rng=np.random.get_state(),python_rng=random.getstate())
        temporary=output/(name+'.tmp')
        torch.save(payload,str(temporary))
        os.replace(str(temporary),str(output/name))

    with (output/'train.jsonl').open('w') as stream:
        for index,batch in enumerate(loader('fit',True),1):
            step_begin=time.time()
            record=step(batch,True)
            assert len(record['rows'])==(2 if index==total else 8)
            seen.extend(record['rows'])
            record.update(step=index,total_steps=total,seconds=time.time()-step_begin,cumulative_seconds=time.time()-start)
            stream.write(json.dumps(record)+'\n')
            if index==1 or index%64==0:
                stream.flush()
                print('QUERY_SUPPORTED_GEOMETRY_PROGRESS '+json.dumps(record),flush=True)
            if index%512==0:
                save_checkpoint('latest.pth',index)
    assert index==3723 and Counter(seen)==Counter(partitions['fit'])
    assert all(torch.equal(model.state_dict()[name].detach().cpu(),initial[name]) for name in core_names)
    save_checkpoint('latest.pth',index)
    os.replace(str(output/'latest.pth'),str(output/'terminal.pth'))
    final_rows,final_receipt=evaluate('terminal')
    transitions={}
    for threshold in (.25,.5):
        repairs=damages=0
        for old,new in zip(initial_rows,final_rows):
            assert old['row_id']==new['row_id'] and old['point_sha256']==new['point_sha256'] and old['root_box']==new['root_box']
            repairs+=old['bbs']['iou']<=threshold<new['bbs']['iou']
            damages+=new['bbs']['iou']<=threshold<old['bbs']['iou']
        transitions[str(threshold)]=dict(repairs=repairs,damages=damages,net=repairs-damages)
    receipt=dict(status='complete',training_steps=index,fit_rows=len(seen),holdout_rows=len(final_rows),formal_rows=0,
        initial=initial_receipt['metrics'],terminal=final_receipt['metrics'],transitions=transitions,
        physical_batch=8,effective_batch=8,accumulation=1,last_batch_rows=2,fit_seen_exactly_once=True,
        parent_and_zero_R_states_exact=True,head_parameters=456102,head_state_tensors=10,fresh_optimizer=True,
        extra_geometry_weight=spec['extra_geometry_weight'],primary_mode='bbs',primary_threshold=.5,
        terminal_sha256=sha(output/'terminal.pth'),train_log_sha256=sha(output/'train.jsonl'),spec_sha256=sha(args.spec))
    write_json(output/'receipt.json',receipt)
    print('QUERY_SUPPORTED_GEOMETRY_FIT_COMPLETE '+json.dumps(receipt),flush=True)
