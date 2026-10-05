    assert spec['probe_batches']==8 and spec['probe_rows']==64
    geometry_head=model.candidate_box_refiner
    head_keys={name for name in model.state_dict() if name.startswith('candidate_box_refiner.')}
    assert len(head_keys)==10
    assert sum(parameter.numel() for parameter in geometry_head.parameters())==456102
    original={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    heads={'parent':{name:original[name] for name in head_keys}}
    for arm in ('control','query_supported'):
        item=spec['terminal_heads'][arm]
        assert sha(item['path'])==item['sha256']
        payload=torch.load(item['path'],map_location='cpu')
        assert payload['step']==3723 and payload['total_geometry_fit_updates']==7446
        assert payload['head_only'] and payload['extra_geometry_weight']==(0. if arm=='control' else 1.)
        assert payload['geometry_terminal_sha256']==spec['geometry_terminal_sha256']
        assert payload['spec_sha256']==item['spec_sha256']
        assert set(payload['state_delta'])==head_keys
        for key in head_keys:
            assert payload['state_delta'][key].shape==original[key].shape
            assert payload['state_delta'][key].dtype==original[key].dtype
        heads[arm]=payload['state_delta']
        del payload

    def load_head(arm):
        prefix='candidate_box_refiner.'
        geometry_head.load_state_dict({key[len(prefix):]:value for key,value in heads[arm].items()},strict=True)
        geometry_head.eval()

    dataset.augment=True
    dataset.augment_det=True
    reset_rng()
    reference=[json.loads(line) for line in Path(spec['reference_rows']).read_text().splitlines()]
    assert len(reference)==64
    records=[]
    start=time.time()
    with (output/'rows.jsonl').open('x') as stream:
        for index,batch_cpu in enumerate(loader('fit',True)):
            row_ids=batch_cpu['local_training_id'].tolist()
            assert row_ids==[row['row_id'] for row in reference[index*8:(index+1)*8]]
            inputs,batch=prepare(batch_cpu,'train')
            load_head('parent')
            head_inputs=[]
            head_hook=geometry_head.register_forward_pre_hook(lambda module,arguments:head_inputs.append(arguments))
            matches=[]
            matcher_hook=set_criterion.matcher.register_forward_hook(
                lambda module,arguments,result:matches.append([(q.clone(),t.clone()) for q,t in result]))
            with torch.no_grad():
                predictions,call=observed_readback_forward(model,inputs)
                assert torch.equal(predictions['last_semantic_query_before_readback'],
                    predictions['last_semantic_query_after_readback'])
                parent_boxes=torch.cat([predictions['last_center'],predictions['last_pred_size']],-1).clone()
                native,predictions=native_loss(predictions,batch)
            head_hook.remove()
            matcher_hook.remove()
            assert len(head_inputs)==1 and len(matches)==7
            assert call['final_semantic_head_calls']==1
            scores=native_root_bbs(predictions['last_sem_cls_scores'],batch).detach().clone()
            masks=[value.detach().clone() for value in
                predictions['last_pred_masks']+predictions['sp_last_pred_masks']+predictions['adaptive_weights']]
            coarse_center=head_inputs[0][2].detach().clone()
            coarse_size=head_inputs[0][3].detach().clone()
            candidates={}
            with torch.no_grad():
                for arm in ('parent','control','query_supported'):
                    load_head(arm)
                    center,size=geometry_head(*head_inputs[0])
                    boxes=torch.cat([center,size],-1)
                    assert torch.isfinite(boxes).all() and (size>0).all()
                    if arm=='parent':
                        assert torch.equal(boxes,parent_boxes)
                    candidates[arm]=dict(boxes=boxes.detach().clone(),
                        logits=predictions['boundary_logits'].detach().clone())
                    assert torch.equal(native_root_bbs(predictions['last_sem_cls_scores'],batch),scores)
                    assert all(torch.equal(current,old) for current,old in zip(
                        predictions['last_pred_masks']+predictions['sp_last_pred_masks']+predictions['adaptive_weights'],masks))
            arrays={name:[] for name in ('row_id','root_box','coarse_box','bbs','query_intersection','query_union',
                'fused_intersection','fused_union','matched_slot','face_target','outside')}
            for arm in candidates:
                for name in ('boxes','iou','max_face_error','max_face_move','dfl'):
                    arrays[arm+'_'+name]=[]
            for bid,row_id in enumerate(row_ids):
                prior=reference[index*8+bid]
                point_digest=hashlib.sha256(batch_cpu['point_clouds'][bid].numpy().tobytes()).hexdigest()
                assert point_digest==prior['point_sha256'] and batch['scan_ids'][bid]==prior['scan_id']
                valid=batch['box_label_mask'][bid].bool().nonzero().flatten()
                assert int(valid[0])==0
                truth=torch.cat([batch['center_label'][bid,0,:3],batch['size_gts'][bid,0]])
                assert truth.cpu().tolist()==prior['root_box']
                queries,targets=matches[1][bid]
                matched=torch.full((256,),-1,dtype=torch.int16,device=truth.device)
                matched[queries]=valid[targets].to(torch.int16)
                point_sp=predictions['superpoints'][bid]
                text=predictions['last_pred_masks'][bid][0]
                query=predictions['sp_last_pred_masks'][bid]
                alpha=predictions['adaptive_weights'][bid]
                mask_truth=batch['gt_masks'][bid,0].bool()
                members=torch.bincount(point_sp,minlength=text.shape[-1]).float()
                root_members=torch.bincount(point_sp[mask_truth],minlength=text.shape[-1]).float()
                support={}
                values=dict(row_id=np.asarray(row_id),root_box=truth,
                    coarse_box=torch.cat([coarse_center[bid],coarse_size[bid].clamp(min=1e-6)],-1),
                    bbs=scores[bid],matched_slot=matched)
                for branch,logits in (('query',query),('fused',alpha*text+(1-alpha)*query)):
                    mask=(logits.sigmoid()>.5).float()
                    intersection=mask@root_members
                    union=mask@members+mask_truth.sum()-intersection
                    support[branch]=intersection/union
                    assert torch.isfinite(support[branch]).all()
                    values[branch+'_intersection']=intersection.to(torch.int64)
                    values[branch+'_union']=union.to(torch.int64)
                target=face_targets(coarse_center[bid],coarse_size[bid],
                    truth[:3].expand(256,3),truth[3:].expand(256,3)).detach()
                knots=target.new_tensor(KNOTS)
                outside=(target<knots[0])|(target>knots[-1])
                clipped=target.clamp(min=knots[0],max=knots[-1])
                left=(clipped[...,None]>=knots).sum(-1).sub(1).clamp(0,32-1)
                right_weight=(clipped-knots[left])/(knots[left+1]-knots[left])
                values.update(face_target=target,outside=outside)
                truth_faces=torch.cat([truth[:3]-truth[3:]/2,truth[:3]+truth[3:]/2])
                coarse=values['coarse_box']
                coarse_faces=torch.cat([coarse[:,:3]-coarse[:,3:]/2,coarse[:,:3]+coarse[:,3:]/2],-1)
                for arm,result in candidates.items():
                    boxes=result['boxes'][bid]
                    faces=torch.cat([boxes[:,:3]-boxes[:,3:]/2,boxes[:,:3]+boxes[:,3:]/2],-1)
                    log_probability=result['logits'][bid].log_softmax(-1)
                    dfl=-(1-right_weight)*log_probability.gather(-1,left[...,None]).squeeze(-1)
                    dfl-=right_weight*log_probability.gather(-1,(left+1)[...,None]).squeeze(-1)
                    values.update({arm+'_boxes':boxes,arm+'_iou':box_iou(boxes,truth),
                        arm+'_max_face_error':(faces-truth_faces).abs().max(-1).values,
                        arm+'_max_face_move':(faces-coarse_faces).abs().max(-1).values,
                        arm+'_dfl':dfl.mean(-1)})
                qualified=(matched<0)&(support['query']>.5)&(support['fused']>.5)&(values['parent_iou']<=.5)
                selected=int(scores[bid].argmax())
                record=dict(row_id=row_id,batch_index=index,scan_id=prior['scan_id'],point_sha256=point_digest,
                    root_box=truth.cpu().tolist(),valid_native_GT_slots=valid.cpu().tolist(),
                    parent_matched_queries=queries.cpu().tolist(),parent_matched_slots=valid[targets].cpu().tolist(),
                    selected_query=selected,parent_qualified_count=int(qualified.sum()),
                    parent_qualified_outside_count=int((qualified&outside.any(-1)).sum()),
                    selected_ious={arm:float(values[arm+'_iou'][selected]) for arm in candidates},
                    frozen_upstream_and_score=True,native_head_calls=1,head_replays=3)
                records.append(record)
                stream.write(json.dumps(record)+'\n')
                for name,value in values.items():
                    arrays[name].append(value.cpu().numpy() if torch.is_tensor(value) else value)
            np.savez_compressed(str(output/('batch_%02d.npz'%index)),
                **{name:np.stack(values) for name,values in arrays.items()})
            stream.flush()
            print('GEOMETRY_COHORT_BATCH '+json.dumps(dict(index=index,rows=len(records),
                elapsed_seconds=time.time()-start)),flush=True)
            del predictions,inputs,batch,head_inputs,candidates
            if index+1==spec['probe_batches']:
                break
    load_head('parent')
    assert len(records)==64
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(value.detach().cpu(),original[name]) for name,value in model.state_dict().items())
    receipt=dict(status='pass',time_cst=datetime.datetime.now().astimezone().isoformat(),rows=64,batches=8,
        candidates=256,optimizer_steps=0,optimizer_constructed=False,weight_files_created=0,
        model_state_restored=True,model_gradients_absent=True,accuracy_result=False,
        parent_qualification_fixed=True,model_mode='eval',augmentation=dict(points=True,detected_boxes=True),
        full_model_forwards=8,native_semantic_head_calls=8,geometry_head_replays=24,
        frozen_upstream_masks_and_score_exact=True,parent_cached_replay_exact=True,
        elapsed_seconds=time.perf_counter()-begin,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),spec_sha256=sha(args.spec),
        runner_sha256=sha(__file__),rows_sha256=sha(output/'rows.jsonl'),
        limitations='Augmented training64 only, fixed parent qualification/matching cohort; not formal accuracy, physical identity truth or inference GT. Cached head comparison does not establish cross-CUDA full-forward bitwise identity.')
    write_json(output/'receipt.json',receipt)
    print('GEOMETRY_COHORT_COMPLETE '+json.dumps(receipt),flush=True)
