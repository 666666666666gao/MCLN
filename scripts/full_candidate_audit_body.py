    # Only this read-only evaluation runs. The original training/preflight and
    # optimizer branches are absent from the generated script.
    import base64
    torch.set_grad_enabled(False)
    dataset.augment=False;dataset.augment_det=False
    model.eval();reset_rng(spec['seed'])
    before = {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    evaluator=GroundingEvaluator(only_root=True,thresholds=[.25,.5],topks=[1,5,10],
        prefixes=['last_'],filter_non_gt_boxes=False,model='PVGround')
    rows=[];begin=time.time();chunks=0

    def box_iou(boxes, targets):
        lo=torch.maximum(boxes[:,None,:3]-boxes[:,None,3:]/2,targets[None,:,:3]-targets[None,:,3:]/2)
        hi=torch.minimum(boxes[:,None,:3]+boxes[:,None,3:]/2,targets[None,:,:3]+targets[None,:,3:]/2)
        intersection=(hi-lo).clamp(min=0).prod(-1)
        result=intersection/(boxes[:,None,3:].prod(-1)+targets[None,:,3:].prod(-1)-intersection)
        assert bool(torch.isfinite(result).all())
        return result

    with (audit_output/'rows.jsonl').open('w') as stream:
        for batch in loader('holdout',False,spec['seed']):
            inputs,batch=prepare(batch,'eval');inputs['train']=False
            predictions=model(inputs)
            matching=[]
            def capture_match(module, inputs, result):
                matching.append([(q.clone(),t.clone()) for q,t in result])
            hook=set_criterion.matcher.register_forward_hook(capture_match)
            _,predictions=native_loss(predictions,batch)
            hook.remove()
            assert len(matching)==7
            last_matches=matching[1]  # native order: proposal, last, 0head,...
            for key in predictions:
                if 'pred_size' in key:predictions[key]=predictions[key].clamp(min=1e-6)
            evaluator.evaluate(predictions,'last_')
            boxes=torch.cat([predictions['last_center'],predictions['last_pred_size']],-1)
            coarse=torch.cat([predictions['p3_coarse_center'],predictions['p3_coarse_size'].clamp(min=1e-6)],-1)
            probabilities=predictions['last_sem_cls_scores'].softmax(-1)
            scores=(probabilities*(batch['positive_map'][:,0,None]>0)).sum(-1)
            for name in ['modify_positive_map','pron_positive_map','rel_positive_map']:
                scores=scores+(probabilities*batch[name][:,0,None]).sum(-1)
            scores=scores-(probabilities*batch['other_entity_map'][:,0,None]).sum(-1)
            arrays={name:[] for name in ('boxes','coarse_boxes','bbs_scores','root_iou','root_mask_iou',
                'matched_GT_slot','best_scene_GT_id','best_scene_GT_class','best_scene_GT_iou',
                'root_joint_best_scene_overlap','no_object_probability','row_id')}
            for bid in range(len(batch['utterances'])):
                row_id=int(batch['local_training_id'][bid]);assert row_id==len(rows)
                assert bool(batch['box_label_mask'][bid,0])
                root=torch.cat([batch['center_label'][bid,0,:3],batch['size_gts'][bid,0]])
                iou=box_iou(boxes[bid],root[None])[:,0]
                coarse_iou=box_iou(coarse[bid],root[None])[:,0]
                rank=scores[bid].argsort(descending=True);selected=int(rank[0])
                valid_slots=batch['box_label_mask'][bid].bool().nonzero().flatten()
                matched=torch.full((256,),-1,dtype=torch.int16,device=boxes.device)
                queries,targets=last_matches[bid]
                assert int((targets==0).sum())==1 and int(valid_slots[0])==0
                matched[queries]=valid_slots[targets].to(torch.int16)
                scene_ids=batch['all_bbox_label_mask'][bid].bool().nonzero().flatten()
                assert scene_ids.numel()>0
                scene_boxes=batch['all_bboxes'][bid,scene_ids]
                target_id=int(batch['target_id'][bid])
                root_scene_slot=(scene_ids==target_id).nonzero().flatten()
                assert root_scene_slot.numel()==1
                assert torch.equal(scene_boxes[root_scene_slot[0]],root)
                scene_overlap=box_iou(boxes[bid],scene_boxes)
                best_overlap,best_local=scene_overlap.max(-1)
                best_id=scene_ids[best_local]
                best_class=batch['all_class_ids'][bid,best_id]
                root_joint_best=scene_overlap[:,root_scene_slot[0]]==best_overlap

                # Count actual input members per superpoint; avoid Q x 50000
                # point-mask tensors. Keep native sigmoid > .5 exactly.
                point_sp=predictions['superpoints'][bid]
                text=predictions['last_pred_masks'][bid][0]
                query=predictions['sp_last_pred_masks'][bid]
                assert text.shape==query.shape and text.shape[0]==256
                truth=batch['gt_masks'][bid,0].bool()
                assert bool(truth.any())
                member_count=torch.bincount(point_sp,minlength=text.shape[-1]).float()
                root_count=torch.bincount(point_sp[truth],minlength=text.shape[-1]).float()
                alpha=predictions['adaptive_weights'][bid]
                support=(alpha*text+(1-alpha)*query).sigmoid()>.5
                intersection=support.float()@root_count
                union=support.float()@member_count+truth.sum()-intersection
                mask_iou=intersection/union
                assert bool(torch.isfinite(mask_iou).all())
                selected_mask=support[selected][point_sp]
                exact_mask=(selected_mask&truth).sum().float()/(selected_mask|truth).sum()
                assert float(mask_iou[selected])==float(exact_mask)

                first_rank={}
                error_candidates={}
                for threshold,suffix in ((.25,'25'),(.5,'50')):
                    good=iou>threshold
                    positions=good[rank].nonzero().flatten()
                    first_rank[suffix]=int(positions[0])+1 if positions.numel() else None
                    error_candidates[suffix]={
                        'matched_root':int((good&(matched==0)).sum()),
                        'matched_other':int((good&(matched>0)).sum()),
                        'unmatched':int((good&(matched<0)).sum()),
                        'qualified_in_topk':[int(good[rank[:k]].any()) for k in (16,32,64,256)],
                        'qualified_root_overlap_proxy':int((good&root_joint_best).sum())}
                record=dict(row_id=row_id,scan_id=batch['scan_ids'][bid],target_id=target_id,
                    utterance=batch['utterances'][bid],root_box=root.cpu().tolist(),
                    point_sha256=hashlib.sha256(batch['point_clouds'][bid].cpu().numpy().tobytes()).hexdigest(),
                    selected_query=selected,selected_iou=float(iou[selected]),
                    selected_coarse_iou=float(coarse_iou[selected]),selected_mask_iou=float(mask_iou[selected]),
                    selected_matched_slot=int(matched[selected]),
                    selected_best_scene_GT_id=int(best_id[selected]),
                    selected_root_joint_best_overlap=bool(root_joint_best[selected]),
                    valid_native_GT_slots=valid_slots.cpu().tolist(),first_qualified_rank=first_rank,
                    geometric_qualification=error_candidates,candidate_count=256)
                rows.append(record);stream.write(json.dumps(record)+'\n')
                tensors={'boxes':boxes[bid],'coarse_boxes':coarse[bid],'bbs_scores':scores[bid],
                    'root_iou':iou,'root_mask_iou':mask_iou,'matched_GT_slot':matched,
                    'best_scene_GT_id':best_id.to(torch.int16),
                    'best_scene_GT_class':best_class.to(torch.int16),'best_scene_GT_iou':best_overlap,
                    'root_joint_best_scene_overlap':root_joint_best,
                    'no_object_probability':probabilities[bid,:,-1]}
                for name,value in tensors.items():arrays[name].append(value.cpu().numpy())
                arrays['row_id'].append(row_id)
            buffer=io.BytesIO()
            np.savez_compressed(buffer,**{name:np.stack(values) for name,values in arrays.items()})
            # Stream chunks directly to the local archive: no large remote cache.
            print('CANDIDATE_CHUNK '+str(chunks)+' '+base64.b64encode(buffer.getvalue()).decode('ascii'),flush=True)
            chunks+=1
            stream.flush()
            if len(rows)%512<8:print('CANDIDATE_AUDIT_PROGRESS '+json.dumps(dict(rows=len(rows),seconds=time.time()-begin)),flush=True)
            del predictions,inputs,batch
            if len(rows)==args.limit:break
    assert len(rows)==args.limit
    for name,value in model.state_dict().items():assert torch.equal(value.detach().cpu(),before[name]),name
    hits={suffix:sum(r['selected_iou']>threshold for r in rows) for suffix,threshold in [('25',.25),('50',.5)]}
    assert all(hits[suffix]==evaluator.dets[('last_',threshold,1,'bbs')] for suffix,threshold in [('25',.25),('50',.5)])
    assert abs(sum(r['selected_mask_iou'] for r in rows)-float(evaluator.dets['mask_pos']))<1e-3
    receipt=dict(status='pass',time_cst=now(),rows=len(rows),chunks=chunks,rec_hits=hits,
        elapsed_seconds=time.time()-begin,optimizer_updates=0,optimizer_constructed=False,
        all_candidates_retained=256,model_state_unchanged=True,official_evaluator_counts_exact=True,
        parent_arm=spec['support_arm'],terminal_step=int(terminal['step']),
        input_manifest=spec['input_manifest'],rows_sha256=sha(audit_output/'rows.jsonl'),
        evidence_limits=('Read-only full-candidate diagnostic, not a retrained ablation. '
            'Native matcher is applied to evaluation inputs; unmatched status is not physical background identity. '
            'IoU qualification and nearest annotated scene overlap are GT-only geometric proxies, not a deployed selection rule. '
            'All 256 are kept; no new pruning, supervision or optimizer update. Fresh forward may differ numerically from archived formal output.'))
    write_json(audit_output/'receipt.json',receipt)
    print('CANDIDATE_AUDIT_COMPLETE '+json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
