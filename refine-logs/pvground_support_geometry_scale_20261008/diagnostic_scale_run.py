"""Actual frozen-parent input-scale probe, no updates and no benchmark scores."""
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from matched_mask_objective import frozen_parent_assignments, matched_mask_loss
from paired_support_loop import PairedSupportRun
from readback_preflight_checks import observed_readback_forward


class ScaleDiagnosticRun(PairedSupportRun):
    def run(self, criterion, manifest, started):
        assert self.args.mode == 'preflight' and self.spec['diagnostic_only']
        panel=json.loads((self.output/'panel.json').read_bytes())
        assert len(panel)==8 and sum(len(row['row_ids']) for row in panel)==58
        assert len({rid for row in panel for rid in row['row_ids']})==58
        assert all(rid in self.partitions['fit'] for row in panel for rid in row['row_ids'])
        self.heads['log_geometry']=copy.deepcopy(self.heads['box_conditioned'])
        assert all(torch.equal(value,self.heads['log_geometry'].state_dict()[name]) for name,value in self.heads['box_conditioned'].state_dict().items())
        original_heads={arm:{name:value.detach().clone() for name,value in head.state_dict().items()} for arm,head in self.heads.items()}
        self.dataset.augment=self.dataset.augment_det=True
        self.reset_rng()
        results=[]
        for item in panel:
            batch_cpu=next(iter(DataLoader(Subset(self.dataset,item['row_ids']),batch_size=len(item['row_ids']),shuffle=False,num_workers=0)))
            assert batch_cpu['local_training_id'].tolist()==item['row_ids']
            inputs,batch=self.prepare(batch_cpu,'train')
            with torch.no_grad():parent,call=observed_readback_forward(self.model,inputs)
            indices,targets=frozen_parent_assignments(parent,batch,criterion.matcher)
            raw=inputs['points'][:,1:].reshape(inputs['batch_size'],50000,6)
            # Build actual member statistics once, using the same native corrector.
            content_masks,geometry=self.heads['content'](parent['support_query_features'],parent['support_super_features'],raw,parent['native_coarse_center'],parent['native_coarse_size'],parent)
            del content_masks
            scales=parent['native_coarse_size'].detach()
            matched=[scales[bid,query_ids] for bid,(query_ids,_) in enumerate(indices)]
            scale_record=dict(all_candidates=int(scales.shape[0]*256),
                all_floor_dimensions=int((scales<=1e-6).sum()),
                all_floor_candidates=int((scales<=1e-6).any(-1).sum()),
                matched_floor_dimensions=sum(int((value<=1e-6).sum()) for value in matched),
                matched_floor_candidates=sum(int((value<=1e-6).any(-1).sum()) for value in matched),
                coarse_size_min=float(scales.min()),coarse_size_max=float(scales.max()))
            records={}
            for arm,head in self.heads.items():
                captures=[]
                def capture(module,args):
                    features=args[0];position=features[...,-9:]
                    raw_max=position.abs().amax(dim=(1,2))
                    transformed=position.sign()*torch.log1p(position.abs()) if arm=='log_geometry' else position
                    bid=len(captures);query_ids=indices[bid][0]
                    used_max=transformed.abs().amax(dim=(1,2))
                    captures.append(dict(raw_query_max=raw_max.detach().cpu().tolist(),used_query_max=used_max.detach().cpu().tolist(),
                        matched_raw_max=float(raw_max[query_ids].max()),matched_used_max=float(used_max[query_ids].max()),
                        raw_values_gt100=int((position.abs()>100).sum()),raw_values_gt10000=int((position.abs()>10000).sum())))
                    if arm=='log_geometry':return (torch.cat([features[...,:-9],transformed],-1),)
                handle=head.member[0].register_forward_pre_hook(capture)
                masks=[]
                for bid in range(len(raw)):
                    masks.append(head.correct_one(parent['support_query_features'][bid],parent['support_super_features'][bid],parent['native_coarse_center'][bid],parent['native_coarse_size'][bid],parent['last_pred_masks'][bid][0],parent['sp_last_pred_masks'][bid],parent['adaptive_weights'][bid],geometry[bid]))
                handle.remove()
                assert len(captures)==len(raw)
                assert all(torch.equal(a,b) for a,b in zip(masks,parent['sp_last_pred_masks']))
                predictions=dict(parent,sp_last_pred_masks=masks)
                loss,loss_record=matched_mask_loss(predictions,indices,targets)
                loss.backward()
                gradient_norms={name:float(parameter.grad.norm()) for name,parameter in head.named_parameters()}
                assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in head.parameters())
                norm=torch.stack([parameter.grad.norm(2) for parameter in head.parameters()]).norm(2)
                assert torch.isfinite(norm)
                assert all(parameter.grad is None for parameter in self.model.parameters())
                # This is a derivative probe. No clipping or optimizer step is performed.
                head.zero_grad(set_to_none=True)
                query_max=np.asarray([value for entry in captures for value in entry['used_query_max']])
                records[arm]=dict(loss=loss_record,gradient_norm=float(norm),parameter_gradient_norms=gradient_norms,
                    zero_output_mask_exact=True,per_query_geometry_max_percentiles=np.quantile(query_max,[0,.5,.9,.99,1]).tolist(),
                    per_expression_features=captures,optimizer_updates=0)
                del masks,predictions,loss
            assert len({record['loss']['loss'] for record in records.values()})==1
            assert all(torch.equal(value,self.heads[arm].state_dict()[name]) for arm,state in original_heads.items() for name,value in state.items())
            result=dict(original_training_step=item['original_step'],row_ids=item['row_ids'],
                point_sha256=[self.sha_bytes(value.numpy().tobytes()) for value in batch_cpu['point_clouds']],
                geometry_scale=scale_record,arms=records,actual_gt_assignments=[dict(queries=q.tolist(),target_ids=t.tolist()) for q,t in indices],
                original_log_gradients=item['original_gradient_norms'],parent_forward_calls=1,final_semantic_head_calls=call['final_semantic_head_calls'])
            results.append(result)
            self.write_json(self.output/'progress.json',dict(status='running',completed_batches=len(results),total_batches=8,optimizer_updates=0))
            print('INPUT_SCALE_BATCH '+json.dumps(dict(original_step=item['original_step'],geometry_scale=scale_record,gradients={arm:value['gradient_norm'] for arm,value in records.items()})),flush=True)
        self.frozen_state_exact()
        record=dict(status='complete',real_training_rows=58,parent_forward_calls=8,optimizer_updates=0,weight_files_created=0,
            candidate_inputs=58*256,accuracy_result=False,seed=2027,results=results,parent_and_head_states_exact=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            elapsed_seconds=time.perf_counter()-started,
            scope='Fresh enhanced training-panel input/derivative evidence only; not historical spike reproduction, trained geometry-head efficacy or formal accuracy.',
            geometric_compression='sign(position)*log1p(abs(position)); input probe only, no new deployed module selected')
        self.write_json(self.output/'diagnostic.json',record)
        self.write_json(self.output/'progress.json',dict(status='complete',completed_batches=8,total_batches=8,optimizer_updates=0))
        print('INPUT_SCALE_DIAGNOSTIC_COMPLETE',flush=True)

    @staticmethod
    def sha_bytes(raw):
        import hashlib
        return hashlib.sha256(raw).hexdigest()
