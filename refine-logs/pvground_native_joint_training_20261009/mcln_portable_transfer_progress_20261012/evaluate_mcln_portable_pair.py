"""Same-input native MCLN / Mask reference / current-best portable A+B replay."""
import argparse
import copy
import datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True)
parser.add_argument('--spec', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--max-batches', type=int, choices=(0,1), required=True)
options = parser.parse_args()
source = Path(options.source)
output = Path(options.output)
assert not output.exists()
output.mkdir(parents=True)
os.chdir(str(source))
sys.path.insert(0,str(source))

import numpy as np
import torch
from main_utils import parse_option
from train_dist_mod import TrainTester
from src.grounding_evaluator import GroundingEvaluator
from models.losses import _iou3d_par, box_cxcyczwhd_to_xyzxyz
from mcln_transfer_initialization import checked_payload, load_mcln_core, install_portable_states

spec = json.loads(Path(options.spec).read_bytes())
assert spec['seed'] == 2027 and spec['dataset'] == 'scanrefer'
random.seed(2027);np.random.seed(2027);torch.manual_seed(2027);torch.cuda.manual_seed_all(2027)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
core = checked_payload(spec['mcln_checkpoint'],spec['mcln_checkpoint_sha256'])
sys.argv = [str(source/'train_dist_mod.py')]
args = parse_option()
vars(args).update(vars(core['config']))
args.data_root = spec['data_root']
args.eval = True;args.debug = False;args.eval_train = False
args.batch_size = 8;args.num_workers = 4;args.rng_seed = 2027
args.use_cs_mcln = False;args.use_source_choice_selector = False
args.eval_use_selector_choice_scores = False;args.use_source_moe = False
args.pp_checkpoint = None
args.mask_loss_scale = 1.0 if args.mask_loss_scale is None else args.mask_loss_scale
args.consistency_loss_scale = 1.0 if args.consistency_loss_scale is None else args.consistency_loss_scale
assert args.model == 'MCLN' and args.dataset == ['scanrefer'] and args.test_dataset == 'scanrefer'
assert args.butd and not args.butd_cls and not args.butd_gt and args.use_color and not args.use_height and not args.use_multiview
assert args.num_target == 256 and args.num_decoder_layers == 6
model = TrainTester.get_model(args)
load_mcln_core(model,core)
del core
install_portable_states(model,spec)
model = model.cuda().eval()
assert len(model.state_dict()) == 1159
state = {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
criterion,set_criterion = TrainTester.get_criterion(args)
_,validation = TrainTester.get_datasets(args)
assert len(validation) == 9508
loader = torch.utils.data.DataLoader(validation,batch_size=8,shuffle=False,num_workers=4)
branches = ('native','mask_reference','full_transfer')
evaluators = {name:GroundingEvaluator(only_root=True,thresholds=[.25,.5],topks=[1],prefixes=['last_'],filter_non_gt_boxes=False,model='MCLN',eval_use_selector_choice_scores=False) for name in branches}
rows = []
started = time.monotonic()

def capture_rng():
    return (random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())

def restore_rng(value):
    random.setstate(value[0]);np.random.set_state(value[1]);torch.set_rng_state(value[2]);torch.cuda.set_rng_state_all(value[3])

def row_evidence(predictions,evaluator):
    probabilities = predictions['last_sem_cls_scores'].softmax(-1)
    main,modify,pron,other,_,relation,truths = evaluator._parse_gt(predictions)
    assert main.shape[-1] == probabilities.shape[-1] == 256
    all_boxes = torch.cat([predictions['last_center'],predictions['last_pred_size']],-1)
    candidate_boxes,candidate_scores,valid,axis = evaluator._resolve_position_candidates(predictions,'last_',all_boxes)
    assert axis=='default_query_axis' and candidate_scores is None and valid.all()
    assert torch.equal(candidate_boxes,all_boxes)
    records = []
    for bid in range(len(probabilities)):
        terms = [(probabilities[bid].unsqueeze(0)*pmap[bid,:1].unsqueeze(1)).sum(-1)
                 for pmap in (main,modify,pron,relation,other)]
        scores = terms[0]+terms[1]+terms[2]+terms[3]-terms[4]
        slot = int(evaluator._position_top_indices(scores,valid[bid][None],axis,1).item())
        assert predictions['box_label_mask'][bid,0]
        truth = truths[bid,:1]
        boxes = all_boxes[bid]
        ious,_ = _iou3d_par(box_cxcyczwhd_to_xyzxyz(truth),box_cxcyczwhd_to_xyzxyz(boxes))
        assert ious.shape == (1,256) and torch.isfinite(ious).all()
        own = predictions['sp_last_pred_masks'][bid][slot]
        text = predictions['last_pred_masks'][bid][0,slot]
        alpha = predictions['adaptive_weights'][bid]
        fused = alpha*text+(1-alpha)*own
        point_slots = predictions['superpoints'][bid].long()
        truth_mask = predictions['gt_masks'][bid,0].bool()
        miou = []
        for logits in (own,fused):
            mask = logits[point_slots] > 0
            union = (mask | truth_mask).sum();assert int(union) > 0
            miou.append(float((mask & truth_mask).sum().float()/union))
        records.append(dict(selected_query=slot,selected_box=boxes[slot].cpu().tolist(),selected_iou=float(ious[0,slot]),full256_max_iou=float(ious.max()),selected_own_mask_iou=miou[0],selected_fused_mask_iou=miou[1],root_gt_box=truth[0].cpu().tolist()))
    return records

iterator = loader if options.max_batches == 0 else itertools.islice(loader,1)
with (output/'paired_rows.jsonl').open('w',encoding='utf-8') as stream,torch.no_grad():
    for batch_index,raw in enumerate(iterator):
        common = capture_rng();branch_rows = {}
        for branch in branches:
            restore_rng(common);model.portable_geometry_arm = branch
            batch = TrainTester._to_gpu(copy.deepcopy(raw))
            inputs = TrainTester._get_inputs(batch);inputs['train'] = False
            predictions = model(inputs)
            for key,value in batch.items():
                assert key not in predictions,key
                predictions[key] = value
            _,predictions = TrainTester._compute_loss(predictions,criterion,set_criterion,args)
            predictions['last_pred_size'] = predictions['last_pred_size'].clamp_min(1e-6)
            if branch == 'native':
                native_next = capture_rng();native_semantic = predictions['last_sem_cls_scores'].clone()
            else:
                assert torch.equal(native_semantic,predictions['last_sem_cls_scores'])
            evaluators[branch].evaluate(predictions,'last_')
            branch_rows[branch] = row_evidence(predictions,evaluators[branch])
        restore_rng(native_next)
        for bid in range(len(raw['utterances'])):
            row = dict(index=len(rows),scan_id=raw['scan_ids'][bid],target_id=int(raw['target_id'][bid]),utterance=raw['utterances'][bid],point_sha256=hashlib.sha256(raw['point_clouds'][bid].contiguous().numpy().tobytes()).hexdigest())
            row.update({branch:branch_rows[branch][bid] for branch in branches})
            rows.append(row);stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        stream.flush()
        if (batch_index+1)%128 == 0:print(json.dumps(dict(completed_batches=batch_index+1,rows=len(rows),elapsed_seconds=time.monotonic()-started)),flush=True)
expected = 9508 if options.max_batches == 0 else 8
assert len(rows) == expected
assert all(torch.equal(value.detach().cpu(),state[name]) for name,value in model.state_dict().items())
metrics,transitions = {},{}
for branch in branches:
    evaluator = evaluators[branch]
    metrics[branch] = dict(rows=expected,primary_score='last/bbs')
    for threshold,suffix in ((.25,'025'),(.5,'050')):
        hits = sum(row[branch]['selected_iou'] > threshold for row in rows)
        assert int(evaluator.gts[('last_',threshold,1,'bbs')]) == expected
        assert int(evaluator.dets[('last_',threshold,1,'bbs')]) == hits
        metrics[branch].update({'hits'+suffix:hits,'acc'+suffix+'_percent':100*hits/expected,'full256_hits'+suffix:sum(row[branch]['full256_max_iou'] > threshold for row in rows)})
    metrics[branch]['selected_query_own_mask_miou'] = sum(row[branch]['selected_own_mask_iou'] for row in rows)/expected
    metrics[branch]['selected_query_fused_mask_miou'] = sum(row[branch]['selected_fused_mask_iou'] for row in rows)/expected
for branch in branches[1:]:
    transitions[branch] = {}
    for threshold,suffix in ((.25,'025'),(.5,'050')):
        repaired = sum(row['native']['selected_iou'] <= threshold < row[branch]['selected_iou'] for row in rows)
        broken = sum(row[branch]['selected_iou'] <= threshold < row['native']['selected_iou'] for row in rows)
        transitions[branch][suffix] = dict(repaired=repaired,broken=broken,net=repaired-broken)
        assert repaired-broken == metrics[branch]['hits'+suffix]-metrics['native']['hits'+suffix]
result = dict(status='MCLN_PORTABLE_PAIRED_REPLAY_COMPLETE',scope='FULL9508_NATIVE_MCLN_TRANSFER' if options.max_batches == 0 else 'REAL8ROW_NATIVE_MCLN_TRANSFER_SANITY',metrics=metrics,transitions=transitions,paired_input_read_once=True,branch_rng_restored=True,original_semantic_scores_equal=True,all_1159_states_unchanged=True,full_state_tensors=1159,optimizer_updates=0,weights_written=0,spec=spec,actual_PV_six_source_read_transferred=False,actual_zero_Face_branch_added=False,elapsed_seconds=time.monotonic()-started,completed_cst=datetime.datetime.now().astimezone().isoformat(),best_promoted=False,normal_training_completed=False,full_goal_complete=False)
(output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
