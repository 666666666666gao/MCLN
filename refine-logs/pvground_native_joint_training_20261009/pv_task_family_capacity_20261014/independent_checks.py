"""Local-only independent saved-box audit; no project import or neural execution."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import ast
import csv
import datetime
import difflib
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path
import numpy as np

BASE = Path("C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012")
OUT = BASE / "selected_box_family_audit_20261014"
SNAP = "span_lr_actual_pair_review_20261014/snapshots/"
ROWS = "OBS_selected_geometry_rows_20261014_files/logs/scanrefer/selected_geometry/1791872741/"
inputs, trace = {}, []

def read(rel):
    p = BASE / rel
    raw = p.read_bytes()
    stamp = dict(path=str(p), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    if str(p) in inputs:
        assert stamp == inputs[str(p)]
    inputs[str(p)] = stamp
    return raw

def jsread(rel):
    return json.loads(read(rel))

def put(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def check(name, ok, detail):
    item = dict(check=name, status="PASS" if bool(ok) else "FAIL", detail=detail)
    trace.append(item)
    if not bool(ok):
        put("DETERMINISTIC_TRACE_PARTIAL.json", trace)
    assert bool(ok), name

def scalar_iou(box, truth):
    intersection = 1.0
    vp, vg = 1.0, 1.0
    for j in range(3):
        intersection *= max(0.0, min(box[j]+box[j+3]/2, truth[j]+truth[j+3]/2) -
                            max(box[j]-box[j+3]/2, truth[j]-truth[j+3]/2))
        vp *= box[j+3]
        vg *= truth[j+3]
    return intersection/(vp+vg-intersection)

def independent_family(native, mask, truth):
    # Work with 1-D interval overlap and interval length; form their outer products.
    n = len(truth)
    options = np.zeros((n, 3, 6))
    options[:, :, 1] = 1.0
    widths, overlaps = [], []
    counts, constant_edges = np.zeros((n, 3), dtype=int), 0
    for a in range(3):
        for r in range(n):
            ml, mh = mask[r,a]-mask[r,a+3]/2, mask[r,a]+mask[r,a+3]/2
            nl, nh = native[r,a]-native[r,a+3]/2, native[r,a]+native[r,a+3]/2
            gl, gh = truth[r,a]-truth[r,a+3]/2, truth[r,a]+truth[r,a+3]/2
            events = {0.0, 1.0}
            for origin, destination in ((ml,nl),(mh,nh)):
                velocity = destination-origin
                if velocity == 0:
                    constant_edges += 1
                    continue
                for boundary in (gl,gh):
                    t = (boundary-origin)/velocity
                    if 0 < t < 1:
                        events.add(t)
            sorted_events = sorted(events)
            options[r,a,:len(sorted_events)] = sorted_events
            counts[r,a] = len(sorted_events)
        weight = options[:,a]
        center = mask[:,a,None] + weight*(native[:,a,None]-mask[:,a,None])
        width = mask[:,a+3,None] + weight*(native[:,a+3,None]-mask[:,a+3,None])
        low = np.maximum(center-width/2, (truth[:,a]-truth[:,a+3]/2)[:,None])
        high = np.minimum(center+width/2, (truth[:,a]+truth[:,a+3]/2)[:,None])
        widths.append(width)
        overlaps.append(np.maximum(high-low,0))
    volume = widths[0][:,:,None,None]*widths[1][:,None,:,None]*widths[2][:,None,None,:]
    inter = overlaps[0][:,:,None,None]*overlaps[1][:,None,:,None]*overlaps[2][:,None,None,:]
    union = volume + truth[:,3:].prod(-1)[:,None,None,None] - inter
    assert (volume>0).all() and (union>0).all()
    values = inter/union
    flattened = values.reshape(n,-1)
    selected = flattened.argmax(-1)
    ids = np.array(np.unravel_index(selected,(6,6,6))).T
    alpha = options[np.arange(n)[:,None],np.arange(3)[None,:],ids]
    return flattened.max(-1), alpha, dict(constant_edges=constant_edges,
        unique_event_count_min=int(counts.min()), unique_event_count_max=int(counts.max()),
        minimum_candidate_volume=float(volume.min()), minimum_candidate_union=float(union.min()))

start = time.perf_counter()
required = [
"task_path_and_family_capacity_20261014/SOURCE_AND_EXECUTION_BINDINGS.json",
"task_path_and_family_capacity_20261014/REPORT.md",
"selected_family_capacity_20261014.py",
"selected_family_capacity_20261014/RESULT.json",
"selected_family_capacity_20261014_attempt1_size_assertion/FAILED_SOURCE.py",
"selected_family_capacity_20261014_attempt1_size_assertion/FAILURE.json",
"recount_selected_rows_20261014.py",
"selected_geometry_cpu_recount_20261014/RESULT.json",
"OBS_selected_geometry_rows_20261014_INTAKE.json",
"span_lr_runtime_staging_20261013_v2/SOURCE_STAGE_RECEIPT.json",
"selected_geometry_terminal_audit_20261014/REVIEW_COMPLETE.json",
"selected_geometry_terminal_audit_20261014/EXPERIMENT_AUDIT.md",
"SOURCE_STAGE_R4_RECEIPT.json",
"source_overlay_v4/selected_geometry_eval_diagnostics.py",
"OBS_selected_geometry_rows_20261014_files/protocol.json",
]
snap_names = ["058_pv_ground.py","055_losses.py","125_mask_support_corrector.py",
"124_extremal_span_mixer.py","126_native_mask_geometry.py","127_native_model_initialization.py",
"128_native_root_bbs.py","130_pvground_semantic_assignment.py","135_selected_support_geometry_objective.py",
"103_joint_det_dataset.py","105_visual_data_handlers.py","102_grounding_evaluator.py",
"107_train_dist_mod.py","050_main_utils.py","136_selected_geometry_eval_diagnostics.py",
"013_initialization_reference.json"]
required += [SNAP+x for x in snap_names]
required += [ROWS+"native_metrics.jsonl"]
for e in range(3):
    required += [ROWS+"native_rows_epoch%03d.jsonl"%e, ROWS+"native_diagnostic_epoch%03d.json"%e,
                 "selected_family_capacity_20261014/epoch%03d.csv"%e]
for rel in required:
    read(rel)
intake = jsread("OBS_selected_geometry_rows_20261014_INTAKE.json")
for entry in intake["files"]:
    raw=read("OBS_selected_geometry_rows_20261014_files/"+entry["name"])
    check("intake/"+entry["name"],len(raw)==entry["bytes"] and hashlib.sha256(raw).hexdigest()==entry["sha256"],
          dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
bindings=jsread(required[0])
stage=jsread("span_lr_runtime_staging_20261013_v2/SOURCE_STAGE_RECEIPT.json")
oldstage=jsread("SOURCE_STAGE_R4_RECEIPT.json")
source_bindings=[]
for relative,entry in bindings["source_stage_bound_paths"].items():
    filename=entry["path"].replace("\\","/").split("/")[-1]
    raw=read(SNAP+filename)
    actual=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    target=stage["source_files"][relative]
    old=oldstage["source_files"][relative]
    check("current_source/"+relative,all(actual[k]==entry[k]==target[k]==old[k] for k in ("bytes","sha256")),actual)
    source_bindings.append(dict(relative=relative,local_path=str(BASE/(SNAP+filename)),**actual,
                                current_stage_matches=True, closed_C_stage_matches=True))
for filename,relative in [
("103_joint_det_dataset.py","src/joint_det_dataset.py"),("105_visual_data_handlers.py","src/visual_data_handlers.py"),
("102_grounding_evaluator.py","src/grounding_evaluator.py"),("107_train_dist_mod.py","train_dist_mod.py"),
("050_main_utils.py","main_utils.py"),("136_selected_geometry_eval_diagnostics.py","selected_geometry_eval_diagnostics.py")]:
    raw=read(SNAP+filename)
    actual=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    check("additional_current_source/"+relative,all(actual[k]==stage["source_files"][relative][k] for k in actual),actual)
    source_bindings.append(dict(relative=relative,local_path=str(BASE/(SNAP+filename)),**actual,
                               current_stage_matches=True,closed_C_stage_matches=actual["sha256"]==oldstage["source_files"][relative]["sha256"]))
oldraw=read("source_overlay_v4/selected_geometry_eval_diagnostics.py")
check("closed_C_diagnostic_source",hashlib.sha256(oldraw).hexdigest()==oldstage["source_files"]["selected_geometry_eval_diagnostics.py"]["sha256"],
      "Current diagnostic adds gates; saved C rows bind the older source, checked separately.")
put("SOURCE_BINDINGS.json",source_bindings)

capacity=jsread("selected_family_capacity_20261014/RESULT.json")
recount=jsread("selected_geometry_cpu_recount_20261014/RESULT.json")
check("capacity_result_binding",hashlib.sha256(read("selected_family_capacity_20261014/RESULT.json")).hexdigest()==bindings["capacity_result_sha256"],bindings["capacity_result_sha256"])
metrics=[json.loads(x) for x in read(ROWS+"native_metrics.jsonl").decode().splitlines()]
previous=jsread("selected_geometry_terminal_audit_20261014/REVIEW_COMPLETE.json")
prior_report_path=str(BASE/"selected_geometry_terminal_audit_20261014/EXPERIMENT_AUDIT.md")
prior_report=[x for x in previous["outputs"] if Path(x["path"])==Path(prior_report_path)]
check("previous_used_report_seal",len(prior_report)==1 and all(inputs[prior_report_path][k]==prior_report[0][k] for k in ("bytes","sha256")),
      "Prior report is provenance context only, not substitute for recount.")

all_epochs,first_rows,first_final=[],None,None
for e in range(3):
    raw=read(ROWS+"native_rows_epoch%03d.jsonl"%e)
    rows=[json.loads(x) for x in raw.decode().splitlines()]
    diagnostic=jsread(ROWS+"native_diagnostic_epoch%03d.json"%e)
    check("epoch%d_row_binding"%e,len(raw)==diagnostic["row_file_bytes"] and hashlib.sha256(raw).hexdigest()==diagnostic["row_file_sha256"] and len(rows)==9508,"9508 primary saved rows")
    check("epoch%d_row_identity"%e,[r["row_index"] for r in rows]==list(range(9508)) and all(0<=r["selected_query"]<256 for r in rows),"Contiguous rows; query slots in [0,255]")
    native,mask,truth,final=[np.array([r[k] for r in rows],dtype=np.float64) for k in ("native_box","mask_endpoint_box","root_gt_box","selected_box")]
    check("epoch%d_box_dimensions"%e,all(np.isfinite(a).all() and (a[:,3:]>0).all() for a in (native,mask,truth,final)),"All finite, all positive dimensions")
    ious={name:np.array([scalar_iou(x,y) for x,y in zip(box,truth)]) for name,box in [("final",final),("native",native),("mask_endpoint",mask)]}
    bound,witness,event_stats=independent_family(native,mask,truth)
    support=np.array([2*r["own_mask_intersection"]>r["own_mask_union"] and 2*r["fused_mask_intersection"]>r["fused_mask_union"] for r in rows])
    check("epoch%d_support_counts"%e,all(0<=r[p+"_intersection"]<=r[p+"_union"] for r in rows for p in ("own_mask","fused_mask")),int(support.sum()))
    claimed=list(csv.DictReader(read("selected_family_capacity_20261014/epoch%03d.csv"%e).decode().splitlines()))
    check("epoch%d_csv_identity"%e,len(claimed)==9508 and all(int(c["row_index"])==i and int(c["selected_query"])==rows[i]["selected_query"] for i,c in enumerate(claimed)),"Every CSV row matched")
    csvdiff={}
    for name,key in [("final","saved_final_IoU"),("native","native_IoU"),("mask_endpoint","mask_endpoint_IoU")]:
        csvdiff[key]=float(np.abs(ious[name]-np.array([float(c[key]) for c in claimed])).max())
    csv_bound=np.array([float(c["GT_only_family_max_IoU"]) for c in claimed])
    csv_alpha=np.array([[float(c["GT_only_alpha_"+axis]) for axis in "xyz"] for c in claimed])
    check("epoch%d_csv_alpha_range"%e,((csv_alpha>=0)&(csv_alpha<=1)).all(),"Claimed witnesses lie in cube")
    wa=np.tile(csv_alpha,(1,2))
    csv_witness=np.array([scalar_iou(x,y) for x,y in zip(mask+wa*(native-mask),truth)])
    csvdiff["GT_only_family_max_IoU"]=float(abs(bound-csv_bound).max())
    csvdiff["alpha_witness_value"]=float(abs(csv_witness-csv_bound).max())
    check("epoch%d_all_csv_numeric"%e,max(csvdiff.values())<1e-11,csvdiff)
    check("epoch%d_all_csv_support"%e,all((c["own_and_fused_support_qualified"]=="True")==bool(support[i]) for i,c in enumerate(claimed)),"All saved support flags reproduced")
    if first_rows is None:
        first_rows,first_final=rows,ious["final"]
    check("epoch%d_fixed_row_GT_fingerprints"%e,all(all(x[k]==y[k] for k in ("row_index","scan_id","target_id","utterance_sha256","root_gt_box","point_cloud_sha256","gt_mask_sha256")) for x,y in zip(first_rows,rows)),
          "Saved fingerprints agree; original point clouds and GT masks not loaded")
    slopes=np.stack([native[:,:3]-mask[:,:3],native[:,3:]-mask[:,3:]],axis=-1)
    move=np.stack([final[:,:3]-mask[:,:3],final[:,3:]-mask[:,3:]],axis=-1)
    norm=(slopes**2).sum(-1)
    inferred=np.zeros_like(norm)
    np.divide((slopes*move).sum(-1),norm,out=inferred,where=norm!=0)
    residual=move-np.clip(inferred,0,1)[:,:,None]*slopes
    ep=dict(epoch=e,rows=len(rows),unique_scenes=len(set(r["scan_id"] for r in rows)),
        unique_scan_targets=len(set((r["scan_id"],r["target_id"]) for r in rows)),
        unique_scan_target_utterance_keys=len(set((r["scan_id"],r["target_id"],r["utterance_sha256"]) for r in rows)),
        support_qualified=int(support.sum()),minimum_GT_volume=float(truth[:,3:].prod(-1).min()),
        minimum_native_size=float(native[:,3:].min()),minimum_mask_size=float(mask[:,3:].min()),
        event_stats=event_stats,csv_max_abs_differences=csvdiff,
        selected_query_slot_changes_vs_E0=sum(x["selected_query"]!=y["selected_query"] for x,y in zip(first_rows,rows)),
        reconstruction_residual_max=float(abs(residual).max()),
        maximum_saved_final_minus_bound=float((ious["final"]-bound).max()),
        positive_saved_final_minus_bound_rows=int((ious["final"]>bound).sum()),
        saved_final_minus_bound_above_1e_12_rows=int((ious["final"]>bound+1e-12).sum()),
        max_cpu64_vs_logged={name:float(abs(values-np.array([r[name+"_iou"] for r in rows])).max()) for name,values in ious.items()})
    ep["bound_excess_witness"]=dict(row_index=int(np.argmax(ious["final"]-bound)),
        primary_jsonl_line=int(np.argmax(ious["final"]-bound))+1,
        selected_IoU=float(ious["final"][np.argmax(ious["final"]-bound)]),
        family_max=float(bound[np.argmax(ious["final"]-bound)]))
    for suffix,t in [("025",.25),("050",.5)]:
        hit=ious["final"]>t
        reachable=bound>t
        wrong=~hit
        q=support&wrong
        endpoint=(ious["native"]>t)|(ious["mask_endpoint"]>t)
        values=dict(actual_selected_hits=int(hit.sum()),
          GT_only_independent_axis_family_upper_bound_hits=int(reachable.sum()),
          saved_errors_with_passing_same_selected_endpoint=int((wrong&endpoint).sum()),
          saved_errors_reachable_in_family=int((wrong&reachable).sum()),
          saved_errors_not_reachable_in_family=int((wrong&~reachable).sum()),
          selected_own_and_fused_support_good_box_bad=int(q.sum()),
          support_good_box_bad_reachable=int((q&reachable).sum()),
          support_good_box_bad_not_reachable=int((q&~reachable).sum()),
          family_max_within_1e_5_of_threshold=int((abs(bound-t)<1e-5).sum()),
          float64_bound_threshold_losses_vs_saved_final=int((hit&~reachable).sum()))
        check("epoch%d_summary_%s"%(e,suffix),values==capacity["epochs"][e][suffix],values)
        check("epoch%d_primary_hits_%s"%(e,suffix),int(hit.sum())==diagnostic[suffix]["hits"]==metrics[e]["hits"+suffix]==recount["epochs"][e][suffix]["hits"],int(hit.sum()))
        ep[suffix]=dict(values,minimum_bound_threshold_distance=float(abs(bound-t).min()),
          bound_nearest_threshold_row=int(abs(bound-t).argmin()),
          accuracy_percent=float(100*hit.mean()),bound_accuracy_percent=float(100*reachable.mean()),
          logged_threshold_disagreements={name:int(((v>t)!=np.array([r[name+"_iou"]>t for r in rows])).sum()) for name,v in ious.items()},
          E0_repairs=int((~(first_final>t)&hit).sum()),E0_breaks=int(((first_final>t)&~hit).sum()),
          logged_full256_coverage=sum(r["best256_iou"]>t for r in rows))
    with (OUT/("independent_epoch%03d.csv"%e)).open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["row_index","saved_final_IoU","independent_family_max","witness_x","witness_y","witness_z","csv_difference"])
        for i in range(len(rows)):
            writer.writerow([i,ious["final"][i],bound[i],*witness[i],bound[i]-csv_bound[i]])
    all_epochs.append(ep)

# Pure function extraction; executing main or any model imports is forbidden.
tree=ast.parse(read("selected_family_capacity_20261014.py").decode())
pure=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ("iou","family_max","independent_checks")])
env=dict(np=np,itertools=itertools)
exec(compile(pure,"audited_pure_numeric_functions","exec"),env)
claimed_numeric=env["independent_checks"]()
check("original_numeric_fixture_summary",claimed_numeric==capacity["meaningful_numeric_checks"],claimed_numeric)
# Additional analytic constant/touch/disjoint/interior/shrinking/expanding cases.
g=np.array([0.,0.,0.,1.,1.,1.])
cases=[
("equal_edges",g,g,g,1.),
("constant_disjoint",np.array([3.,0,0,1,1,1]),np.array([3.,0,0,1,1,1]),g,0.),
("constant_tangent",np.array([1.,0,0,1,1,1]),np.array([1.,0,0,1,1,1]),g,0.),
("interior_translation",np.array([.5,0,0,1,1,1]),np.array([-.5,0,0,1,1,1]),g,1.),
("independent_axis_choice",np.array([0.,0,0,2,1,1]),np.array([0.,0,0,1,2,1]),g,1.),
("stationary_lower_edge",np.array([.5,0,0,2,1,1]),g,g,1.),
("stationary_upper_edge",np.array([-.5,0,0,2,1,1]),g,g,1.),
("tiny_positive_constant",np.array([0.,0,0,1e-8,1e-8,1e-8]),np.array([0.,0,0,1e-8,1e-8,1e-8]),np.array([0.,0,0,1e-8,1e-8,1e-8]),1.),
("threshold_exact_half",np.array([0.,0,0,.5,1,1]),np.array([0.,0,0,.5,1,1]),g,.5),
("threshold_exact_quarter",np.array([0.,0,0,.25,1,1]),np.array([0.,0,0,.25,1,1]),g,.25)]
analytic=[]
for name,n,m,t,expected in cases:
    independent,_,_=independent_family(n[None],m[None],t[None])
    author,_=env["family_max"](n[None],m[None],t[None])
    check("analytic/"+name,abs(independent[0]-expected)<1e-12 and abs(author[0]-expected)<1e-12,dict(expected=expected,independent=float(independent[0]),author=float(author[0])))
    analytic.append(dict(name=name,expected=expected,value=float(independent[0])))
# Deterministic independent dense search is a lower-bound sanity check, not a proof.
rng=np.random.RandomState(419)
target=np.concatenate((rng.normal(size=(40,3)),np.exp(rng.uniform(-2,2,(40,3)))),axis=-1)
n=target.copy();m=target.copy()
n[:,:3]+=rng.normal(size=(40,3));m[:,:3]+=rng.normal(size=(40,3))
n[:,3:]*=np.exp(rng.uniform(-1,1,(40,3)));m[:,3:]*=np.exp(rng.uniform(-1,1,(40,3)))
best,_,_=independent_family(n,m,target)
author,_=env["family_max"](n,m,target)
dense=np.zeros(40)
for alpha in itertools.product(np.linspace(0,1,17),repeat=3):
    box=m+np.array(alpha+alpha)[None]*(n-m)
    value=np.array([scalar_iou(x,y) for x,y in zip(box,target)])
    dense=np.maximum(dense,value)
check("independent_random_dense_grid",np.all(best+1e-12>=dense) and abs(best-author).max()<1e-11,
      dict(cases=40,triples_per_case=17**3,minimum_bound_minus_dense=float((best-dense).min()),maximum_author_difference=float(abs(best-author).max())))
failed=read("selected_family_capacity_20261014_attempt1_size_assertion/FAILED_SOURCE.py").decode().splitlines()
successful=read("selected_family_capacity_20261014.py").decode().splitlines()
diff=list(difflib.unified_diff(failed,successful,fromfile="FAILED_SOURCE.py",tofile="selected_family_capacity_20261014.py",lineterm=""))
check("failed_attempt_only_changed_assertion",
      [x for x in diff if x.startswith("-") and not x.startswith("---")]==["-        assert min(native[:, 3:].flat) >= 1e-6 and min(mask[:, 3:].flat) >= 1e-6"] and
      [x for x in diff if x.startswith("+") and not x.startswith("+++")]==["+        assert (native[:, 3:] > 0).all() and (mask[:, 3:] > 0).all()"],
      dict(diff=diff,float32_floor_as_float64=float(np.float32(1e-6)),literal_double=1e-6))
# Do not execute failed historical main (it writes its own output directory).
put("INDEPENDENT_RECOUNT.json",dict(epochs=all_epochs,analytic_numeric_fixtures=analytic,
    original_numeric_fixture_summary=claimed_numeric,
    independent_dense_fixture=dict(seed=419,cases=40,triples_per_case=17**3),
    evaluation_classification="saved_real_gt_coordinate_recount_and_GT_conditioned_oracle_bound",
    CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"],neural_forwards=0,optimizer_updates=0,
    model_factory_calls=0,project_imports=0,checkpoint_loads=0,remote_queries=0,
    original_dataset_GT_reloaded=False,point_masks_recomputed=False,elapsed_seconds=time.perf_counter()-start))
put("DETERMINISTIC_TRACE.json",dict(runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__,cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"]),checks=trace))
put("INPUTS_BEFORE.json",list(inputs.values()))
print(json.dumps(dict(status="COMPLETE",inputs=len(inputs),checks=len(trace),elapsed_seconds=time.perf_counter()-start,
    epochs=[dict(epoch=e["epoch"],actual=[e["025"]["actual_selected_hits"],e["050"]["actual_selected_hits"]],
      family=[e["025"]["GT_only_independent_axis_family_upper_bound_hits"],e["050"]["GT_only_independent_axis_family_upper_bound_hits"]],
      max_csv_difference=e["csv_max_abs_differences"]["GT_only_family_max_IoU"],
      max_saved_minus_bound=e["maximum_saved_final_minus_bound"]) for e in all_epochs])))

