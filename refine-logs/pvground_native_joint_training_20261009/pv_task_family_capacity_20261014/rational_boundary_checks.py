"""Exact rational checks at precision-sensitive rows; no project imports."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import csv
import hashlib
import itertools
import json
from fractions import Fraction as Q
from pathlib import Path

BASE=Path("C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/deployment_geometry_training_20261012")
OUT=BASE/"selected_box_family_audit_20261014"
R="OBS_selected_geometry_rows_20261014_files/logs/scanrefer/selected_geometry/1791872741/"
summary=json.loads((OUT/"INDEPENDENT_RECOUNT.json").read_text())
def product(x):
    v=Q(1)
    for y in x:
        v*=y
    return v
def overlap_iou(b,g):
    width=[max(Q(0),min(b[a]+b[a+3]/2,g[a]+g[a+3]/2)-max(b[a]-b[a+3]/2,g[a]-g[a+3]/2)) for a in range(3)]
    inter=product(width)
    return inter/(product(b[3:])+product(g[3:])-inter)
def optimum(n,m,g):
    options=[]
    for a in range(3):
        events={Q(0),Q(1)}
        for edge_sign in (-1,1):
            origin=m[a]+edge_sign*m[a+3]/2
            end=n[a]+edge_sign*n[a+3]/2
            if end!=origin:
                for truth_sign in (-1,1):
                    boundary=g[a]+truth_sign*g[a+3]/2
                    at=(boundary-origin)/(end-origin)
                    if 0<at<1:
                        events.add(at)
        options.append(sorted(events))
    best=Q(-1);bestalpha=None
    for alpha in itertools.product(*options):
        box=[m[j]+alpha[j%3]*(n[j]-m[j]) for j in range(6)]
        value=overlap_iou(box,g)
        if value>best:
            best,bestalpha=value,alpha
    return best,bestalpha,len(options[0])*len(options[1])*len(options[2])
# Fractions are exact for the binary floats represented in JSON. This is not
# exact arithmetic for the original unsaved tensors or a full rational rerun.
output=[]
for ep in summary["epochs"]:
    e=ep["epoch"]
    rows=[json.loads(x) for x in (BASE/(R+"native_rows_epoch%03d.jsonl"%e)).read_text().splitlines()]
    csvrows=list(csv.DictReader((BASE/("selected_family_capacity_20261014/epoch%03d.csv"%e)).open()))
    indexes={ep["025"]["bound_nearest_threshold_row"],ep["050"]["bound_nearest_threshold_row"],
             ep["bound_excess_witness"]["row_index"]}
    for idx in sorted(indexes):
        row=rows[idx]
        n,m,g,f=([Q(x) for x in row[name]] for name in ("native_box","mask_endpoint_box","root_gt_box","selected_box"))
        best,alpha,count=optimum(n,m,g)
        actual=overlap_iou(f,g)
        recorded=float(csvrows[idx]["GT_only_family_max_IoU"])
        assert abs(float(best)-recorded)<1e-11
        assert (best>Q(1,4))==(recorded>.25) and (best>Q(1,2))==(recorded>.5)
        output.append(dict(epoch=e,row_index=idx,jsonl_line=idx+1,cartesian_candidates=count,
            exact_max_numerator=str(best.numerator),exact_max_denominator=str(best.denominator),
            exact_max_as_float=float(best),witness_rational=[str(x) for x in alpha],
            saved_final_exact_IoU_as_float=float(actual),
            saved_final_minus_exact_max_as_float=float(actual-best),
            exact_saved_final_exceeds_ideal_max=actual>best,
            rational_vs_CSV_abs_difference=abs(float(best)-recorded),
            hits025=best>Q(1,4),hits050=best>Q(1,2)))
(OUT/"RATIONAL_BOUNDARY_CHECKS.json").write_text(json.dumps(dict(status="PASS",rows=output,
    scope="Nine selected rows: six nearest thresholds and three maximum saved-bound gaps; exact binary-rational saved coordinates; numerical checks only"),indent=2)+"\n")
print(json.dumps(dict(status="PASS",cases=len(output),max_rational_vs_csv=max(x["rational_vs_CSV_abs_difference"] for x in output),
    excess=[dict(epoch=x["epoch"],row=x["row_index"],gap=x["saved_final_minus_exact_max_as_float"]) for x in output if x["exact_saved_final_exceeds_ideal_max"]])))

