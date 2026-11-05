## Current Override: 2026-11-05 01:19 UTC

Actual remote observation time is2026-11-05 01:19UTC,also consistent
with local2026-11-04 20:19-05:00. The triggering heartbeat timestamp
2026-11-04 23:13UTC is earlier; do not backdate these observations.

Full current-run histories now contain NR1-78 and SR1-29. NR78 full7899
last semantic alignment Top1=4122/3380 hits025/050; best66=4255/3435
unchanged under025-first selection. SR29 full17726=11524/9269;
best15=11743/9434 unchanged. Both latest own CPU checkpoints passed
epoch78/29,1134 tensors,nonempty optimizer/scheduler and SHA audits,
CUDA uninitialized. NR78 SHA
dba99280cdf7e2ac9803b7550582d7ed77afdc92a0208524aac51dd70dfda564;
SR29 SHA abaf53061f236661423ba2fe8a01199b5243e22af4b49ee356b566637390e71d.
Existing retention watchers keep NR66/78 and SR15/29. The earlier SR28
save-before-validation window is resolved:full28=11434/9296. Its metric
and historical CPU audit remain;its weight was removed by authorized
retention. No manual cleanup,
trainer restart,watcher restart,configuration or initialization change.

At01:19UTC all six original NR/SR trainer/keeper/retention processes
were live. Null checkpoint/pp initialization,innovations-off settings,
original240/140 budgets and optimizer recipes unchanged. Recent logged
losses finite,no recent error lines. NR progress370/3742 and SR2471/6486
are training counters,not validation scores. At observed rates next full
validation is approximately02:20-02:40UTC NR,03:10-03:25UTC SR; update
from actual speed. Best-target deficits remain NR469 and SR559 Acc025
hits;Scan B2=5483/4656 remains222/289 short. No target achieved.
Scan01:21:26UTC:GPU idle0MiB,no compute process,only existing retention
screen;B2 SHA unchanged. Prepared experiment remains unlaunched;do not
start a new run. Resource receipt:scanrefer-current-resources-20261105T0120.txt.
Receipts:*current-full-history-20261105T0119.json and
*current-scratch-progress-20261105T0119.json,under
docs/results/mcln_pvground_20261101/. Local file/Git/results only;
remote calls exclusively work/remote_pvground.ps1.

## Previous Override: 2026-11-04 21:12 UTC

NR3D completed full7899 epochs73=4198/3311 and74=4228/3438 hits025/050.
Acc025-first best remains66=4255/3435; do not combine best66's025 with
epoch74's050. Own CPU epoch74/1134 tensors/nonempty optimizer/scheduler
and SHA passed,CUDA uninitialized:
b4fd3dd35177c7ee61dc1fe00ee95786b1dc5d86f160c9165ae9f3dda7d4319a.
Existing watcher retained66/74 at21:04:46UTC and removed preserved73;
all full metric rows and audits remain. No manual cleanup or restart.

SR28 was in validation at21:10UTC,407/1478 batches; full17726 score is
not available yet. The collector correctly rejected comparing the saved
epoch28 last checkpoint to the last completed log row27. Current-only
retention inspection confirms best15/latest-complete27 preserved, native
checkpoint28 saved,trainer/keeper/watcher live. Keep latest confirmed
full27=11671/9338 and best15=11743/9434; do not assign either to28.
This is an observed checkpoint-before-validation synchronization window,
not evidence of a trainer failure. Retried after240seconds: same guarded
epoch mismatch while validation remained incomplete. At21:16:19UTC,
validation advanced to833/1478 with live processes,finite recent losses
and no error lines. Estimated completion21:25-21:27UTC; leave pending28
unscored until its full validation/keeper/retention synchronization.
Raw scoped retention state:sr3d-current-retention-state-20261104T2116.json.

At21:10UTC all six NR/SR trainer/keeper/retention processes were live,
original scratch configs/budgets and null initialization unchanged,
innovations off,recent logged losses finite,no recent error lines.
NR75 training569/3742 suggests next full result around22:05UTC.
Scan21:12:42UTC:GPU idle0MiB,no compute process,only existing retention
screen;B2 exists with unchanged SHA46ecfb6a6831fca0704c55fe87fa96c9c92e749824f13173d67d5697da1b92f8.
No new Scan run launched. Best-target gaps remain Scan222/289,NR469,
SR559 hits; all targets unmet. Full NR history receipt:
nr3d-current-full-history-20261104T2111.json; current progress receipts:
nr3d/sr3d-current-scratch-progress-20261104T2110.json. Local work remains
file/Git/result handling only;remote calls exclusively approved wrapper.

## Previous Override: 2026-11-04 19:11 UTC

Current-only remote collectors audited all NR epochs1-72 and SR epochs1-27.
NR latest72 full7899=4158/3362; best66=4255/3435 remains unchanged.
SR latest27 full17726=11671/9338; best15=11743/9434 remains unchanged.
These are same-checkpoint last semantic alignment Top1/BBF hit pairs025/050.
Latest own CPU epochs72/27,1134 tensors,nonempty optimizer/scheduler and
SHA audits passed with CUDA uninitialized. NR72 SHA
18ad4176f19a2d924f5e55927226379251ae85f9159b2c919f4b6a36752e1489;
SR27 SHA c06138b624e12b2dd0bb18df0887616885694fa3a0d79bea7ed2fbb5c7779a90.
Existing watchers retain NR66/72 and SR15/27. Removed historical weights
are marked in the complete epoch table; all metric/audit history remains.
No manual cleanup or watcher/trainer restart.

At19:11UTC NR trainer96960/keeper135075/retention156800 and SR
trainer92354/keeper129032/retention148725 were all live. Original scratch
configs/budgets, null initialization and innovations-off settings were
unchanged. Recent logged losses were finite with no recent error lines.
Progress was NR705/3742 and SR1675/6486; these counters are not scores.
Next validation is approximately20:00UTC NR and21:15UTC SR at current
rates; update ETA from real speed. Best-target deficits remain NR469
and SR559 Acc025 hits. Scan best B2=5483/4656,gaps222/289. All unmet.

Scan BBF quality experiment preparation completed remotely17:09:45UTC
in scanrefer_bbf_quality_20261104. CPU-only preflight verified exact B2
1168-tensor load,zero adapter identity,fresh optimizer and synthetic loss
eligibility/finite gradients. This is preparation,not GPU training or full
validation. Latest heartbeat forbids automatic new Scan launches; keep
the prepared experiment unlaunched. Do not restart completed Scan trials.
Raw full-history receipts:*current-full-history-20261104T1911.json;
preparation:scanrefer-bbf-quality-preparation-20261104T1709.json,under
docs/results/mcln_pvground_20261101/. Local work was files/Git/results
only;all remote calls used work/remote_pvground.ps1.

# Current MCLN + PV-Ground Handoff

## Progress Override: 2026-11-04 14:58 UTC

NR3D run1793553113 completed epoch68 full7899 validation:4141/3420 hits
(Acc@0.25/0.50=52.424358%/43.296620%). It remains below best epoch66
4255/3435 by114/15 hits;best Acc@0.25 is469 hits short of target4724.
CPU own-epoch68/1134 tensors/nonempty optimizer/scheduler/SHA audit
passed;SHA d508d840391823d997d74aa4a3ffe9222e3cee52f43eb8102d32d37f7487d70c.
Best66 SHA remains ca645bf8f191a0609b603b77e829b61e492b6459805125ae012c164b783fcb5b.
Existing watcher applied14:47:25UTC and retained66/68,removed only
preserved67. Full receipt is
`docs/results/mcln_pvground_20261101/nr3d-current-full-history-20261104T1455.json`;
epoch68 audit is
`docs/results/mcln_pvground_20261101/nr3d-epoch68-current-audit-20261104T1455.json`.

At14:58UTC NR trainer/keeper/retention were live in epoch69=774/3742;
SR run1793552465 trainer92354/keeper129032/retention148725 were live in
epoch26=3106/6486. Both retain original scratch configuration/budget,
null initialization and innovations disabled; recent losses are finite
and no recent error lines were found. Latest complete SR25=11686/9455,
best15=11743/9434; Scan best B2=5483/4656. All targets remain unmet.

## Progress Override: 2026-11-04 14:19 UTC

Current-only approved-wrapper checks confirmed NR3D run1793553113
trainer96960/keeper135075/retention156800 live at epoch68=2288/3742 and
SR3D run1793552465 trainer92354/keeper129032/retention148725 live at
epoch26=1475/6486. Both use their unchanged original scratch configs and
budgets, null initialization and innovations disabled. Recent parsed
losses are finite and no recent error lines were found. No complete
validation or retention event was observed; do not infer accuracy from
the progress counters. Latest complete remain NR67=4089/3312 (best66=
4255/3435), SR25=11686/9455 (best15=11743/9434); Scan best B2=5483/4656.
All targets remain unmet. Approximate remaining train time is20min NR68
and2h SR26 at the observed rates, then full validation. Raw progress
receipt:`docs/results/mcln_pvground_20261101/mcln-monitor-progress-20261104T1419Z.json`.

## Progress Override: 2026-11-04 14:03 UTC

NR3D run1793553113 completed epoch67 full7899 validation:4089/3312
hits (Acc@0.25/0.50=51.766046%/41.929358%),below epoch66 best4255/3435
by166/123 hits. The best remains469 Acc@0.25 hits below target4724.
Remote CPU audit verified epoch67 checkpoint identity,1134 model tensors,
nonempty optimizer/scheduler,SHA
f780fefd797f1c6afab780091c0605feb04bbe92effd334aee519e5964bdffd6;
best66 SHA remains
ca645bf8f191a0609b603b77e829b61e492b6459805125ae012c164b783fcb5b.
Existing retention selected best66/latest67. Detailed audit:
`docs/results/mcln_pvground_20261101/nr3d-epoch67-current-audit-20261104T1400.json`.

SR3D run1793552465 completed epoch25 full17726 validation:11686/9455
hits (Acc@0.25/0.50=65.925759%/53.339727%). Best remains epoch15=
11743/9434;epoch25 is -57/+21 hits versus best and616 Acc@0.25 hits
below target12302. CPU own-epoch25/1134 tensors/nonempty optimizer/
scheduler/SHA audit passed;SHA
dfc15899a0abd0a72d4034f48d76046af9630385ceda2c7be17c0b4ada1b54bb.
The watcher retained15/25 and removed only old preserved24. Detailed
audit:`docs/results/mcln_pvground_20261101/sr3d-epoch25-current-audit-20261104T1353.json`.

At14:03UTC both original scratch trainers,keepers and retention watchers
were live. NR epoch68 was1191/3742;SR epoch26 was790/6486. Original
recipes/budgets, null initialization and innovations-off settings remain
unchanged;recent losses were finite and no recent error lines were found.
Scan best B2=5483/4656 remains222/289 hits short of5705/4945. All targets
remain unmet. Full-history receipts and `all-epochs.csv` contain verified
current results.

## Progress Override: 2026-11-04 13:53 UTC

SR3D run1793552465 completed epoch25 full17726 last semantic alignment
Top1 validation:11686/9455 hits (Acc@0.25/0.50=65.925759%/53.339727%).
This is not a new best:epoch15 remains11743/9434;epoch25 is -57/+21
hits versus best and remains616 Acc@0.25 hits short of target12302.
The current-run retention audit at13:46:40UTC verified epoch25's own CPU
epoch,1134 model tensors,nonempty optimizer/scheduler,SHA
dfc15899a0abd0a72d4034f48d76046af9630385ceda2c7be17c0b4ada1b54bb,
and retained it as latest alongside best15. The watcher removed only the
older preserved24 checkpoint;its metric remains recorded. Full audit:
`docs/results/mcln_pvground_20261101/sr3d-epoch25-current-audit-20261104T1353.json`.

At13:53UTC SR trainer92354/keeper129032/retention148725 remained live;
epoch26 had started at305/6486. Its original scratch config and budget
remain unchanged. No error lines were reported and recent logged losses
were finite. NR last verified13:29:52UTC at epoch67 3064/3742,with all
three processes live,original config unchanged,finite recent losses and
no error lines. Current NR progress needs a fresh check before reporting
any later state. Best NR66=4255/3435 remains below target4724 by469 hits;
best SR15=11743/9434 remains616 hits below target12302;Scan best B2=
5483/4656 remains222/289 hits short of5705/4945. No target is met.

The full-history collector collided with the SR checkpoint transition
after epoch25;the later scoped retention-state inspection resolved the
epoch25 metric and checkpoint audit. No manual retention or training
operation was performed.

## Progress Override: 2026-11-04 13:04 UTC

Current-only checks through the approved wrapper confirmed NR3D run
1793553113 trainer96960/keeper135075/retention156800 live at epoch67
1265/3742, and SR3D run1793552465 trainer92354/keeper129032/
retention148725 live at epoch25 5418/6486. Both retain original
null-initialized scratch configs, budgets and innovations-off setting;
recent logged losses are finite and no recent error lines were found.
These are progress observations only, not new validation scores. Latest
complete metrics remain NR66=4255/3435 and SR best15=11743/9434,
latest24=11421/9200. Progress receipts:
`docs/results/mcln_pvground_20261101/nr3d-current-scratch-progress-20261104T1301.json`
and
`docs/results/mcln_pvground_20261101/sr3d-current-scratch-progress-20261104T1301.json`.

## Current Override: 2026-11-04 12:51 UTC

NR3D current run1793553113 completed epoch66 full7899 validation:
4255/3435 hits (Acc@0.25/0.50=53.867578%/43.486517%),new best/latest.
Versus epoch64 this is+65/+41 hits; target4724 Acc@0.25 remains469 hits
away. Current epoch66 checkpoint SHA
ca645bf8f191a0609b603b77e829b61e492b6459805125ae012c164b783fcb5b.
CPU audit verified own epoch66,1134 model tensors,nonempty optimizer/
scheduler, and retention manifest selection. Existing watcher at12:47UTC
kept66 and removed only old preserved64/65; all full-epoch metrics remain.
Trainer96960 is live in epoch67 (78/3742 at12:47:24); original scratch
recipe and240-epoch budget unchanged. One finite loss24.1624 appeared in
the recent window; surrounding losses finite and no error lines. This
single finite value is not evidence of instability or a root cause.

Scan remains best B2=5483/4656; completed Span-only weight0.1 treatment
is5483/4608 (+13/+40 versus matched control5470/4568, but0/-48 versus
B2), exploratory single-seed/epoch only. SR latest full remains24=
11421/9200,best15=11743/9434; original run/config unchanged. No dataset
target has been met. Exact new NR receipt:
`docs/results/mcln_pvground_20261101/nr3d-current-full-history-20261104T1250.json`.

## Current Override: 2026-11-04 12:13 UTC

Canonical user handoff synchronized at12:13UTC:
`C:/Users/gb/Desktop/document/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md`.

Span-only ScanRefer matched pilot completed both full9508 validations and
exited0. Control weight0=5470/4568; treatment weight0.1=5483/4608
(57.667228%/48.465924%),+13/+40 hits versus matched control. Versus
immutable B2=5483/4656 treatment is0/-48; B2 remains global best under
Acc@0.50-first selection. This single-seed/single-epoch result is
exploratory, not a target achievement or robust causal/core-innovation
claim. CPU audits verified both own epoch1,1168 tensors,nonempty
optimizer/scheduler and SHA; all1154 frozen non-Span tensors equal B2 and
14 Span tensors changed. Treatment SHA
6acbcf58351b9ddae8499a6e65392e52c73647e537b0f50a02f5aff4aed54be9;
control SHA f7082ff3b1fd16572eed9b9dc13433321e2b2ce79951e8683519ac4bb3ce755d.
No cleanup or promotion; preserve B2 and both pilot checkpoints.

Latest full current-run metrics at12:13UTC:
NR3D best64=4190/3394,latest65=4095/3254 over7899; latest SHA
b85364f8c13f160f3333d708cafc1c6e0cc17fd03954f70d8b6df0f821dc084b.
SR3D best15=11743/9434,latest24=11421/9200 over17726; latest SHA
391fbb769d7a0cd352faa0876dd89837fece27046e9e6a16f9213beb17a8da6b.
Both original scratch trainers live; configs and full budgets unchanged.
Targets still unmet:Scan best222/289 hits short of5705/4945;NR best534
hits short of4724 Acc@0.25;SR best559 hits short of12302 Acc@0.25.
Use same-checkpoint last semantic Top1 only;never combine thresholds from
different epochs. Full-history receipts and updated tables are archived
in docs/results/mcln_pvground_20261101/.

## Current Override: 2026-11-04 11:35 UTC

Remote receipts11:34:40-47UTC:Scan treatment3419/6081,controller265532/
launcher268814/GPU268817 live;34 train losses finite,last6.9048/6.9039/
6.9074. NR65=3211/3742,trainer/keeper/retention live,70 losses finite,
last9.6084/9.6252/9.6497,rough validation ETA~7min. SR25=1362/6486,
trainer/keeper/retention live,64 losses finite,last9.4725. Original
NR/SR null-init configs and full budgets unchanged. No new full scores.
Best remain Scan B25483/4656,NR64 4190/3394,SR15 11743/9434.
Receipts archived;last published branch3964552.

Remote receipts11:25:27-32UTC:Scan treatment2369/6081,controller265532/
launcher268814/GPU268817 live;23 train losses finite,last6.9396/6.9367/
6.9335,no errors. NR65=2545/3742,trainer/keeper/retention live,69 recent
losses finite,last9.6311/9.6084/9.6252. SR25=937/6486,trainer/keeper/
retention live,64 recent losses finite,last9.4817. Configs unchanged;
no complete new metrics. Best remain Scan B25483/4656,NR64 4190/3394,
SR15 11743/9434. Push recovered11:26UTC;queued commit chain and current
progress receipts published asf3d3caf. No local model/tests.

11:14:29UTC treatment reached1100/6081;controller265532/launcher268814/
GPU268817 live,5.5GiB GPU memory.11parsed train-only losses all finite,
last7.0651/7.0369/7.0132. Rough remaining train44min then full9508
validation. No result yet.

11:09:23UTC treatment real update gate passed:step1/2 finite losses
6.557239/7.686158,Span14/14 updated with finite nonzero gradient/delta,
frozen parameters/buffers unchanged. At520/6081,~1.89batch/s,5 recent
train losses finite,last7.2928/7.2559/7.1997;~49min train plus full val.
No score yet;require comparison to control and B2. Receipt
scanrefer-span-only-progress-20261104T1109.json.

11:04:45UTC follow-up:Scan controller265532/GPU268817 live on treatment;
launcher268814 elapsed6m16s,own GPU memory1168MiB. Treatment log confirms
independent model-only B2 load1168 tensors with fresh optimizer/scheduler;
no first-update witness at that observation,so real optimization was not
yet verified. Controller owns transition;do not duplicate launch.
SR trainer92354/keeper129032/retention148725 live;epoch24 validation
completed and epoch25 started11:04:48. No new full score at that timestamp.

Scan Span-only control completed10:58:21UTC,full9508=5470/4568,
57.530501%/48.043753%,-13/-88hits versus B2. CPU own1/1168 tensors/
optimizer/scheduler/SHA passed;1154non-Span tensors exactly equal B2.
SHAf7082ff3b1fd16572eed9b9dc13433321e2b2ce79951e8683519ac4bb3ce755d.
Existing controller265532 started independent treatment0.1 at10:58:35UTC;
actual11:01launcher268814live,loading data,no GPU/update witness yet.
No duplicate launch. Compare against matched5470/4568 and B25483/4656.
Control-only gain would not establish beating B2. No recipe changes.
Collector now separates explicit Train/Eval headers:521.9311 was Eval100,
not a training spike. Exact log context archived,not a divergence diagnosis.
See PVG_SPAN_ONLY_COMPARISON.md and progress-20261104T1101 receipt.
NR best/latest64=4190/3394;SR24 actual10:57validating902/1478,
live trainer/keeper/retention,finite losses/no errors,ETA11:05UTC.
All targets unmet. No local model/test execution.

## Previous Override: 2026-11-04 10:51 UTC

NR64 full7899=4190/3394 (53.044689%/42.967464%),new best/latest64,
up29/63hits versus prior best59.025 target4724 remains534hits away.
CPU own64/1134 tensors/nonempty optimizer/scheduler/SHA passed,CUDAoff:
1b29f09b8eda0db5baab229fb35e31fd345adda225f6f101aa5d9efed79304f4.
Existing watcher kept64 and removed preserved59/63 at10:50:39UTC.
All full metrics/audits retained;raw nr3d-current-full-history-20261104T1050.json.
Trainer96960 live,next epoch begun,original scratch recipe/budget unchanged.
Scan10:49:25UTC controller265532/GPU266278 live,control validation124/1189;
no full result/final frozen audit yet.61recent parsed losses finite,
including isolated521.9311;no failure inferred from this alone. Existing
pipeline owns treatment transition,ETA full control10:58-11:02UTC.
SR10:49:27UTC trainer/keeper/retention live,validation227/1478,finite
recent losses,no errors,full24 ETA about11:05UTC. No new Scan/SR scores.
No local ML execution or recipe changes;all targets remain unmet.

## Previous Override: 2026-11-04 10:27 UTC

10:37UTC cohort supplement:B2025 selected gate-family failures3875
split411no valid final candidate/2350valid outsideTop10/1114valid inTop10.
Thus3464have another valid candidate,3337multiple-instance. B2050 split
1155/2127/1068,total4350. These are disjoint observed categories,not
recoverable gains or isolated Reader causes. See capacity note/raw
scanrefer-span-continuous-oracle-cohorts-20261104.json.
Actual10:36-10:37UTC Scan control4768/6081,NR3251/3742,SR6038/6486;
trainers live,losses finite,scratch configs unchanged,no new full score.

New evidence changes the next-method scope:continuous-gate oracle on
archived B2 selected queries reaches5633/5158 over9508. Even GT-selected
gates leave Acc025 72hits below5705target;gate-only tuning of this fixed query/
native/reference family cannot reach60%. This is not an achieved score,
global MCLN bound,or proof that one gate setting attains both thresholds.
Reader archive oracle5629/5150. Breakpoint proof plus remote numerical
witnesses64random cases x1331grid points passed;no oracle is within1e-9
of either threshold. See docs/PVG_SPAN_FIXED_QUERY_CAPACITY.md and raw
scanrefer-span-continuous-oracle-summary-20261104.json. No model imports
or new forward. Finish registered Span matched trial for learnability,
then improve selection or upstream geometry from actual remaining errors;
do not assume gate supervision alone can deliver joint60/52. No recipe
or original NR/SR budget changed.

Latest actual Scan check10:19:32UTC:controller265532/GPU266278 live,
control2769/6081. Scoped collector now reports recent periodic losses
using existing NR/SR parsing:27values finite,last6.9144/6.9086/6.9102,
no error lines. This fixes observation of losses beyond the two initial
update witnesses;training source/recipe unchanged. Full validation and
final1154-tensor freeze audit pending,rough ETA10:55-11:05UTC.
Evidence:scanrefer-span-only-progress-20261104T1019.json.

10:11UTC offline evidence supplement:archived selected-gate audit found
B2 projection-proxy050 gains451,but only35 have any raw gate outside
[0,1];416have no outside gate. Reader39of486,remaining447interior.
Saturation therefore does not describe most measured projection-gap
samples;it is not a training-gradient or causal finding. No gate function,
network,loss weight or active recipe changed. See Span comparison doc
and scanrefer-span-proxy-saturation-summary-20261104.json. GT proxy/grid
are diagnostic only;sampled grid is not a certified continuous bound.

Scan Span-only trial is actively training its matched control. Controller
265532 and launcher266273 were live;GPU worker266278 used5472MiB.
Control started09:49:09UTC from independent B2 model-only initialization
with fresh optimizer/scheduler,weight0,seed0. Its own first two steps
(batch6/12) passed:finite losses6.5439133644104/7.6695380210876465,
Span14/14 tensors updated at both steps,and frozen parameters/buffers
unchanged. At10:00:03UTC control reached531/6081 (~1.92batches/s).
Rough full-control validation ETA10:55-11:05UTC;update from actual speed.
Existing screen pvg_scanrefer_span_only owns control->span transition;
do not duplicate launch. Treatment weight0.1 starts independently from B2.
No full result yet. At each save require full9508,CPU own epoch/model/
optimizer/scheduler/SHA plus exact equality of1154 non-Span tensors to B2.
Evidence:scanrefer-span-only-progress-20261104T1000.json and
docs/PVG_SPAN_ONLY_COMPARISON.md. Preflight12batches/two steps also passed.

NR63 completed full7899=4150/3291 (52.538296%/41.663502%).
Best59 remains4161/3331 (52.677554%/42.169895%);latest is-11/-40hits.
CPU own63/1134 tensors/nonempty optimizer/scheduler/SHA verified09:52UTC.
SHA674e4982928ce6052bd9ab136be963513e35aebe0b0a1a5892093a30fe3d08a5.
Existing retention kept59/63 and removed62;full epoch metrics remain.
SR latest full23=11388/9006,best15=11743/9434;SR24 was live09:41UTC.
NR/SR original scratch configurations and full budgets remain unchanged.
All targets remain unmet. Local work was file/Git/result processing only.

## Previous Override: 2026-11-04 09:46 UTC

New Scan trialscanrefer_span_only_20261104,screenpvg_scanrefer_span_only,
controller265532 owns preflight/control/span;started09:42:21UTC.
At09:45:56 preflight loading training data,no optimizer/GPU witness yet.
Remote preparation and actual-method optimizer/mode unit test passed;
full-model12-batch/two-step frozen-state gates pending. Do not duplicate.
Both full conditions independently from B2,original Reader,fresh optimizer/
scheduler,only14 Span tensors trainable,others/BN frozen. Control gate
projection weight0,treatment0.1,seed0/batch8/accum6/moduleLR2e-5,one full
48655training epoch+9508validation each. Full save must verify all1154
other model tensors unchanged. See PVG_SPAN_ONLY_COMPARISON.md.

Archived-box stdlib diagnostic:B2 current0504656,GT projection5103,
125-grid oracle5152;Reader4609/5090/5136. Reconstruction within5e-7.
GT diagnostics,not achieved accuracy;motivate isolating learning before
replacing the proxy. Source CRLF patch rejection halted before execution;
explicit finish after verified diff succeeded. No fallback/restart.

09:41UTC NR63=3579/3742,SR24=3470/6486;actual trainer/keeper/retention
live,finite loss,no errors,original configs/budgets unchanged. No new
complete metric. All remote work approved wrapper,no local model/tests.

## Previous Override: 2026-11-04 09:24 UTC

Reader sample audit completed09:20:26UTC,exit0,full9508 forward/semantic
records,evaluator agreement and exact ordered identity/points/GT pairing
to B2 passed. Reeval5471/4609,training validation5471/4608 kept separate.
Same own epoch1/1169-tensor checkpoint SHA67cef44cd6ec253806eced03b1ca79f115d13d4e681c9da69730ca767b0fda35.
No new trained epoch or determinism claim. Diagnostic controller/GPU
exited;do not relaunch.

050 gains105/losses152 net-47;lost cases3 no valid candidate,21 valid
outsideTop10,128 valid insideTop10. Of152 losses,52 selected reference
boxes valid,11 native valid,union62. Overall oracle8252vsB28260,Top10
5917vs6021. Prioritize valid-candidate selection alongside fusion
reliability;not isolated module causality or additive recoverable gains.
025 gained100/lost112 net-12. Best B2 unchanged5483/4656;matched trained
control5477/4620 separate. No new training or module-source change.
See PVG_READER_COMPETITOR_EXPERIMENT.md and
PVG_READER_QUALITY_ALIGNMENT_REFERENCE.md for detailed results,pinned
TOOD/VarifocalNet review and MCLN matcher/BBF constraints. Duplicate
neighbor rate is not measured. Raw summary/log and separate evaluation
event CSV archived;training all-epoch tables unchanged.

NR63/SR24 trainers/keepers/retainers last actually live09:08UTC;original
null-init configs and240/140budgets unchanged. Latest complete/best below
unchanged. Two-hour automation updated,quiet preference preserved.
No local model/test operation;all remote work through approved wrapper.

## Previous Override: 2026-11-04 09:08 UTC

Read-only Reader trial full9508 audit launched09:08:04UTC,screen
pvg_scanrefer_reader_result_audit,controller263822;observed live loading
validation data at09:08:52UTC. Remote CPU own epoch1/1169 tensors/
optimizer/scheduler/SHA verified. No new training or parameter updates.
At09:12:05UTC GPU263830 live,2280/9508 forward and semantic records,
no footer;remaining~8minutes,rough completion09:20UTC.
Directory:offline_audits/reader_result_20261104. Do not duplicate launch.
Use tools/check_scanrefer_reader_result_audit.sh via approved wrapper;
after completion run tools/summarize_scanrefer_reader_result_audit.sh.
Require9508 records,evaluator agreement,and exact identity/points/GT
pairing with B2 before interpreting gains/losses and candidate coverage.
This compares different trained checkpoints,not isolated Reader causality
or the matched trained control. See PVG_READER_COMPETITOR_EXPERIMENT.md.

09:08UTC NR63=1293/3742,SR24=1921/6486. Actual trainer/keeper/retention
PIDs all live,original scratch configs unchanged,finite recent losses,
no errors. Latest complete/best below unchanged. Next full results roughly
NR09:50UTC/SR11:00UTC;use actual progress. No local model/test operation.

## Previous Override: 2026-11-04 08:55 UTC

### Latest complete metrics

All metrics are same-checkpoint last semantic alignment Top1; percentages
are hits over the complete validation split.

| Dataset / run | Latest complete | Best by primary metric | Target / remaining from best |
| --- | --- | --- | --- |
| ScanRefer | Reader epoch1: 5471/4608 of9508 (57.541018%/48.464451%) | Stage B epoch2: 5483/4656 (57.667228%/48.969289%), Acc@0.50 primary | 5705/4945; short by222/289 hits |
| NR3D `1793553113` | epoch62: 4141/3425 of7899 (52.424358%/43.359919%) | epoch59: 4161/3331 (52.677554%/42.169895%), Acc@0.25 primary | 4724 Acc@0.25; short by563 hits |
| SR3D `1793552465` | epoch23: 11388/9006 of17726 (64.244612%/50.806725%) | epoch15: 11743/9434 (66.247320%/53.221257%), Acc@0.25 primary | 12302 Acc@0.25; short by559 hits |

NR epoch62 full7899 validation and CPU audit were confirmed by08:53UTC.
The current-run CPU collector verified own epoch62,1134 model tensors,
nonempty optimizer and scheduler,and preserved checkpoint existence. SHA256:
`9b3f429b4696b6d1378d3452bbaa9e274b1aa4bed96914188c36135898648d98`.
Best epoch59 remains keeper-selected and was independently audited with
SHA256 `4d000014f205f23a03ce2d7102dc7e7813359a4a65c17e0e401bffc06ed2885f`.
Epoch62 is -20/+94 hits versus epoch59 on Acc@0.25/Acc@0.50. Receipt:
`docs/results/mcln_pvground_20261101/nr3d-current-full-history-20261104T0854.json`.

SR epoch23 full17726 validation and CPU audit were confirmed by08:26UTC.
The collector verified own epoch23,1134 model tensors,nonempty optimizer
and scheduler,and preserved checkpoint existence. SHA256:
`1863c30fe8e3f9b870ba64162afb89f7ab6d0eda4c09c948ec9e6bd168c30abd`.
Epoch23 is -355/-428 hits versus keeper-selected epoch15. The best15 SHA256
remains `386647b8ecac817614090cb23f11b5d7a4d36b7449f67c07f43f84baf23e3969`.
Receipt:
`docs/results/mcln_pvground_20261101/sr3d-current-full-history-20261104T0826.json`.

At08:55UTC NR trainer96960,keeper135075,and retention watcher156800 were
live in epoch63 at461/3742 on the unchanged null-init scratch recipe.
All68 recent losses were finite, latest9.6938/9.6963/24.1625, with no errors.
At08:54UTC SR trainer92354,keeper129032,and retention watcher148725 were
live in epoch24 at1283/6486 on the unchanged null-init recipe. All64 recent
losses were finite, latest16.5046/9.3763/9.4747, with no error lines.
These isolated finite higher losses do not establish instability. NR's
latest higher finite loss is insufficient by itself to justify a recipe
change. The progress displays estimated about45 minutes to NR training
completion and about1h54m for SR training; these are rough live estimates,
validation is additional. Both recipes remain unchanged.

Scan process state was not queried in this pass, and no Scan run was started.
The one-epoch Reader result remains below its matched control; it does not
justify launching another Scan run without a diagnosis and registered
experiment. All three targets remain unmet. Existing NR/SR retention
watchers remain the only cleanup authority; no manual checkpoint deletion
occurred. All remote checks used the approved wrapper, and no local
model/test operation was run.

## Current Override: 2026-11-04 06:12 UTC

### Latest complete metrics

All values are same-checkpoint `last_semantic_alignment_top1`; ScanRefer
uses Acc@0.25/0.50, NR3D and SR3D also report those thresholds. Counts are
hits over the complete validation split.

| Dataset / run | Latest complete | Best by primary metric | Target |
| --- | --- | --- | --- |
| ScanRefer Reader `1793763075` | 5471/4608 of9508 (57.541018%/48.464451%), epoch1 | Stage B2: 5483/4656 (57.667228%/48.969289%), selected by Acc@0.50 | 5705/4945 hits |
| NR3D `1793553113` | epoch59: 4161/3331 of7899 (52.677554%/42.169895%) | epoch59: 4161/3331, selected by Acc@0.25 | 4724 Acc@0.25 hits |
| SR3D `1793552465` | epoch22: 11352/9308 of17726 (64.041521%/52.510437%) | epoch15: 11743/9434 (66.247320%/53.221257%), selected by Acc@0.25 | 12302 Acc@0.25 hits |

Reader epoch1 full validation and own-checkpoint CPU audit completed at
05:10UTC: 1169 tensors, own epoch1, nonempty optimizer/scheduler, SHA256
`67cef44cd6ec253806eced03b1ca79f115d13d4e681c9da69730ca767b0fda35`.
It is -6/-12 hits versus matched control5477/4620 and -12/-48 versus B2.
The result does not support a benefit from this one-seed/one-epoch
candidate-relative Reader residual and does not prove why. `competitor_scale`
is not established as a cause. Run details and limitations are in
`PVG_READER_COMPETITOR_EXPERIMENT.md`.

NR epoch59 completed full7899 validation at05:49UTC with4161/3331 hits.
CPU audit verified own epoch59,1134 tensors,nonempty optimizer/scheduler,
SHA256 `4d000014f205f23a03ce2d7102dc7e7813359a4a65c17e0e401bffc06ed2885f`.
The keeper manifest selected59; the existing retention watcher remains the
sole cleanup authority. Best is563 Acc@0.25 hits short of target4724.
At05:59UTC trainer96960 was live at279/3742 in epoch60; original null-init
scratch config unchanged, recent losses finite and no recent errors.

SR epoch22 completed full17726 validation at05:50UTC with11352/9308 hits
(64.041521%/52.510437%). CPU audit verified own epoch22,1134 tensors,
nonempty optimizer/scheduler,SHA256
`47d010804e89282fcb9779f2c91738d15e5fbe5be4e729e3c59af746d23996b9`.
Best remains epoch15=11743/9434; epoch22 is -391/-126 hits versus best
and950 Acc@0.25 hits short of target12302. The existing watcher is the
only retention authority. At05:56UTC trainer92354 was live at478/6486 in
epoch23, with original null-init recipe unchanged, finite recent losses
and no recent errors.

At06:12UTC current-only checks confirmed NR epoch60=1205/3742 and SR
epoch23=1196/6486. Both trainers, keepers and retention watchers were
live; original null-init scratch configs unchanged, recent logged losses
finite and no error lines. NR had one finite24.32 value followed by values
near9.8; this isolated point is not enough to diagnose instability or
change the recipe. No new full validation, checkpoint promotion or
retention event is claimed from this progress snapshot. Recheck near the
actual validation windows; do not manually manage checkpoints.

The three targets remain unmet. Do not mix thresholds across checkpoints.
NR/SR continue their original full scratch budgets; no Scan trainer is
active and no additional Scan launch is authorized by this exploratory
negative result. Full Reader result and one-row table are archived at
`docs/results/mcln_pvground_20261101/scanrefer-reader-competitor-result-20261104.json`
and `scanrefer-reader-competitor-20261104.csv`. NR59/SR22 receipts are
`nr3d-current-full-history-20261104T0556.json` and
`sr3d-current-full-history-20261104T0551.json`. The all-epoch table
contains all three new complete rows. No local model tests or execution
were performed.

## Current Override: 2026-11-04 05:02 UTC

NR3D run1793553113 completed epoch58 full7899 same-checkpoint last
semantic alignment Top1 validation: 4103/3298 hits, Acc@0.25/0.5 =
51.943284%/41.752121%. Best remains epoch49=4123/3325
(52.196481%/42.093936%); epoch58 is -20/-27 hits versus best and is
621 Acc@0.25 hits short of the 4724 target. The CPU audit verified own
epoch58, 1134 model tensors, nonempty optimizer/scheduler, retained path,
and SHA256
705115d68949d5963e8f7c518b6ce6b43f90f9ce37bbe8ea90d7c0459c92a417.
The existing retention watcher kept49/58 and removed only preserved57;
the epoch57 metric remains recorded. At05:02UTC the trainer, keeper and
watcher were live at374/3742 training progress. Recent logged loss values
included one finite24.8732 among values near9.8; no recent errors or
nonfinite losses were found. This isolated finite observation is not
enough to diagnose instability; leave the recipe unchanged and monitor.
Receipt: `docs/results/mcln_pvground_20261101/nr3d-current-full-history-20261104T0458.json`.

ScanRefer Reader run1793763075 reached6068/6081 training batches at04:59
and entered full validation. At05:01UTC it was at168/1189 validation
batches, with the GPU worker active. No full9508 result or own-checkpoint
audit exists yet.

SR3D run1793552465 was live at epoch22 batch5313/6486 (82%) at05:02UTC;
recent losses were finite and no errors were found. Its original null-init
scratch recipe remains unchanged. No new SR3D validation was complete.
All current metrics and the NR58 row are in the all-epoch table; do not
infer a ScanRefer gain from the in-progress validation.

## Current Override: 2026-11-04 04:47 UTC

Current-only remote checks through `work/remote_pvground.ps1`:
ScanRefer Reader run1793763075 was live at5118/6081 training batches
(84%) at04:46UTC; GPU worker254229 was active, using22072MiB. The first
SSH attempt timed out during banner exchange; a retry connected and
confirmed the run was progressing. No reader completion receipt or full
9508 validation exists yet; do not report an accuracy gain.

NR3D run1793553113 trainer96960/keeper135075/retention156800 were live at
epoch58 batch3496/3742 (93%) at04:47UTC. The last three logged losses
9.8165/9.8196/9.8267 were finite and no recent errors were found.
Original scratch config remains null-init, innovations off, batch12,
accumulation4, LR1e-4, backbone1e-3, max240, decay150.

SR3D run1793552465 trainer92354/keeper129032/retention148725 were live at
epoch22 batch4616/6486 (71%) at04:47UTC. The last three logged losses
9.6489/9.6431/9.6444 were finite and no recent errors were found.
Original scratch config remains null-init, innovations off, batch12,
accumulation2, LR1e-4, backbone1e-3, max140, decay30,40.

No complete validation changed. Latest complete remains ScanRefer Stage B4
5485/4623 over9508 (57.688263%/48.622213%), with B2 still best at
5483/4656 (57.667228%/48.969289%) by the primary Acc@0.50 criterion;
NR3D latest57=4090/3305 and best49=4123/3325 over7899; SR3D latest21=
11579/9206 and best15=11743/9434 over17726. Reader remains pending;
do not combine its future metrics with these prior checkpoints.

## Current Override: 2026-11-04 03:10 UTC
SR3D run1793552465 epoch21 completed full17726 semantic Top1 validation:
11579/9206 hits (65.322126%/51.935011%). Best remains15=11743/9434;
latest21 is -164/-228 hits versus best. CPU audit verified own epochs,
1134 model tensors,nonempty optimizer/scheduler and SHA for best15/latest21.
Epoch21 SHA 1114d6f8d0e9a087fd2231692c54b194c0470736da170558cae9cb9407cc22c1.
The existing watcher kept15/21 and removed preserved20 at03:08:49UTC.
Trainer92354 is live in epoch22, unchanged scratch recipe; NR trainer96960
is live in epoch57. Scan remains idle. Receipt:
`docs/results/mcln_pvground_20261101/sr3d-current-full-history-20261104T0309.json`.

## Current Override: 2026-11-04 03:00 UTC

NR3D run1793553113 epoch56 completed full7899 last semantic Top1 validation:
4090/3316 hits (51.778706%/41.979997%). Best remains49=4123/3325;
latest56 is -33/-9 hits versus best and634 Acc@0.25 hits below target4724.
The02:58UTC collector verified best49/latest56 own CPU epochs,1134 tensors,
nonempty optimizer/scheduler, SHA and preserved links. Epoch56 SHA:
b5b62f509cf9776ba4d3b87ea52fc448f7951edce194243dac6dcd740e19975f.
Retention applied02:57:18UTC, kept49/56 and removed only preserved55;
all epoch metrics and historical audits remain. Receipt:
`docs/results/mcln_pvground_20261101/nr3d-current-full-history-20261104T0259.json`.
Trainer96960 was live at02:59UTC in epoch57, with original scratch
configuration unchanged.

At02:58UTC SR3D run1793552465 remained live in epoch21 full validation
at677/1478 batches, not a complete result. Original scratch config and
retention watcher unchanged. Last complete20=11445/9060, best15=
11743/9434 over17726. No ScanRefer restart or local model execution.

## Current Override: 2026-11-04 02:00 UTC

At01:57UTC NR3D run1793553113 completed full7899 validation for epoch55:
4108/3255 hits, Acc@0.25/0.5=52.006583%/41.207748%. Best remains
epoch49=4123/3325; latest55 is -15/-70 hits versus best. The target4724
Acc@0.25 hits is616 above latest and601 above best; the target is unmet.
CPU collector verified best49/latest55 own epochs,1134 model tensors,
nonempty optimizer/scheduler, preserved paths and SHA. Latest55 SHA:
721e834de0e1cd7605e66e06fd5a68fe44cc54926b0b91299fc595df2159873d.
Best49 SHA remains
0987ce9208dca84686a45a7d80bad4f93f126f34be9c8303899a1b6bf68ed66b.
Retention applied at01:57:07UTC, kept49/55 and removed only preserved54;
epoch54's metric row remains with its weight marked removed. Full-history
receipt is `docs/results/mcln_pvground_20261101/nr3d-current-full-history-20261104T0156.json`.
At01:59UTC trainer96960 was live in epoch56 at277/3742 on the unchanged
batch12/accum4/LR1e-4/backbone1e-3/max240/decay150 null-init recipe;
innovations remain disabled.

At01:58UTC SR3D run1793552465 remained live with its log advancing and
the original batch12/accum2/LR1e-4/backbone1e-3/max140/decay30,40
null-init recipe unchanged. No new complete validation was present.
Latest remains epoch20=11445/9060; best15=11743/9434 over17726.
Both datasets remain below their Acc@0.25 targets; no recipe or retention
policy changed. SR full-history receipt:
`docs/results/mcln_pvground_20261101/sr3d-current-full-history-20261104T0156.json`.

ScanRefer remains idle. Its B2 full-set bypass and one-epoch Span
supervision pilot are already complete; the pilot's +7/+34 hits over its
matched control is only +1/-2 versus strong B2, so it is exploratory, not
a robust gain. Do not restart Scan without explicit authorization.
No local Windows/WSL model tests, imports, forward, evaluation or training.

## Current Override: 2026-11-04 01:29 UTC

At01:28UTC NR3D run1793553113 trainer96960/keeper135075/retention156800
remained live;epoch55=2350/3742. Original batch12/accum4/null-init
recipe unchanged;69 recent losses finite,latest three9.9954/9.9468/9.9411,
no errors. Latest full54=4004/3233,best49=4123/3325;next full validation
estimated01:55-02:00UTC.

SR3D run1793552465 trainer92354/keeper129032/retention148725 remained
live;epoch21=2716/6486. Original batch12/accum2/null-init recipe
unchanged;64 recent losses finite,latest three9.6869/9.6754/9.6662,
no errors. Latest full20=11445/9060,best15=11743/9434;next validation
estimated03:00-03:15UTC.

The source-reference update is published with commits ebfccfe and73cac54.
DETR/3DETR inform only explicit assignment eligibility/unmatched-query
handling;no loss or matcher was copied. The MCLN auxiliary matcher is
disconnected in the audited path,and verified human anchor IDs were not
present in the inspected ScanRefer training records. This is a design
constraint,not a demonstrated root cause. No training code/config changed.

## Current Override: 2026-11-04 01:22 UTC

At01:21UTC NR3D run1793553113 was live with trainer/keeper/retention
96960/135075/156800,epoch55=1852/3742. The original null-init recipe
was unchanged;68 recent losses were finite,latest9.9468,no errors.
Latest full remains54=4004/3233,best49=4123/3325;next full result is
estimated01:55-02:00UTC.

At01:21UTC SR3D run1793552465 was live with trainer/keeper/retention
92354/129032/148725,epoch21=2385/6486. The original null-init recipe
was unchanged;64 recent losses were finite,latest9.6754,no errors.
Latest full remains20=11445/9060,best15=11743/9434;next full result is
estimated03:00-03:15UTC. Retention watchers remain the sole cleanup
authority.

Source research: official DETR and3DETR implementations use explicit
Hungarian assignment for object-level auxiliary losses and define the
unmatched-query path. The transferable rule is assignment eligibility and
gradient routing,not direct port of their matching costs. MCLN's audited
auxiliary target currently has no connected matcher/loss,and verified
human anchor IDs are absent from the inspected ScanRefer training records.
This supports further train-only label-coverage/mapping verification before
a Reader change;it does not establish a root cause or accuracy gain. Details
and repository references:PVG_READER_ANCHOR_SUPERVISION_REFERENCE.md.
No source change,training launch or target attainment is claimed.

## Current Override: 2026-11-04 01:12 UTC

At01:11UTC NR3D run1793553113 remained live with trainer96960,
keeper135075 and retention156800. It advanced to epoch55,1139/3742
batches; its unchanged scratch/null-init recipe had69 recent finite
logged loss averages. The last three were24.3912,9.946,9.9954; the isolated
higher value is followed by values near9.95 and does not establish
instability. No error lines;GPU22837MiB/77%.
Latest complete remains epoch54=4004/3233;best49=4123/3325 over7899.
The current result is720 Acc@0.25 hits below target4724;best is601 short.
Epoch55 full validation estimate is01:55-02:05UTC.

At01:11UTC SR3D run1793552465 remained live with trainer92354,
keeper129032 and retention148725. It advanced to epoch21,1914/6486
batches;the unchanged scratch/null-init recipe had64 recent finite losses
(latest9.6869),no error lines and GPU22924MiB/52%. Latest complete
epoch20=11445/9060;best15=11743/9434 over17726. Latest is857 Acc@0.25
hits below target12302;best is559 short. Epoch21 full validation estimate
is03:00-03:15UTC. Retention watchers remain the only checkpoint cleanup
authority.

ScanRefer remains idle;the completed B2 bypass and matched one-epoch Span
supervision pilot are exploratory,not a robust gain. Do not restart Scan
training without explicit user authorization. No new complete validation,
config/source change or local ML operation occurred in this status check.

## Current Override: 2026-11-04 00:00 UTC

NR3D run1793553113 completed epoch53 full7899 validation:
4107/3258 hits at Acc@0.25/0.5 (51.993923%/41.245727%). Best remains
epoch49=4123/3325. The 4724 Acc@0.25 goal is617 hits above latest and
601 above best. The first collection fell in a save/retention window;
the unchanged retry after180seconds passed. CPU audit confirms best49 and
latest53 own epochs,1134 model tensors,nonempty optimizer/scheduler,
SHA and preserved checkpoint links. Epoch53 SHA:
e8d6d69f9af514f3f5d51973c64642d3f7a992eb362723527d4eda60a5dba991.
Trainer96960 was live at23:59UTC at435/3742, with original scratch
configuration and null initialization unchanged. Full receipt:
`docs/results/mcln_pvground_20261101/nr3d-current-full-history-20261103T2358.json`.

SR3D trainer92354 remained live at23:59UTC. GPU use was21%, train-log
size8954452bytes and latest periodic line23:50:30UTC, confirming log
advance. Latest complete result remains epoch19=11426/9256 and best
epoch15=11743/9434 over17726; target12302 remains559 Acc@0.25 hits
beyond best. No new full SR validation. Original scratch config and null
checkpoint paths remain unchanged.

ScanRefer has no active trainer; its completed single-seed Span pilot
remains exploratory. No repeat has been authorized or started. No local
model tests/imports/forward/evaluation/training.

## Current Override: 2026-11-03 23:09 UTC

NR3D remains at best49=4123/3325 and latest full epoch52=4048/3238 over
7899; its CPU-audited full-history receipt and epoch52 SHA are recorded
below in the22:58 override and in the corresponding receipt. The trainer
was live at335/3742 at22:58UTC, on its unchanged scratch configuration.

SR3D is live, not stalled. At23:08:53UTC trainer92354 was present and the
GPU reported22924/24564MiB. The train log grew from8760660 bytes at22:58
to8821668 bytes at23:08, and its latest periodic timestamp advanced from
22:56:35 to23:07:24UTC. The old visible2515/6486 tqdm tail was stale;
the helper exposes no current microbatch count, so do not diagnose a stall.
Latest full validation remains19=11426/9256, best15=11743/9434 over17726.
Live status receipt:
`docs/results/mcln_pvground_20261101/sr3d-current-scratch-progress-20261103T2308.json`.
No new full validation. Both runs retain original scratch settings and
null initialization paths.

No ScanRefer trainer is active. The completed Span pilot remains
exploratory (+7/+34 versus control, +1/-2 versus strong B2); no repeat
has been authorized or started. No local model tests/imports/forward/
evaluation/training.

## Current Override: 2026-11-03 22:58 UTC

NR3D run1793553113 completed full7899 validation epoch52=4048/3238
hits at Acc@0.25/0.5 (51.246993%/40.992531%). Best remains epoch49=
4123/3325; latest52 is 676 hits from the4724 target, while best is601
hits short. Remote CPU audit at22:56:37UTC verified best49/latest52 own
epochs,1134 tensors,nonempty optimizer/scheduler and SHA. Epoch52 SHA:
a94838e87ddace0e0be872e6c8ac7216a9ad92dc3692b5a76ed5db59e6954ff9.
At22:58UTC trainer96960 remained live at335/3742; finite recent losses,
no error lines, original scratch recipe and null initialization unchanged.

SR3D trainer92354,keeper129032 and retention148725 were live at22:58UTC.
Latest complete result remains epoch19=11426/9256, best epoch15=11743/
9434 over17726; target12302 Acc@0.25 remains559 hits beyond best.
The helper's visible2515/6486 tail was stale; the23:08 check confirmed
advancing log timestamp and bytes. No new complete SR validation.
CPU-audited full-history receipts:
`docs/results/mcln_pvground_20261101/nr3d-current-full-history-20261103T2255.json`
and `sr3d-current-full-history-20261103T2255.json`.

ScanRefer has no active trainer and remains paused pending interpretation
or a user decision to repeat the completed single-seed Span pilot. Its
weight0.1 result remains+7/+34 versus the weaker control but+1/-2 versus
strong B2; it is exploratory, not robust causal evidence. Do not start a
new Scan run. No local model tests/imports/forward/evaluation/training.

## Current Override: 2026-11-03 22:20 UTC

Live check22:19:01UTC: NR trainer96960, keeper135075, retention156800
all live; NR is at1812/3742, GPU22837MiB/14%, recent logged losses finite,
no error lines. Estimated training remainder is about29minutes before
validation; expected full result roughly22:50-23:05UTC.
SR trainer92354, keeper129032, retention148725 all live; SR is at1256/6486,
GPU22924MiB/63%, recent logged losses finite, no error lines. Estimated
training remainder is about1hour50minutes before validation; expected full
result roughly00:10-00:25UTC November4.

NR/SR configs and null initialization paths are unchanged. Latest complete
same-checkpoint last semantic alignment Top1 remains NR best49=4123/3325,
latest51=4100/3295 over7899; SR best15=11743/9434, latest19=11426/9256
over17726. No target is met. Best/latest CPU audits and full history remain
in the22:02 receipts; no new complete validation or retention event yet.
Progress snapshots are in outputs/*-current-scratch-progress-20261103T2219.json.
Check shortly before estimated validation completion, not on a short
fixed-interval loop.

ScanRefer has no active trainer. The fixed-B2 diagnostics and one-epoch
Span supervision pilot are complete; treatment was+7/+34 against its
matched control but+1/-2 against strong B2, so it remains exploratory.
Do not start another Scan run until this is interpreted or a repeat is
chosen. No local model tests/imports/forward/evaluation/training.

## Current Override: 2026-11-03 22:02 UTC

This supersedes the 20:01 snapshot. ScanRefer has no active trainer; do not
restart a new Scan run until the completed Span pilot is interpreted or the
user chooses a repeat. The claim that two-stage fine-tuning stayed below E0
is incorrect: matched B2 improved over official E0 by48/329 hits at025/050.
It remains below the final target.

Completed B2 fixed-checkpoint diagnostics and the registered Span
supervision pilot are documented in PVG_FORWARD_AUDIT.md and
PVG_SPAN_GATE_COMPARISON.md. On full9508, fixed B2 was5483/4656; weight0
control was5477/4620; weight0.1 treatment was5484/4654. Treatment versus
control is+7/+34, but versus strong B2 is+1/-2. This single-seed/epoch
result is exploratory, not robust causal evidence. No Reader rewrite or
new Scan training has started.

At22:01UTC the exact original NR/SR trainer, keeper and retention PIDs were
live and their scratch recipes unchanged. NR is training after full epoch51,
log progress456/3742; SR is training after full epoch19, progress339/6486.
Both retain null checkpoint and pp_checkpoint, innovations off, and their
original240/140epoch budgets.

Latest complete same-checkpoint last-semantic-alignment Top1 results:
NR best epoch49=4123/3325 over7899; latest epoch51=4100/3295.
SR best epoch15=11743/9434 over17726; latest epoch19=11426/9256.
Targets remain NR4724 and SR12302 Acc025 hits. Neither is met.
Best/latest files were CPU-audited for own epochs,1134 model tensors,
nonempty optimizer/scheduler and SHA. NR49 SHA
0987ce9208dca84686a45a7d80bad4f93f126f34be9c8303899a1b6bf68ed66b;
NR51 SHA b1ff22ea55d9cc26dd21b81e7f5221350ccae1ee31de346834d3ce0892aa8841.
SR15 SHA 386647b8ecac817614090cb23f11b5d7a4d36b7449f67c07f43f84baf23e3969;
SR19 SHA 7a179f1a0c49c30b72a28ef2ef1ae1e951ac0c2e7cf94bad6581621c1940338c.

Full-history receipts are in docs/results/mcln_pvground_20261101:
nr3d-current-full-history-20261103T2202.json and
sr3d-current-full-history-20261103T2202.json. The combined CSV includes
NR48-51 and SR19. NR48/50 and SR18 weights were removed by the authorized
best/latest watcher; metrics and retention audits remain. The scoped
collector now uses the actual dataset-level preserved_checkpoints directory
and keeper aliases. No local model tests/imports/forward/evaluation/training.

## Current Override: 2026-11-03 20:01 UTC

Verifiedwait20:01:23UTC:actualspanGPU240217live,3140/6081,
21977MiB71%,no recenterrors,no spancomplete/result. FullETA20:55-21:10
retained;no newmetrics/tables75/11/pilot1. Existingcontroller unchanged.
NR not queried this turn;lastobservabilityfailure19:55UTC is historical,
lastverifiedlive17:59:18 remains. No restart/SSH override.

New live-source audit19:59:10UTC refines the anchor-label finding:
native criterion constructsauxi_target but comments its matcher and
setsauxi_indices=None. Four native losses have0body uses of that argument
and0directauxi_box/auxi_entity_positive_map fields. Do NOT infer an
active explicit anchor-matching objective from dataset field existence.
Other-entity/relation token supervision remains;this is not proof of
absent contextual learning or an audit of all historical optional losses.
Current optional verifier/reranker/SACR/sourceMOE/selector switchesOFF;
nonzero default loss weights do not prove a branch active.
Actual_get_inputs returnspoint_clouds/text;train/eval append batch
labels toend_points aftermodel(inputs). No auxi_box input via that entry.
No inference-leakage allegation or new Reader implementation.
Actual source hashes/line ranges/config/AST blocks in receipt
scanrefer-auxiliary-supervision-source-audit-20261103T1959.json and
PVG_READER_ANCHOR_SUPERVISION_REFERENCE.md.
All read-only/stdlibrary/remote;no model/dataset imports or source mutation.

Scoped scratch health helper:tools/check_current_scratch_progress.sh
only enumerates registered NR1793553113/SR1793552465,never a Scan
or historical fine-tune run. Source/log/config/process reads only,
no model import/checkpoint load/forward/evaluation/training. Use via
work/remote_pvground.ps1 instead of old inspect_sr3d_current_epoch.sh,
which points to a stopped historical module fine-tune.
RemoteSR execution19:36:44UTC verified92354/129032/148725live and
the exact current screen. Training1070/6486,GPU22924MiB;65recent
logged losses finite,latest9.8206,no recent error records.
Original nullcheckpoint/pp,innovationsOFF,batch12accum2/LR1e-4/
backbone1e-3/max140/decay30,40 verified. Next fullSR validation
ETA21:45-22:05UTC from~1.3s/microbatch plus validation,not guaranteed.
Receipt:sr3d-current-scratch-progress-20261103T1936.json.
No new full result;bestSR15/latest18 and tables75/11/pilot1 unchanged.
Scan last actual19:34:01UTC:240217live,1137/6081,no recenterrors;
fullETA20:55-21:10 remains. NR lastlive17:59:18,not newly verified.

New source/data-only evidence19:31:22UTC:actual live240217/source cwd,
use_sacr_source=false,augment_det=true. RawScan TRAIN36665records have
no explicit anchors/anchor_ids/anchors_types/anchor_boxes/path fields;
actual loader initializes anchors/anchor_ids empty. Existing auxiliary
box source picks the parsed class's nearest valid non-target box to GT
target center,not a verified relation-labelled anchor. This does not
prove lack of all contextual supervision or inference leakage. No
runtime heuristic coverage/quality measured. Do not blindly port CoT
anchor-index loss;specify supervision and query mapping first.
Read PVG_READER_ANCHOR_SUPERVISION_REFERENCE.md and raw
scanrefer-train-anchor-annotation-audit-20261103T1931.json.
No dataset/model imports,forward/eval/training or source mutation.

Verified wait19:29:00UTC:actual spanGPU240217live,773/6081,
21977MiB/84%,no recent errors;config unchanged. FullETA20:55-21:10UTC
remains. No new full validation;tables75/11/pilot1 unchanged.
NR requery21393 again failed before status;lastlive17:59:18 remains,
not proof of training exit. No restart or SSH override.
Previous goal turn was progress(control/SR18 audits/source notes/push);
this turn adds actual train-label evidence plus a verified live wait.

Scan pilot control completed19:12:00UTC,full9508=5477/4620,
57.604123/48.590660%,below immutable B2=5483/4656 by6/36hits.
Independent audit19:14:28UTC passed:five source hashes,full semantic
log/evaluator counts,ownCPUepoch1/1168tensors/nonempty optimizer/scheduler/
three constant LR groups/SHA,CUDAoff. SHA
4f18e127321107f9f709c876226437bbc871ec3e5cd101cbf992ceede172f8d8.
This is a separate pilot epoch1,not StageB epoch5. StrongB2 remains.
Controller236004 transitioned19:12:04UTC to spanrun1793733129.
At19:21:55UTC worker240217live,250/6081,GPU21941MiB/83%,no recent
errors. OWN first update epoch1/batch6/size8/step1 loss7.0388007164001465
finite;Reader10/10,Support10/10,Span14/14 actual updates,all gradient/
delta norms finite/nonzero. This is not a complete validation.
Treatment full ETA20:55-21:10UTC from actual1.24microbatches/s.
Do not duplicate the controller or implement Reader before this comparison.
Audit/progress receipts:scanrefer-span-gate-independent-audit-20261103T1914.json,
scanrefer-span-gate-progress-20261103T1921.json.

SR18 full17726=11563/9150,65.231863/51.619091%,+27/+113vs17.
Best remains15=11743/9434;latest18 trails it180/284hits.
OwnCPU15/18/1134/nonempty optimizer/scheduler/SHA verified,CUDAoff.
Latest18 SHA9505e32be485ef25c9fabca4f37cd5bbcc442bca78a49d6c5ae6367b4e8e23ca.
Actual original92354live19:15:35UTC,next training95/6486.
Retainer19:15:37UTC kept15/18,removed only preserved17;free
19739729920->20532826112bytes. Receipt observed19:20:21UTC;
PowerShell local display14:20:21 is NOT UTC. Initial collector save/keeper
window resolved by unchanged retry after180s;no fallback or trainer edit.
All original SR140/NR240 budgets and null initializations remain.

NR latest wrapper query still exited1 before status:
"Connection closed by UNKNOWN port 65535". Last verified liveness
17:59:18UTC;last audited full47=4114/3248,SHA
a89e77471e8a5461838f60648651be891baaa658cb79a0f5d446fd54b8f673c3.
No fresh liveness,termination or root-cause claim;no restart/SSH override.

HistoricalCSV75rows(allSR1-18),Scanrestart11,separatepilotCSV1 audited
control row. No pending treatment metric is fabricated. Best target gaps
remainScan222/289,NR610,SR559hits;no target attainment verified.
Read-only pinned CoT3DRef source findings are in
PVG_READER_ANCHOR_SUPERVISION_REFERENCE.md:explicit anchor supervision
and segmented-object context differ from MCLN predicted queries;
distractor label1 means same-class NON-target,not IoU quality.
No reference code executed or Reader implementation launched.
No local ML tests/imports/diagnostics/forward/evaluation/training.
Preserve unrelated dirty files and the sole desktop handoff policy.

## Previous Override: 2026-11-03 18:52 UTC

Verifiedwait18:50:38UTC:Scan236606live/control5322of6081,
21679MiB/no recenterrors/~11mintrainingleft. SR92354/129032live
18:50:41UTC,6248of6486/~5mintrainingleft,originalconfigunchanged.
FullETA remainsScan19:10-19:25/SR19:05-19:20UTC,updatefromactual.
Scanreceipt:scanrefer-span-gate-progress-20261103T1850.json.
No new complete validation or pilot accuracy result;74/11tables unchanged.

New NR transport evidence:from Scan server through the unchanged wrapper,
18:45:53UTC TCP connection to current NR endpoint timed out(timeout20s),
before SSH banner/no authentication. Original wrapper requery18:46:30UTC
after240s sincepreviousquery stillclosedbeforestatus. This secondvantage
observation does NOT establish rootcause,universaloutage or trainerexit.
No direct-routefallback/SSHoverride/credentialchange/trainingrestart.
Helper:tools/check_current_nr_ssh_banner.sh,execute remotely ONLY.
Receipt:nr3d-endpoint-two-vantage-observation-20261103T1847.json.
NR current liveness remains unknown,lastverified17:59:18UTC/full47.
Scan/SR progress independently;preserveNR240/SR140 and existingpilot
control->span controller. Finishbothpilotconditions before nextReader
implementation. No target attainment verified;goal active.

## Previous Override: 2026-11-03 18:43 UTC

Verifiedwait:actualScanGPUworker236606live18:39:08UTC,
control4509/6081,21679MiB,no recenterrors;~22mintrainingremaining.
SR92354/129032live18:39:12UTC,5711/6486,~16mintrainingremaining,
originalnullinit/max140 unchanged. FullETA Scan19:10-19:25,
SR19:05-19:20UTC;checknearcompletion,notcontinuousrapidpolling.
NR18:42:30UTC requery after180s scheduledwait stillclosedbeforestatus.
LastNRverified17:59:18UTC/full47;no inferredtermination or restart.
New independentpilot auditor tools/audit_scanrefer_span_gate_comparison.sh
is executed ONLY remotely through work/remote_pvground.ps1. Its18:42UTC
pendingbranch verifiedfivepreparedsourcehashes/CUDAoff;bothconditions
not_completed. Completed-log/ownCPUcheckpoint/SHAcomparison branches
await real results;do not claim they passed. Existingcontrollerowns
control->span;no running source/recipe modified.
Receipts:scanrefer-span-gate-progress-20261103T1839.json and
scanrefer-span-gate-independent-audit-20261103T1842.json.
No newfullmetric;74/11tables unchanged,allgoaltargetsstillunmet.

## Previous Override: 2026-11-03 18:32 UTC

SR query recovered18:26:47UTC:original92354/129032live,
trainerelapsed2-01:25:50,GPU22924MiB/70%,5135/6486train.
Null task/PP initialization,innovationsoff,batch12accum2/native1e-4/
backbone1e-3/max140/decay30,40 verified unchanged. Best15=11743/9434,
not a new full evaluation or CPU checkpoint audit. FullETA19:05-19:20UTC.
NR endpoint still closes before status on this turn's query and240s
retry. Last confirmedNR96960live17:59:18UTC;do not infer termination.
No SSH/proxy/trainer/retention change or restart. Recovery without a
configuration change does not prove the transport fault's root cause.
Receipt:mcln-monitor-connectivity-update-20261103T1832.json.
Scan last confirmed18:18:34UTC,236606live/control3048of6081,
fullETA19:10-19:25UTC;no premature repeat Scan query.
No new full metrics,74/11tables unchanged. All original budgets preserved,
goal active;only NR current observability remains degraded.

## Previous Override: 2026-11-03 18:24 UTC

NR/SR observation connections failed twice each,including an180s-spaced
retry,before remote output:"Connection closed by UNKNOWN port 65535".
This is an observability interruption,NOT proof of training termination.
Last confirmed live:NR96960at17:59:18UTC;SR92354at18:02:55UTC.
Reobserve the same registered jobs;no restart/config/retention/SSH changes.
The configured local ProxyCommand alone is not a proven root cause.
No credential or proxy-command value was recorded.
Receipt:mcln-monitor-transport-interruption-20261103T1823.json.
Scancontrol236606verified live18:18:34UTC,3048/6081,
GPU21679MiB,zero recent error records. Full9508pending,ETA19:10-19:25.
Receipt:scanrefer-span-gate-progress-20261103T1818.json.
No new full metrics,74/11tables unchanged. NR47/SR15best/latest17
remain last-audited facts,not fresh NR/SR liveness assertions.
Keep original budgets and pilot controller ownership. Goal remains active.

## Previous Override: 2026-11-03 18:12 UTC

New read-only running-source audit:worker236606cwd verified,five
LF-normalized source identities equal e73dc4a;raw SHA256 retained.
See PVG_READER_BBF_INTERFACE_AUDIT.md and its source receipt.
No local model operations/tests,Reader implementation or pilot change.
Baseline already has language-conditioned candidate spatial attention;
Reader changes a shared query feeding BBF and geometry. Uniform candidate
scalar bias before BBF token softmax cancels and cannot rerank candidates.
Existing unmatched semantic targets are"not mentioned",so a new all-query
positive objective needs explicit assignment/gradient analysis.
These are source/interface constraints,not proof of a new method's benefit.
Pinned officialMCLN/ViL3DRef source verified as text,not executed.
Finish matched Span comparison before the next Reader implementation.
Actual18:02Scancontrol236606live1928/6081/noerrors,SR92354live
4025/6486/originalrecipe. FullETAsScan19:10-19:25,SR19:05-19:20;
NRnextfull~18:55-19:05. No new full results,74/11tables unchanged.
NR47best/latest4114/3248,SRbest15/latest17,ScanB2anchor remain.
No target reached. Check near ETA and preserve all original budgets.

## Previous Override: 2026-11-03 18:01 UTC

NR47 full7899=4114/3248hits,52.082542/41.119129%,best/latest47,
025-first. Versus46:+29/-46hits;no threshold mixing. OwnCPU47/1134/
nonempty optimizer/scheduler/SHA verified,CUDAoff,793092691bytes:
a89e77471e8a5461838f60648651be891baaa658cb79a0f5d446fd54b8f673c3.
Retainer17:55:38UTC kept47only and removed only older preserved46.
Receipt:nr3d-current-full-history-20261103T1759.json,observed17:59:18UTC.
Initial collector hit missing preserved47 in the synchronization window;
after checking actual keeper47/source and waiting180s,the unchanged
collector passed. No fallback,trainer change or duplicate watcher.
Original trainer96960/scratch240budget live,NR target gap610hits.
SRbest15/latest17 and ScanB2/B4 unchanged;no target reached.
Training tables74historical/11restart rows;all middle epochs retained
as metric rows,deleted46path marked authorized removal.
Paired diagnostics/controlupdate code and evidence published46c7814.
Span comparison control real updates verified as below,full results
pendingETA19:10-19:25UTC;SRfull~19:05-19:20,NRnextfull~18:55-19:05.
Recompute ETA from actual speed;do not poll early or restart controllers.
No local model operations/tests.

## Previous Override: 2026-11-03 17:44 UTC

Formal Span comparison controlrun1793726936 actually trains,not just a
screen launch. At17:38:48UTC:220/6081,GPUworker236606,
own epoch1/batch6/size8/step1 finite loss7.0552873611450195;
Reader10/10,Support10/10,Span14/14updated with finite nonzero norms.
Controller236004 retains ownership;do not launch a second copy.
Controlweight0 then freshB2 treatmentweight0.1,one full epoch/full9508
each. No formal pilot validation yet. Control ETA19:10-19:25UTC,
next useful Scan query near19:05;update with actual speed.

New completed-B2 paired postprocessing:no new inference. Of3604strict050
oracle-selection misses,3388are multiple-instance,216unique.
262native-fail/reference-pass/final-fail cases include only9with any
clipped axis;clamp alone does not explain this subset.26cases fail both
endpoints but pass interior fusion. Reader bypass net+17includes+12
changed-query and+5same-query,so it also affects geometry.
These are GT-conditioned diagnostics,not attainable gains/trained ablations.
New receipt scanrefer-b2-decision-subgroups-20261103.json and source in
tools/analyze_pvground_decision_subgroups.py;details PVG_FORWARD_AUDIT.md.
No running source/config change,local model operation or target attainment.
NR/SR remain original scratch240/140;latest auditedNR46/SR17 as below.

## Current Override: 2026-11-03 17:32 UTC

Complete diagnostics/NR46/SR17/73,11tables/Span code published234d38a.
REAL GPU preflight passed17:28:44UTC. Firstbatch auxiliaryloss
0.05352035164833069,total8.344083786010742,48validmatchedaxes,
rawgradientL1=0.24082067608833313,3outsideclampentrieshavegradient.
Atbatch6/size8/step1,total7.0388007164001465finite,Reader10/10,
Support10/10,Span14/14tensorsactuallyupdated;allmodulegradient/delta
normsfinite/nonzero. Not a validation result or accuracy-gain proof.
Preflight updates discarded;no validationmetrics computed.
Controller236004 transitioned17:28:52UTC to freshB2 controlweight0,
run1793726936;actual17:29:33UTC dataloading/noerrors/GPU3MiB.
No control update yet. Samecontroller next runs freshB2weight0.1;
both full one-epoch pilots/full9508eval. No duplicate start.
Preflight evidence includes own config/firstupdate/bounded trainreceipt/
full log/rawgradient receipt. Updated collector reads current exact-run
config/first-update even before completion.
Original NR/SR recipes/budgets unchanged;no targets/no localtests.

## Previous Override: 2026-11-03 17:26 UTC

All-four fixed-B2 diagnostics completed17:21:26UTC,pipelineexit0:
none5483/4656,Reader5504/4673,Support5475/4616,Span5407/4262.
Each full9508,footerverified,orderedidentity/pointSHA/GT exactlypaired.
At050 gained/lost83/66,39/79,114/508;querychanges349/0/0.
These are conditional inference interventions,not trained ablations.
Span has394netbeneficialhits but114breaks;minimal supervision is a test,
not an assumed gain. See PVG_FORWARD_AUDIT.md and archived full logs/summary.

New separate Span pilot controller started17:21:50UTC,runner236004,
screen pvg_scanrefer_span_gate_comparison.
Remote scanrefer_span_gate_comparison_20261103,actual17:25UTC preflight
loading train/text decoupling,noerrors,GPU3MiB,no compute process yet.
No actual parameter-update proof at this snapshot.
SixremoteCPUtests/compilationpassed;5criticalremote/localfileSHAsmatch.
Order:6microbatch treatmentpreflight,controlweight0,Spanweight0.1.
Both formal conditions freshB2model-only/opt/scheduler/seed0/Brecipe,
onefull48655-entryepoch(native6081B8drop_last=48648consumed),full9508eval.
Preflight computes no validationmetrics and its updates are discarded.
No newparameters/attention/soft extent/verifier;rawgateSmoothL1 only,
matchedGT detachedperaxisprojectionproxy,notIoU-optimal. Defaultoff.
Source copied from actual stage_b_source;oldB/diagnostics not overwritten.
See PVG_SPAN_GATE_COMPARISON.md. NR/SR remainoriginal240/140scratch.
Best/latestNR46=4085/3294,SRbest15=11743/9434/latest17=11536/9037.
Tables73/11rows,keptNR46/SR15,17/B2,B4;no newpilotweights yet.
No localtests/modeloperations/targetattainmentclaim.

## Previous Override: 2026-11-03 17:14 UTC

New user direction from the pasted recommendation is accepted as a research
sequence, not as verified claims: finish fixed-B2 diagnostics, then implement
minimal matched-object raw Span gate supervision before any contextual Reader
verifier. Preserve MCLN/Reader/Support/Span, hard Mask extent and clamp; no new
attention or soft extent. NR/SR original scratch training remains unchanged.

NR46 is complete and CPU-audited:4085/3294 over7899 =
51.715407/41.701481%,best/latest46. SHA256
05370b318703e972d1ba18afdf81495f51b99daad8f8db417f77a2764a14dbf8.
Own epoch46/1134 tensors/nonempty optimizer/scheduler verified;trainer96960
live,CUDAoff for audit. Latest full SR17=11536/9037 over17726 =
65.079544/50.981609%;best15=11743/9434,latest17.
SR17 SHA71f9c5fa2d828a54a91f2055d3405762e87e32b7ac8955a1093a4060aa39da00.
SR retention16:39:15UTC keeps15/17 and removed only16.
Full training tables now73 historical rows and11 Scan restart rows.
Best target deficits:Scan222/289,NR639,SR559hits;none reached.

Fixed-B2 none finished full9508=5483/4656, exactly matching archived B2
counts in this repeat. Reader bypass finished full9508=5504/4673.
The summary verifies exact ordered sample identities, point SHA and GT.
Reader bypass changes349 selected queries;025 gained47/lost26/net+21;
050 gained83/lost66/net+17. These are inference interventions on B2,
not separately trained ablations or proof that deleting Reader is beneficial.
At17:08:16UTC Support had9200/9508 records,noerrors;Span is next,
all-four ETA~17:22UTC. No partial Support/Span metric is accepted.

Baseline050 candidate oracle8260,top10coverage6021,selected4656:
3604 samples contain a successful candidate but semanticTop1 misses.
Selected native/reference/final050=4262/4710/4656,not trained variants.
Seven selected native boxes have a nonpositive size axis and exactly these
seven have Support correction mean>1000;all seven fail both thresholds.
The deployed size.clamp_min(1e-6) normalization is consistent with amplification,
not sole-cause proof;seven cases cannot explain289 missing050hits.
Current Span clipping by x/y/z lower158/137/128,upper340/313/272 is per-axis.

Prototype now edited locally, NOT remotely tested/deployed/launched yet:
models/pvground_span_supervision.py plus surgical mcln/losses/main_utils
changes. Per-axis equal-center/size squared-projection proxy, target and
endpoints detached, final Hungarian-matched real GT only, valid references
and nondegenerate directions;SmoothL1 on raw gates,default loss weight0.
It is not an IoU-optimal target. Forward clamp/initialization stays unchanged.
All tests must run remotely;no local tests/model imports/evaluation.
Deploy a separate source snapshot;never overwrite the running diagnostic.
Remote stage_b_source has the completed B training hook and is the intended
parent;its main_utils SHA differs from source and local checkout.

## Previous Override: 2026-11-03 16:33 UTC

The active goal continues. A new isolated fixed-B2 inference diagnostic
is now actually forwarding on the free Scan server; the completed Stage B
trainer was NOT restarted. NR/SR original scratch trainers are unchanged.
New audit directory:
/root/autodl-tmp/mcln_pvground_ft_20261101/offline_audits/b2_fixed_audit_20261103_lrpositive
Screen pvg_scanrefer_b2_fixed_audit, runner232612, evaluator232624.
Started16:30:52UTC; actual16:32:47UTC GPU5407MiB and360 forward plus360
semantic records, no recent errors. Current condition=none. This is NOT
a complete metric. Estimated baseline completion~16:44UTC; four sequential
conditions none/reader/support/span roughly finish~17:25UTC if speed holds.
Use actual logs/progress, and180-300second intervals or near expected finish.

Input is immutable selectedB2, own CPU epoch2/1168 model tensors/nonempty
optimizer/scheduler and SHA46ecfb6a6831fca0704c55fe87fa96c9c92e749824f13173d67d5697da1b92f8
rechecked16:29:50UTC. Batch8/seed0, same source and evaluator, eval-only.
Use positive native2e-6/backbone4e-6/module2e-5 arguments because the
existing optimizer-group builder runs before the eval-only return and
rejects zero decoder LR. No optimizer updates or training are requested.
No model_only_initialization/checkpoint_start_epoch/PP initialization.
The first diagnostic attempt at offline_audits/b2_fixed_audit_20261103
failed before forward with decoder_lr must be finite and positive, exit1.
Its metadata-only JSONL, full log, runner and progress receipt are retained;
do not resume or poll that failed attempt. Corrected invocation uses the
separate lrpositive directory. No model/trainer source was modified.

These are interventions on a trained network, not separately trained
ablations or proof of single-module gain. Require9508 forward and semantic
records plus evaluator-verified completion footer for each condition.
Before comparing, check same ordered identities, input point SHA256 and GT
boxes. Record native/reference/final IoUs, candidate oracle/rank, selected
gates, Reader scale and Support correction/foreground changes. Research
concrete GitHub methods only after measured problems are established.
The preparation receipt records exact deployed file hashes: remote
main_utils.py differs from current local repository, so the recording
commit fcc1e7e does NOT identify every deployed source file.

Latest16:31UTC live checks: SR17 validation1151/1478, finite latest training
loss9.9059/noerrors, ETA~16:36UTC; NR46 training3254/3742, latest loss10.2675
finite/noerrors, full ETA~16:46-16:50UTC. Latest complete audited metrics,
retained sets and71/11row tables remain those in the previous override.
No local tests/model imports/diagnostics/forward/evaluation/training.

## Previous Override: 2026-11-03 15:49 UTC

Scan Stage B run1793674354 finished normally after four full9508 epochs.
Terminal reason=two_consecutive_acc050_decreases;pipeline and transition
exit codes both0. Trainer207977 and the transition screen exited,GPU free.
The final CPU receipt verifies the selected B2 and latest B4 checkpoints.
Do not restart or poll this finished Scan run. NR/SR remain active.

| Dataset / Checkpoint | Examples | Acc025 Hits / Percent | Acc050 Hits / Percent |
| --- | ---: | ---: | ---: |
| NR best44 | 7899 | 4059 / 51.386251% | 3142 / 39.777187% |
| NR latest45 | 7899 | 3979 / 50.373465% | 3156 / 39.954425% |
| SR best15 | 17726 | 11743 / 66.247320% | 9434 / 53.221257% |
| SR latest16 | 17726 | 11534 / 65.068261% | 9328 / 52.623265% |
| Scan best B2 | 9508 | 5483 / 57.667228% | 4656 / 48.969289% |
| Scan latest B4 | 9508 | 5485 / 57.688263% | 4623 / 48.622213% |

All pairs are same-checkpoint last semantic alignment Top1/bbf.
B4 versusB3:+12/-31hits;Acc0504656->4654->4623 gives streak2.
Acc050-first selection retainsB2;do not mixB4's025 withB2's050.
Target deficits using selected bests:Scan222/289,NR665,SR559 hits.
No target is met and no individual-module causal claim is supported.
NR44 versus43:+34/-41hits;latest45 versus44:-80/+14hits.

Remote CPU own-epoch/tensor/nonemptyoptimizer/scheduler/SHA audits passed,
CUDAoff. New NR44/45 SHAs:
fcf49cb3cb64b0054eef3e0e39ed429d352a95a6c11e64e57b76ac78bed41985,
9315aa603eed7bce40ad6670f1d2d6b874971627b113c6252e2775123f1293ae.
B4 SHAe493862b4dbf1d1f97860276fe2a1871decb4367dcc2892279daa8cccfa9c431.
NR/SR weights793092691bytes/1134tensors;B weights795607307bytes/1168.
Current retained setsNR44/45,SR15/16,B2/B4. NR retention14:47UTC
removed43;15:47UTC kept44/45. Scan retention15:07:20UTC audited2/4
and removedonly3;free11485032448->12280643584bytes.
Logs/configs/allmetrics/audits and official/A3 initialization remain.

Main CSV71rows,complete NR1-45/SR1-16;restart CSV11rows,B1-4 complete.
Raw receipts under docs/results/mcln_pvground_20261101:
nr3d-current-full-history-20261103T1547.json,
sr3d-current-full-history-20261103T1540.json,
scanrefer-stage-b-final-full-history-20261103T1544.json.
At15:43UTC SR17 was4878/6486,finite9.9208,noerrorlines;
full ETA~16:35-16:45UTC. NR trainer96960 live,latest explicit epoch45
logged loss10.3036 finite. Original NR/SR null-init configurations,
innovationsoff and240/140 budgets remain unchanged.
No new run,cleanup,local test,model import,forward or evaluation ran.
One Scan SSH attempt closed;successful subsequent terminal CPU collection
resolved it. The scheduled180-second NR follow-up completed.
Prepared fixed-checkpoint bypass diagnostics remain unexecuted.

## Previous Override: 2026-11-03 14:05 UTC

After the scheduled300-second wait,the current-run collector captured
complete SR16:11534/9328 over17726=65.068261%/52.623265%,
same-checkpoint last semantic alignment Top1/bbf.
Versus best15:-209/-106hits. Best15 remains11743/9434,latest16.
Best target deficit559hits;latest16 deficit768. No final target reached.
Remote CPU audits checked actual15/16 own epochs,1134model tensors,
nonempty optimizer,scheduler and SHA256;CUDAoff,trainer92354live.
SR16:793092691bytes,SHA
d0278332a96079086f65024f22b74350da31b1ff871bf2d18fa6ddaa6746824a.
Retention receipt identifieslatest16;current retained SR set15/16.
Main CSV now69rows,including allNR1-43/SR1-16;restart CSV10rows.
Raw receipt:docs/results/mcln_pvground_20261101/sr3d-current-full-history-20261103T1404.json.
NR best/latest43 and Scan bestB2/latestB3 remain as below;B4continues
with strict Acc050-decrease streak1. No training/configuration change,
duplicate cleanup,local tests or model operations were performed.

## Previous Override: 2026-11-03 13:50 UTC

The current-run full-log collector recovered every completed NR1-43,
SR1-15 and Scan Stage B1-3 row. Historical CSV:68 rows; restart CSV:10.
All metrics below are same-checkpoint last semantic alignment Top1/bbf.

| Dataset | Best / Latest | Examples | Acc025 Hits / Percent | Acc050 Hits / Percent |
| --- | --- | ---: | ---: | ---: |
| NR3D | 43 / 43 | 7899 | 4025 / 50.955817% | 3183 / 40.296240% |
| SR3D | 15 / 15 | 17726 | 11743 / 66.247320% | 9434 / 53.221257% |
| Scan B best | 2 | 9508 | 5483 / 57.667228% | 4656 / 48.969289% |
| Scan B latest | 3 | 9508 | 5473 / 57.562053% | 4654 / 48.948254% |

NR43 exceeds previous best41 by18/47 hits and the last published37 by31/31.
SR15 exceeds previous best10 by119/315 and latest14 by182/432.
Target deficits using selected bests are NR699,SR559,Scan222/289 hits.
None of the three targets has been reached. Historical SR14 deficit741
belongs to latest14; the best10 deficit at that time was678.

Remote CPU audits at13:45-13:47UTC checked current best/latest checkpoint
own epochs,1134 NR/SR or1168 Scan tensors,nonempty optimizer,scheduler
and SHA256;CUDA stayed uninitialized. New NR38-42 rows have retained
historical CPU audit evidence even though their weights were removed.
Current SHA256:

- NR43:bd1e4c9ba5ac96faf65d3df7fb748f710950ef031d5a89ac8b3d94277475b7b6
- SR15:386647b8ecac817614090cb23f11b5d7a4d36b7449f67c07f43f84baf23e3969
- B2:46ecfb6a6831fca0704c55fe87fa96c9c92e749824f13173d67d5697da1b92f8
- B3:f23c416f9019abc4765015dd2a0137c03c9c6c98dbc0f55593f2d28e9d3e0430

The existing watchers retain NR43,SR15 and B2/B3. NR retention13:47UTC
removed41/42; SR retention11:14UTC removed10/14; Scan retention12:07UTC
kept2/3. B1 is deleted and its CSV path now says
removed_authorized_best_latest_20261103. Initialization inputs,all logs,
configs,metrics and audit history remain. No additional cleanup was run.

B3's Acc050 fell2 hits againstB2:strict-decrease streak1,
stop_requested=false. B4 is running; only a second strict decline against
B3 would trigger the existing early stop. At13:49UTC trainer207977 was
at3696/6081,latest logged loss8.6958 finite,no error lines. B4 full
validation ETA approximately15:05-15:15UTC,not a measured result.
SR16 was validating739/1478 at13:49UTC,ETA~13:59-14:03UTC.
NR96960 was live; latest explicit logged epoch43 train loss10.4272
was finite at3500/3742. Do not infer the next epoch from a reset tqdm bar.

Original NR/SR scratch configurations and240/140 budgets remain unchanged:
null task/PP checkpoint paths,innovations disabled. Scan B recipe remains
A3 model-only initialization,fresh optimizer/scheduler,frozen text,
native2e-6/backbone4e-6/modules2e-5,batch8/accum6,max10.
No local test/model import/forward/evaluation/training was run.

Raw full-history receipts are under docs/results/mcln_pvground_20261101:
nr3d-current-full-history-20261103T1349.json,
sr3d-current-full-history-20261103T1345.json,
scanrefer-stage-b-full-history-20261103T1345.json.
They include all completed counters,current configs,selection manifests,
best/latest CPU identities and full retention history.
Collector:tools/collect_current_full_history.sh.
The first NR collection met a not-yet-preserved latest43 checkpoint;
the subsequent inventory and successful audit verified save/keeper/retainer
synchronization. No training change or fallback was introduced.

## ScanRefer Stage B Update: 2026-11-03 09:44 UTC

B2 completed full9508 at08:59:14UTC,last semantic alignment Top1/bbf:
5483/4656hits=57.667228%/48.969288%,same checkpoint.
VersusB1:+9/+3hits. Acc050 increased,so the consecutive strict-decrease
streak is0 and training continues. Target5705/4945 remains222/289hits
away; target false.

Remote CPU audit09:41:58UTC matched full-epoch history and complete log,
and verified own epoch2/1168tensors/nonemptyoptimizer/scheduler/three
parameter groups;CUDAoff. Weight795607307bytes,SHA:
46ecfb6a6831fca0704c55fe87fa96c9c92e749824f13173d67d5697da1b92f8.
Best manifest selects2. Retention09:03:02UTC audited2/SHA/resume states,
removed onlyB1,kept2best/latest. Free12284051456->13079662592bytes.
RestartCSV9rows;raw audit and retention receipt preserved.
ActualB207977 live09:43UTC epoch3=1555/6081;first-update finite loss
6.407628059387207,three modules updated. No complete B3 validation yet.
Run recipe/config unchanged.

## SR3D Update: 2026-11-03 09:40 UTC

SR14 completed08:16:33UTC,full17726,last semantic alignment Top1/bbf:
11561/9002hits=65.220580%/50.784159%,same checkpoint. Acc025 is
63hits below best10=11624/9119 (65.575990%/51.444206%); best10 remains.
Latest14 is741hits from12302; best10 was678hits away at this observation.
Epoch14 was the latest resumable checkpoint.

Own remoteCPU14/1134model tensors/nonemptyoptimizer/scheduler verified,
CUDAoff. Weight793092691bytes,SHA256:
1c8759ef21a5ee0de5cb04fc817b81df98aad3d9d814b379300be997ba8424be.
Retention08:18:13UTC audited best10/latest14,including optimizer/scheduler
and both SHAs,then removed only old13. Free bytes
19739271168->20532367360. Full-historytable61rows;CSV marks13 deleted.
Raw receipts:sr3d-latest-complete-epoch-14.json,
sr3d-checkpoint-identity-epoch-14.json,
sr3d-best-latest-retention-epoch-14.json.

Actual SR92354 live09:38UTC,epoch15=3393/6486,GPU22924MiB,
latest logged training loss10.1074at3000,no errors/config changes.
NR96960 live09:38UTC,epoch39=117/3742,loss10.6725at3500;
Scan B207977 live09:37UTC,epoch3=1363/6081,loss7.4498at1300.
Original NR/SR scratch configurations/budgets unchanged;Scan B recipe
unchanged. No new full NR/Scan validation. No local tests/model changes.

## NR3D Update: 2026-11-03 07:42 UTC

NR37 completed07:30:27UTC,full7899,last semantic alignment Top1/bbf:
3994/3152hits=50.563362%/39.903785%. Both metrics belong to the
same checkpoint. Versus36:+41/+51hits. New025-firstbest37;
the target4724hits remains730hits away. No target is achieved.

Own remoteCPUcheckpoint epoch37/1134tensors/optimizer/scheduler verified,
CUDAoff. Weight793092691bytes,SHA256:
8cd1a07bb9ea191a3a243864bdc71a0a6c28f2665f970d777f25e61d0494bb3b.
Bestmanifest37 independentlyverified07:34:15UTC with the sameSHA.
Retention07:34:08UTC independentlyaudited37/nonemptyoptimizer/scheduler/
SHA,removedonlyolder36,andkept37best/latest. Freebytes increased
21180428288->21973524480. Historicalmetrics/logs/configs/audits remain;
the CSV marks36'spathdeleted. Historicaltable60rows,Scanrestart8rows.
Raw receipts:mcln-nr3d-epoch37.json,nr3d-checkpoint-identity-epoch37.json,
nr3d-best-latest-retention-epoch37.json in the current results directory.

ActualNR96960live07:41:26UTC,epoch38=721/3742,GPU22833MiB/97%,
finite latestloggedtrainloss10.4481at500,noerrors/configchanges.
Originalnulltask/ppinitialization,batch12accum4,max240/decay150 unchanged.
SR92354live07:33:23UTC,epoch14=5535/6486,finite10.1909at5500.
ScanB207977live07:33:20UTC,epoch2=3338/6081,finite7.4426at3300.
Theirconfigurationsremainunchanged;no newfullSR/Bvalidation.
NextfullETA:SR14~08:10-08:25UTC,ScanB2~09:00-09:15UTC,
NR38~08:30-08:40UTC. Estimatesonly. No localtests or modelchanges.

## NR3D Update: 2026-11-03 06:30 UTC

NR36 full7899 completed06:27:31UTC: last semantic alignment Top1/bbf
3953/3101hits=50.044309%/39.258134%.
Versus previousbest34: +62/-45hits; versus35:+67/+51.
The025-first criterion selects36,although050 is lower than34.
Do not combine36's025with34's050. Target4724 remains771hits away.
At06:30:09UTC keepermanifest still showed34; wait for its scheduled
refresh before treating best-alias synchronization/cleanup as complete.
Follow-up06:34:17UTC independently confirmed manifest36 and the sameSHA.
Retention06:34:00UTC audited own36/nonemptyoptimizer/scheduler/SHA,
removedonly34/35,kept36best/latest. Free20387758080->21973950464bytes.
Raw retention receipt accompanies the result; deleted CSV paths marked.
OwnCPUepoch36/1134model tensors/optimizer/scheduler verified,CUDAoff.
SHA13d3373c28f10271872982a105cbb7099be6f7b3a9c0acb1c143ab67a31220ba,
793092691bytes/3links at06:29:36UTC. Trainer96960/keeper135075/
retention156800 live. Null task/pp initialization and original
batch12accum4/LR1e-4/backbone1e-3/max240/decay150 unchanged.
Historical table59rows; Scan restart8rows separate. No localtests.

ActualScanB207977live06:30:04UTC,epoch2=1018/6081,70finite logged
traininglossrecords,latest7.5115at1000,noerrors/configchanges.
SR92354live06:15:03UTC,epoch14=2335/6486,finite10.2005at2000,
originalnullinit/budgetunchanged. Targetsremainunachieved.
RawNR36receipt and fileidentity accompany the full-epoch table.

## ScanRefer Update: 2026-11-03 06:06 UTC

Scan B1 full9508 completed06:01:50UTC,lastsemanticTop1/bbf
5474/4653hits=57.572570%/48.937737%.
Compared with matched A3=5461/4566:+13/+87hits.
Compared with matched official=5435/4327:+39/+326hits.
Same-checkpoint targets5705/4945 remain231/292hits away.
No individual-module causal or target-completion claim.

OwnCPUepoch1/1168tensors/nonemptyoptimizer/scheduler/threeLRgroups
verified06:04:19UTC,independent complete log/evaluator counts agree.
795607307bytes,SHA cda46832c35f95fb70710da23768ea28eb27f8130bb6ae787de2665f0cf92f48.
Best/latestB1;retention06:02:53 independentlyauditedSHA/resumestates,
removednone. A3 remainsinitializationinput. RestartCSV8rows,
historicalNR/SR/oldScanCSV58rows unchanged.

B2actual207977live06:04:56UTC104/6081,finitefirstupdate7.124665260314941,
all34module tensors updated. Keep original B recipe,max10 and two
consecutive050-decreases rule;B1 setsbaseline/streak0. No localtests.
NR36actual96960live06:04:49UTC2772/3742,GPU22833MiB;
SR14actual92354live06:05:28UTC1931/6486,GPU22924MiB,
recentloggedtrainlossesfinite/noerrors/originalnullinit/configunchanged.
NextfullETA NR36~06:25-06:35UTC,SR14~08:10-08:25UTC,
ScanB2~09:00-09:15UTC;estimatesonly.
See StageArestartdoc/rawB1receipt for exactmarkersandcollectionscope.

## Current Results: 2026-11-03 05:24 UTC

Two new complete validations were audited on remote CPU; each checkpoint
has its own expected epoch,1134 model tensors,optimizer and scheduler.
No CUDA was initialized in the audit processes.

| Dataset | Latest Epoch | Full Samples | Hits025 / Hits050 | Acc025 / Acc050 | Best Epoch |
| --- | ---: | ---: | --- | --- | ---: |
| NR3D | 35 | 7899 | 3886 / 3050 | 49.196101% / 38.612483% | 34 |
| SR3D | 13 | 17726 | 11253 / 8812 | 63.483019% / 49.712287% | 10 |

All metrics are same-checkpoint last semantic alignment Top1/bbf.
NR35 completed05:21:01UTC,down5/96 hits versus34; best34 remains
3891/3146 (49.259400%/39.827826%),833hits below the025target.
SR13 completed05:19:41UTC,down268/311 versus12 and371/307 versusbest10.
Best10 remains11624/9119 (65.575990%/51.444206%),678hits below025target.
Do not combine SR10's025withSR12'shigher050.

NR35 SHA:d708c5d37ba45121b678aa57b5488b876a77ac034c6c6ccdb7d40f4228ea15aa.
SR13 SHA:50b45aec894a908d63a90e8542fded5e6499eea58d79d87821e2bf98e8aa73ac.
Each793092691bytes. SRbest10 was rehashed05:22:16UTC; its original
null task/pp initialization,batch12accum2/max140/decay30,40 remain unchanged.
At05:21:51UTC SRretention auditedbest10/latest13 and removedonly12;
freebytes19739373568->20532469760. Historical metric rows remain.
The full-epoch table now has58rows; Scan restart's7rows remain separate.
NRretention05:21:52UTC independently auditedbest34/latest35 with
nonemptyoptimizer/scheduler/SHA; removednone,free21181136896bytes.
NR/SR continue their original240/140budgets without configuration changes.
Single-epoch decreases do not identify a causal defect or authorize
changing their requested reproduction runs.

Scan B1 was live05:11:05UTC at4705/6081,47finite logged trainlosses,
latest7.3849,noerrorlines/configchange; still no complete B validation.
Its first full result remains estimated around05:50-06:05UTC.
At05:25:47UTC actualB207977/retention209444 remained live,B1=5224/6081,
GPU21989MiB,currentmodel-onlyA3/freshoptimizer/scheduler recipe unchanged.
TrainingETA~05:50UTC; completevalidation remains pending.
No target is verified reached,no local tests ran.

2026-11-03 04:49UTC: complete Stage A checkpoint preservation audited
on remote CPU. Selected epoch3 versus official54:1134 native serialized
tensors exactly unchanged,including204 BatchNorm buffers;34 new tensors
all belong to the three module prefixes. Both knownSHAs verified.
This extends the first-update freeze checks to the actual final A3 state,
not a Stage B freeze/individual-module causal claim. Raw receipt and
reproduction helper are included. CUDAoff,no model import/forward/localtests.
Current B207977live04:49:01UTC,3927/6081,finite logged trainloss7.3798
at3900,no errorlines/configchange. Full Bvalidation still pending.

## Current Override: 2026-11-03 04:23 UTC

NR3D epoch34 completed validation on all7899 examples: last semantic
alignment Top1 Acc@0.25=3891/7899 (49.259400%), Acc@0.50=3146/7899
(39.827826%). Relative to prior best epoch32, this is +29/+111 hits.
At04:26:45UTC the semantic-best manifest independently confirmed34.
The target Acc@0.25 4724/7899 remains833 hits away.
Remote CPU audit confirmed checkpoint own epoch34,1134 model tensors,
optimizer and scheduler present; CUDA was not initialized. SHA256
b6964cd073ae582b88eabc57de28173dbe99b64a42b688583684d7ff5099ce20,
793092691 bytes. Best and latest resumable are both34. Retention checks
the completed checkpoint and best manifest before cleanup; all logs,
configurations and metric rows are retained.
At04:25:43UTC retention independently audited34,removed32/33 and kept
only34;free bytes increased20388360192->21974556672.

NR original scratch configuration remains unchanged: batch12/accum4,
LR1e-4/backbone1e-3,max240,checkpoint_path and pp_checkpoint null.
SR3D and ScanRefer Stage B were live at their latest checks; no new
complete validation there. Do not claim any target met. No local tests.

Raw receipts: outputs/mcln-nr3d-epoch34.json and
outputs/nr3d-checkpoint-identity-epoch34.json. Full-epoch CSV now has
56 historical/current-run rows.

2026-11-03 03:40UTC: ScanBactual207977advanced1315->1477/6081
between03:35:22and03:39:49UTC with240s scheduled recheck.
All14logged cumulative traininglossrecordsfinite,latest7.4088at1400,
norecenterrorlines,GPU21989MiB. NR/SRactual96960/92354live03:37UTC,
59/54finite loggedtrainingrecords,originalconfigs/budgets unchanged.
NewcompacthealthhelperpairsTrainheader+lossand excludesEval;fixed
an observedmonitor-labelmixup,not a trainingdefect. See StageArestartdoc
andthree training-loss-health rawreceipts. No newcompletevalidation,
architecture/configchange,modelimport/forward/localtest.

2026-11-03 03:29UTC: remoteCPU audit covers all1,201 training superpoint
files,exact ScanNet train metadata match,50,000 integer labels each,
zero single-cluster/negative-label scenes,min90/median843/max4363.
No overlap with141 ScanRefer validation annotation scenes; previous312val
audit was directory-wide. Logged48655examples=36665ScanRefer+1199*10
joint detection;1199 inferred from recipe/loggedlength,not reconstructed
object filtering. This explains a count difference,not a training defect.
Trainer207977/retention209444 live before/after;CUDAoff,no localtests.
Historicalmesh/geometry/metriccauses unproven. See PVG_CURRENT_DATA_AUDIT.md
and scanrefer-current-all-train-superpoint-statistics.json.
NRretention03:21:36 receipt auditsbest32/latest33ownCPUepochs/optimizer/
scheduler/SHA,removednone. No newcompletevalidation sinceNR33.

2026-11-03 03:20UTC: NR33 completed03:17:37UTC full7899,
3856/3027hits=48.816306%/38.321306%, down6/8versus32.
Own CPU epoch33/1134tensors/optimizer/scheduler verified,CUDAoff.
Best remains32=3862/3035,latest33; best025target deficit862hits.
At03:19:49UTC latest weight793092691bytes/3links,SHA256
964c70a97f3b8895e05c200c25bbd4b7a8f041eae9ad19bcaced2bd4af63f967.
Actual trainer96960/keeper135075/retention156800 live,free21181689856bytes.
Raw mcln-nr3d-epoch33.json/nr3d-checkpoint-identity-epoch33.json;
historicalCSV55rows. Original recipes/budgets unchanged,no localtests.
Compact Scan monitor correction published1d7426a; Bfullvalidation pending.

2026-11-03 03:06 UTC: Scan B run1793674354/PID207977 remains live.
Own first update passed: epoch1/batch6/size8/step1, finite loss
7.021632194519043, all34 Reader/Support/Span tensors updated with
finite nonzero gradients. At03:01UTC training reached66/6081 batches.
No complete B validation yet; estimated first full result around05:50UTC.
A1/A2 cleanup completed03:02UTC, freeing1,202,454,528bytes; A3 remains
warmup best/latest and B input. Scan retention PID209444 live03:03UTC,
waiting for complete B validation. NR/SR best/latest retention also live;
initial deletions freed19,034,316,800/7,930,966,016bytes respectively.
All epoch rows/logs/configs retained; no local tests or NR/SR recipe changes.
Detailed desktop handoff is maintained only in
C:/Users/gb/Desktop/document/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md.

2026-11-03 02:55 UTC: user authorized best+latest-only weight retention.
NR deleted24 older preserved weights, kept best/latest32; free21.97GB.
SR deleted10, kept best10/latest12; free20.53GB. Both trainers live;
240s retention watchers deployed. All metrics/logs/audits kept; deleted CSV
paths marked unavailable. See storage policy and exact deletion receipts.

Scan matched full9508 passed: official5435/4327, selectedA3=5461/4566,
57.435844/48.022718%; selected-minus-official+26/+239, official-minus-E0
0/0. Point/order/seed/backend/SHA checks passed. A3 original validation
5461/4565 remains separate; reevaluation differs by one050hit.
StageB run1793674354/trainer207977 was loadingdata02:53UTC;
own first real update still needs verification. No local tests.

02:35UTC SR12 completed02:29:59UTC full17726=11521/9123hits,
64.994923/51.466772%,+97/+286versus11. Bestremains10=11624/9119;
do not pair10's025with12's050. OwnCPUepoch12/1134/optimizer/scheduler
verifiedaftercheckpointsave,CUDAoff. Preserved793092691bytes/3links,
SHA d9b4e0f496ca72b0d330e9f44e36f89d50b2f8545cafc75732f6f2c39c32d234.
Actual92354/129032live02:34UTC,free12601630720bytes77%used.
Raw mcln-sr3d-epoch12.json;historicalCSV54rows.
NR/SRoriginal240/140epochrecipesunchanged;no localtests.

02:29UTC ScanA complete02:23:49UTC/exit0;controller remoteCPUaudited
allthreefull9508checkpoints. A3selected5461/4565=57.435844/48.012200%,
owncheckpoint3/1168tensors/optimizer/scheduler verified,CUDAoff.
SHA f4e98cce830eb474a94402df00dbd160c379b963e7d285c759b86c4a60b0a898.
Targetdeficits244/380. ActualmatchedevaluationPID205803liveunder204119;
pairpending/Bnotstarted. RestartCSV5rows;seeStageArestartdoc/rawreceipt.

2026-11-03 02:19UTC: NR32 completed02:15:21UTC,full7899 last-semantic
Top1=3862/3035hits,48.892265%/38.422585%. New025-firstbest32,
versus31 +46/+24hits;025targetdeficit862. OwnCPUcheckpointepoch32,
1134tensors/optimizer/scheduler verified,CUDAoff;hardlink
runs/nr3d/preserved_checkpoints/ckpt_epoch_32.pth,793092691bytes.
SHA d160096bd6f795feb1cf3d5512d5039bc8d492ab28e7b56298b29de83e212af3.
Bestmanifest32verified02:19:29UTC;originalconfigSHA unchanged
f837245468d816600a3910223e05c802a689ace776c76038f24e7a3926283072.
Raw mcln-nr3d-epoch32.json/nr3d-semantic-best-identity-epoch32.json;
all-epochCSV53rows. No localtests orNR/SRrecipechanges.
At02:18UTC ScanA3 validation628/1189,SR12 validation531/1478,
actualtrainerslive. No completeA3/SR12metricsyet. ExistingScantransition
controller owns matchedvalidation/conditionalB;no duplicate launch.

2026-11-03 01:42UTC:ScanRefer guardedtransitiondeployed,
screen pvg_scanrefer_restart_transition,actualbashPID204119 verifiedlive
01:39:51UTC besideAtrainer189152. It checks every240seconds,then requires
Aexit0/allthreeownauditedepochs/originalgate,performs existing matched
fullvalidation,and only on success/emptyGPU runs existingBrecipe.
Failures exit,preservinglogs,no retries or gatechanges.
Do NOT manually start matchedvalidation/Bwhilethiscontrollerislive.
All phases use the transition screen;inspectactualBprocess/ownupdate
receipt rather than expecting an independentBscreen. No Btrainingverifiedyet.
See docs/PVG_SCANREFER_STAGE_A_RESTART.md andstartupreceipt
scanrefer-guarded-transition-started.json. Existingtwohourmonitorprompt
updated;rrule/notificationpolicy/targetchatunchanged. NR/SRunchanged.

2026-11-03 01:31:59UTC:current-restart diagnosticimportpreflight passed.
Actual deployedentry imported withoutmain;trainer/evaluator/IoU/converter
resolved insidecurrentrestartsource,audithelpers insideobservations_v2.
CUDAoff,no modelconstruction/checkpointload/forward/evaluation/localtests.
Raw scanrefer-restart-audit-import-preflight.json;docs/PVG_FORWARD_AUDIT.md
updated. A3/matchedvalidation/freeGPU stillrequired beforediagnosticforward.
At01:30UTC NR32=1014/3742,ScanA3=4284/6081,SR12=4581/6486,
currentNR/SRtrainer/keeperprocesseslive/configsunchanged. No newfullresult.

2026-11-03 01:27UTC data evidence:currentScanloader uses
data/superpoints/{train,val};noautomaticswitchtomeshdirectory.
All312valfiles fingerprinted and remoteCPUdata-onlyaudited:
50,000labels/file,superpoints min157/median1001/max3894,
zero single-superpoint/negative-labelscenes. CUDAoff,no modelwork or
localtests. ManifestSHA
f989fc5c7aeb25707ad75ae26e6db0dae400163e0ea9b34fc0f50113fdb5acc2.
ValannotationSHA matches canonicalhandoffhistoricalidentity.
Historicalmesh equivalence/train-splitquality remainunverified;
not evidence of a metric-gapcause. See docs/PVG_CURRENT_DATA_AUDIT.md
and raw scanrefer-current-data-fingerprint.json/
scanrefer-current-superpoint-statistics.json. CurrentScanPIDs live01:25UTC.

2026-11-03 01:20UTC update: latest scope preserves original NR240/SR140
scratch runs unchanged; ScanRefer restart proceeds through three Stage A
epochs and matched validation before conditional Stage B. No local model work.

NR31 full7899 completed01:15:45UTC:3816/3011hits,
48.309913%/38.118749%. Versus30 +3/-44;newAcc025-firstbest31,
target025deficit908. OwnCPUcheckpoint epoch31/1134tensors/optimizer/
scheduler audited. Preserved793092691bytes,fourlinks01:19UTC.
SHA256 e9d220e4b7a0917e59702d3c5021ab66da288e494a9ca0e1d5d873edf3be41cb.
Bestmanifest31verified;originalconfigSHAunchanged:
f837245468d816600a3910223e05c802a689ace776c76038f24e7a3926283072.
Actual96960/135075live;all-epochs.csv now52rows.
Raw mcln-nr3d-epoch31.json and nr3d-semantic-best-identity-epoch31.json.

ScanA3=3814/6081at01:19UTC,finite recorded real updates/all3modules
updated/frozenstateunchanged. Best completeA2=5460/4563over9508.
SR12=4070/6486,actual92354/129032live,originalconfigunchanged;
latestSR11=11424/8837,bestSR10=11624/9119over17726.
ScanA3fullETA02:20-02:40UTC,SR12fullETA02:30-02:45UTC.
No StageBlaunch before A3selection and matched-protocol validation.

NRfree3734265856bytes,94%used at01:19:19UTC:atmost4additional
793092691-byteweights ignoringgrowth,not exactfailureETA.
Cleanupapprovalpending,no deletion. Olderkeep29/30proposalstale;
recompute currentbest/latest and resolvedpaths after approval.

Storage2026-11-03 00:25:30UTC:NRfree4527525888bytes92%used,atmost5new
793092691-byte weights ignoringlogs/transientwrites. Actual96960/135075
live. Best/latest30 sharesinode15040694278 withlast/semanticbest/
preservedaliases,4links. Current23distinct preservedepochs remain.
Raw nr3d-storage-20261103T002530Z.json. Pendingpolicy proposalcurrently
keep29/30,allmetrics/logs/audits retained;recomputeafterapproval.
No cleanupauthorization/deletion. NR240budgetstillunchanged.

00:21UTC ScanA2 existing-log subgroup audit:unique025/050=1240/1075
over1419 (87.385483%/75.757576%);multiple=4220/3488over8089
(52.169613%/43.120287%). Sums exactly5460/4563. ComparedA1:unique
+8/+5,multiple+6/+15hits. Multiple remainingmisses3869/4601 vsunique
179/344. Aggregate allocation,not paired fixes or module causality.
Raw scanrefer-stage-a-epoch2-group-evidence.json;restartdoc updated.
At00:19UTC actualScan189152/188533/188537 live,13GiBfree;
SR92354/129032live,SR12=1437/6486,configunchanged. No new fullepoch.

2026-11-03 00:16UTC:NR30 completed00:14:12UTC,full7899 semanticTop1:
3813/3055hits=48.271933%/38.675782%,new025-firstbest30.
Versus29 +45/+94;versuspreviousbest27 +38/+147hits.
OwnCPUcheckpoint30/1134tensors/optimizer/scheduler verified,CUDAoff,
hardlinked;bestalias30verified00:16:11UTC,793092691bytes,SHA256
c994e74ebc48ec43e037f700df07e19d2cc7b0fda8364d000ce72107e3f74f61.
ConfigSHA unchangedf837245468d816600a3910223e05c802a689ace776c76038f24e7a3926283072,
nulltaskinit/noinnovations/batch12accum4/max240/decay150.
Target025deficit911hits,not achieved. Trainer96960/keeper135075live
00:15UTC,NR31=59/3742. HistoricalCSV51rows(oldScan10/NR30/SR11).
ScanA3ownfirsttwo updates finite5.8771486282/6.6740288734,all3modules
nonzero gradients/deltas,frozenparameters/buffersunchanged;A3=231/6081
23:57UTC. Finish3epochs/matchedvalidation beforeconditionalB.
No localtests,oldruninspection,trainingchanges orcleanup.

2026-11-02 23:54UTC update:
Scan restart A2 completed23:51:44UTC,full9508 last-semantic Top1:
5460/4563hits=57.425326%/47.991165%,versusA1 +14/+20hits.
OwnCPUcheckpoint2/1168tensors/optimizer/scheduler verified,CUDAoff,
hardlinked601225876bytes/4links,SHA256
e22d6b7a4968fbafb6e4fcf97342ce88497c300e3fb8183b2da35c003b8f39e9.
Semanticbest2 verified;target deficits245/382hits,not achieved.
Gate pending completeepochs[1,2];A3 trainer189152/keeper188533/
preserver188537 live23:53UTC. EstimateA3full02:20-02:40UTC Nov3,
then matchedofficial/selected evaluation before conditionalStageB.
OriginalE0 delta+25/+236 crosses documentedbackend boundary,not
module attribution. NewScanCSV4rows:officialE0/identityE0/A1/A2.

SR11 completed23:46:21UTC,full17726 semanticTop1:
11424/8837hits=64.447704%/49.853323%,versus10 -200/-282hits.
OwnCPUcheckpoint11/1134tensors/optimizer/scheduler verified,CUDAoff,
hardlinked793092691bytes/3links,SHA256
a9e29f0782b6e61f6ce09a3d13760a9760ee0edf8335a08fcef3c93289ff5721.
Best remains10=11624/9119,65.575990%/51.444206%,target025deficit678.
Actual92354/129032live23:48UTC,original140epochconfig unchanged.
HistoricalCSV50rows(oldScan10/NR29/SR11),newScan separate.
SRfree13394792448bytes76%used,atmost16weights ignoringgrowth.
NRlastfree5320933376bytes91%used23:15UTC;cleanupstillunapproved.
No localtests,old-runinspection,architecture changes orBlaunch.

NR29 completed2026-11-02 23:13:35UTC,full7899 last-semantic Top1:
3768/2961hits=47.702241%/37.485758%. OwnCPUcheckpoint29/1134tensors/
optimizer/scheduler verified,CUDAoff;hardlink retained. SHA256
2ef30f51358a2817acce17e3c56297f765e5f95913b046b730eb962ae743947d,
793092691bytes/3links. Versus28 +36/+128hits;versus27 -7/+53.
Acc025-first best remains27=3775/2908,47.790860%/36.814787%.
Do not combine29's050 with27's025. OriginalNR240epoch training continues,
actual96960/135075 live23:15UTC. Historicaltable49rows(oldScan10/NR29/SR10).
ScanA2=5016/6081 and SR11=5953/6486 at23:15UTC,stilltraining;
nextfullSR about23:40-23:50UTC,ScanA2 about23:50-00:05UTC.
NRfree5320933376bytes(91%used),at most6additional same-sizeweights
ignoringlogs/transientgrowth. Cleanupapproval stillpending,no deletion.
Raw nr3d-epoch29.json and nr3d-epoch29-file-identity.json.
No localtests,old-run inspection,architecture/config changes orBlaunch.

Monitoring correction,2026-11-02 22:29UTC:local
work/check_current_mcln_keeper_status.sh previously enumerated stopped
oldScan1793579788 alongside NR/SR. Removed the oldScan entry and restricted
screen/PID selection to current NR/SR before reuse. Repo
docs/PVG_FORWARD_AUDIT.md now retires its stopped oldScan epoch8 command.
Future bypass diagnostics require the current restart's selected own
checkpoint/epoch/SHA,completed three-epoch/matched-validation gates,free
remote GPU and import resolution against the restart source. Historical
old-source import preflight does not verify restart imports. No old-run
inspection,local tests,model execution or training changes were made.

NR28 completed2026-11-02 22:12:54UTC,full7899 last-semantic Top1:
3732/2833hits=47.246487%/35.865299%,versus27 -43/-75hits.
OwnCPUcheckpoint28/1134tensors/optimizer/scheduler verified,CUDAoff,
hardlink runs/nr3d/preserved_checkpoints/ckpt_epoch_28.pth retained.
SHA2563dcb6345d716073b1947036a6a9189a353ad0761e772c588fc6743b5ae16a101,
793092691bytes/3links at22:16:33UTC. Raw nr3d-epoch28.json and
nr3d-epoch28-file-identity.json in docs/results/mcln_pvground_20261101/.
Best remains27:3775/2908,47.790860%/36.814787%,target025deficit949.
At22:17:06UTC actual96960/135075 live,NR29=277/3742,configunchanged:
nulltaskinitialization/noinnovations/batch12accum4/240epochs/decay150.
Bestalias27 verified793092691bytes/2links;historical4linksnapshot changed
after native retention,but weight survives. NextNRfull~23:10-23:20UTC.
Historicaltable48rows(oldScan10/NR28/SR10);newScan separate.
At22:07UTC ScanA2=2090/6081,SR11=2983/6486,actual GPUtrainingcontinues.
NextScanA2~23:50-00:05UTC,SR11~23:45-23:55UTC;noBlaunch.
NRdiskfree6114398208bytes(89%used),at most7additional same-sizeweights
ignoringlogs/transientgrowth. Cleanupunapproved/unexecuted;do not reuse
old keep-lists. No localtests or currenttrainingchanges.

21:34UTC file-only identity supplement:
NR27 SHA2564bc94f69e4b39ce97304fef924173c2f61152a86ee53740fb656a196c8749e0b,
SR10 SHA25693dccfce1086f0c146524df308d88f75d487cff70ee5d57e300fb4338b14ee41.
Both793092691bytes. Own epochs/1134tensors and optimizer/scheduler are
established by the separate existing full-result CPU audits,not this hash.
Configs stillnull taskcheckpoints/noinnovations;no recipe/budget change.
Raw docs/results/mcln_pvground_20261101/nr3d-semantic-best-identity-epoch27.json
and sr3d-semantic-best-identity-epoch10.json include config hashes and
selection manifests. No localtests/modelruns orcleanup.

21:26UTC A1 semantic log-only subgroup audit:unique1232/1070over1419,
multiple4214/3473over8089,sum5446/4543. Multiplemisses3875/4616,
uniquemisses187/349. OriginalE0 subgroup differences are unmatched
aggregate differences,not paired fixes or established module attribution.
Details/rawloglines:docs/PVG_SCANREFER_STAGE_A_RESTART.md and
docs/results/mcln_pvground_20261101/scanrefer-stage-a-epoch1-group-evidence.json.
ActualScan189152/188533/188537,NR96960/135075,SR92354/129032verifiedlive.
NRfree6907711488bytes(88%used),SR14190972928(74%used);at most8/17
new same-sizeweights fit ignoringlogs/transientgrowth. Historicalcleanup
proposal requires freshselection and explicitapproval;no cleanup done.
No localtests/modelruns/configchanges.

Scan restart A1 completed21:19:22UTC2026-11-02, full9508 last-semantic
Top1 hits5446/4543=57.278082%/47.780816%. Owncheckpoint1/1168tensors/
optimizer/scheduler CPUverified and hardlinked; semantic_best selects1.
Checkpoint601225876bytes/4links,SHA256
9fc7999242e766ec44ea569042eb750ca35662947713c8177bef51badf28e519.
Original E0 delta+11/+216 hits crosses the documented protocol boundary;
not established module attribution. Matchedvalidation remains required.
A1 meets declared5425/4328metricgate, but decision remains pending until
all3Aepochs and successfulpipelineexit. Finaltargetdeficits259/402hits.
At21:20:34UTC A2=42/6081,actual two A2updates finite6.6050129/8.2219801,
all3modules nonzero gradients/deltas, frozen parameters/buffers unchanged.
Immutablestartupreceipt retains A1proof;per-epochupdatefiles nowA2.
NewScanCSV E0official/E0identity/A1 separate from historicaloldScan.
A2ETA23:50-00:05UTC,A3~02:25-02:45UTCNov3,subject to throughput.
NoBlaunch or configurationchange. NRbestalias27 verified21:16:37UTC,
793092691bytes/4links,NR28=271/3742;nextfull~22:10-22:20UTC.
NR27resultpush09ed9bf. No localtests or storagecleanup.
Raw docs/results/mcln_pvground_20261101/scanrefer-stage-a-epoch1-identity.json
and scanrefer-stage-a-complete-epochs-through1.json.

NR27 completed 2026-11-02 21:12:16 UTC, full7899 last-semantic Top1:
3775/2908 hits = 47.790860%/36.814787%. Own CPU checkpoint epoch27,
1134 tensors, optimizer/scheduler verified, CUDA uninitialized; hardlink
runs/nr3d/preserved_checkpoints/ckpt_epoch_27.pth. New observed025-first
best27; versus26 +90/+95 hits, versus25 +48/-37. Target025 deficit949.
Never combine27's025 with25's050. Historical table47rows: oldScan10,
NR27,SR10. Original240epoch recipe unchanged; trainer96960/keeper135075
live at21:12:17 and next epoch starting. Best alias refresh not yet read.
ScanStageA collector21:12 returns no complete rows, pending decision;
no StageB launch. No localtests or cleanup. Raw
docs/results/mcln_pvground_20261101/nr3d-epoch27.json.

21:05 UTC verification: SR semantic_best.json now selects epoch 10 with
11624/9119 hits over 17726 examples; the 793092691-byte best weight has
4 hardlinks. This supersedes the pending alias note below. Original
SR11 is at 301/6486, NR27 at 3680/3742, and Scan restart A1 at 5954/6081.
Scan has no complete new Stage A validation yet. Its first two actual
updates have finite loss, nonzero three-module updates and unchanged
frozen parameters/buffers. SR10 results and raw evidence were pushed
in commit 8427315. No local tests or training changes; cleanup unapproved.

SR10 completed2026-11-02 20:58:39UTC:full17726 last-semantic Top1,
hits11624/9119=65.575990%/51.444206%,new selected025-firstbest10.
Owncheckpoint10/1134tensors/optimizer/scheduler remotelyCPUverified and
hardlinked. Versus9:+522/+487hits;versuspreviousbest8:+500/+533.
Target025deficit678hits;not achieved. Keep original140epochs unchanged.
Historicaltable46rows:oldScan10/NR26/SR10;newScan separate.
Raw docs/results/mcln_pvground_20261101/sr3d-epoch10.json.
At21:01:31UTC actualSR92354/keeper129032 live,SR11=117/6486,
GPU22924MiB. Bestalias still8 on that read,pending keeper refresh;
numeric10 independently preserved. ScanA1=5769/6081,NR27=3394/3742.
Nextfull estimates NR27~21:10-21:15UTC,ScanA1~21:15-21:25UTC,
SR11~23:45-23:55UTC. No localtests,trainingchanges or approvedcleanup.

20:34UTC real storage risk:NRfree7700983808bytes(86%used),SR14984093696.
Each preserved full checkpoint793092691bytes;only9/18additional same-size
weights fit ignoring transient/log growth. Keeping every epoch cannot
support NR240/SR140 on50GiB disks. Read-only exact inventory and proposal:
docs/MCLN_STORAGE_RETENTION_PROPOSAL.md. Keep best+latesttwo and allmetrics
would currently exclude NR17oldweights/SR7oldweights,free13.48/5.55GB
on separate disks. No weights deleted or bulk archived;no recurringcleanup
deployed. Destructive cleanup awaits explicit human approval per supplied
AGENTS.md. Continue live training unchanged and monitor free space.

20:26UTC prepared Stage B now enforces successful full matched validation
of its selected epoch/SHA before importing the GPU trainer. Prior entry
could bypass that documented prerequisite. Eleven remote CPU tests passed;
entry SHA dcb44691123301a9806ae700e51d038896647d6ca33cbda95b65c1a9fc3ce2a6.
See later stage_b_matched_gate_prepared.json,which supersedes the entry
hash in historical stage_b_prepared.json. Live Stage A unchanged.
No matched GPU evaluation or Stage B training has occurred. No local tests.

NR26 completed2026-11-02 20:11:19UTC:7899 last-semantic Top1 examples,
hits3685/2813=46.651475%/35.612103%. Own epoch26/1134tensors/optimizer/
scheduler remotely CPU verified and checkpoint hardlinked. Down42/132hits
versus25;selected025-firstbest remains25,3727/2945=47.183188/37.283200%.
Target025deficit997hits for that best checkpoint. Continue original240
epochs unchanged;no early stop or method failure inferred from one epoch.
Historical current-run table45rows:oldScan10/NR26/SR9;newScan separate.
Raw docs/results/mcln_pvground_20261101/nr3d-epoch26.json.
At20:13:24UTC original trainer96960/keeper135075 live,NR27=129/3742,
GPU22833MiB,next full ETA~21:10-21:15UTC. Current-root-only progress helper:
work/check_nr3d_current_mcln_progress.sh. Do not use stale
work/inspect_nr3d_current_epoch.sh,which points to a stopped20261030 run.

New ScanStageA still awaits complete metrics. StageB source/entry is
prepared separately (5remoteCPUtests) with correct all3module2e-5 rates
and semantic two-decrease stop. Matched official/selected initialization
entry is prepared in protocol_audit (6remoteCPUloadingtests);when StageA
is complete and GPU free,run the matched check before attributing tiny
gains and before launching StageB. No matchedGPUevaluation or StageB
training occurred. OriginalE0/currentgate/live source unchanged.
Details,raw test/preparation evidence and secure helper commands:
docs/PVG_SCANREFER_STAGE_A_RESTART.md. No local tests.

19:16UTC keeper refresh confirmed:NRsemantic_best.json/pth nowepoch25,
3727/2945hits,1134tensors,optimizer/scheduler,793092691bytes/4links.
Previous pendingalias24 note is superseded. Raw manifest:
docs/results/mcln_pvground_20261101/nr3d-semantic-best-epoch25.json.
Original NRmain96960/keeper135075 live,NR26=415/3742,config unchanged.
ScanStageA1=1347/6081,GPU11369MiB;SRmain92354/keeper129032 live,
SR10=2758/6486,GPU22924MiB,nullcheckpoints. No new completeepoch,
44rows unchanged. Next full validation estimatesNR26~20:10-20:15UTC,
SR10~21:00-21:10,Scan1~21:15-21:25. PollnearETA,no localtests.

NR25 complete19:10:21UTC,full7899 semantic Top1 hits3727/2945=
47.183188%/37.283200%,new observed025-firstbest25. Owncheckpoint25,
1134tensors/optimizer/scheduler remoteCPUverified and hardlinked.
44 historicalrows:oldScan10/NR25/SR9. NR target4724hits025,deficit997.
At19:13:36UTC original main96960/keeper135075 alive,NR26=216/3742,
config unchanged;semantic_best alias still24 on that read,pending240sec
keeper cycle. Do not claim25alias before confirming it.
ScanStageA still pending first full epoch;SRlatest9/best8 unchanged.
New tools/collect_scanrefer_stage_a_epochs.py deployed separately,
real initial emptyrows and2remoteCPUfixtures passed,CUDAuninitialized;
receipt docs/results/mcln_pvground_20261101/scanrefer-stage-a-collector-remote-tests.json.
Collector90251d6 and final acceptance policy6181634 pushed;no localtests.

19:00UTC live-process verification: Scan main189152/keeper188533/
preservation188537 alive;keeper waiting_for_complete_validation.
Scan18:58=604/6081,logged cumulative batch600loss6.9913 finite.
NR original main96960/keeper135075 live,epoch25=3308/3742;
SR original main92354/keeper129032 live,epoch10=1991/6486.
NR/SR configs unchanged;no new complete metrics or best selection.
Next full validation estimates:NR25~19:10-19:15UTC,Scan1~21:10UTC,
SR10~21:00UTC. Monitor near ETA with180-300sec intervals.
All targets remain unachieved;no method change justified by partial logs.

VERIFIED18:48UTC: Scan StageA run1793644724 has two actual optimizer
updates at epoch1 batch6/12,finite losses6.477235794067383/7.649016857147217.
All Reader/Support/Span gradients and deltas finite/nonzero;frozen native
parameters and buffers exactly unchanged twice. Full9508 E0 passed,
official+identity5435/4327hits=57.162390%/45.509045%,same input/orderSHA.
Startup immutable evidence:
docs/results/mcln_pvground_20261101/scanrefer-stage-a-startup-receipt.json.
Progress129/6081,GPU10125MiB;first full trained validation ETA21:10UTC.
No complete StageA epoch yet. Source/config recipe and conditionalStageB
in docs/PVG_SCANREFER_STAGE_A_RESTART.md. Native/BNfrozen,batch8/accum6,
moduleLR1e-4,max3epochs. Per-epoch update witnesses may later be overwritten.
NR25/SR10 continue original trainers/source/config/RNG unchanged,verified
18:41:47UTC. NRbest24/SRbest8;historical43rows;new E0 table separate.
No local Windows/WSL tests/model execution. This supersedes pending notes.

Latest user request overrides historical status below: restart ScanRefer
using the supplied plan, while NR/SR continue original full scratch runs.
Details and remote launch/evidence:docs/PVG_SCANREFER_STAGE_A_RESTART.md.
OldScan1793579788 stopped18:06:56UTC November2;best8/latest10/log preserved.
New isolated scanrefer_restart_20261102 pipeline second attempt18:13:26UTC:
matched full9508 official-versus-identity E0 then conditional module-only
StageA,batch8/accum6/moduleLR1e-4,3epochs,native+BNfrozen.
Do not claim real StageA training until own first-update receipt passes.
First E0 attempt failed before eval on training-only loader restriction;
independent E0 initializer now checks1134 native tensors/checkpointepoch54.
Production training loader restriction unchanged. No localtests.

NR24 complete18:10:00UTC7899hits3666/2805=46.410938%/35.510824%;
own checkpoint24/1134 tensors/optimizer/scheduler audited and preserved.
New NR025-firstbest24. SR9 complete18:12:48UTC17726hits11102/8632=
62.631163%/48.696830%;own9/1134 audited,preserved;bestSRremains8.
Combined historical current-run table43rows:oldScan10/NR24/SR9.
Keep newScanE0/StageA rows separate. All targets remain unverified/unmet.
NR/SR trainer/config/source/RNG unchanged;never loadScanweightsinto them.
Two-hour automation now points at restart pipeline;notification preference preserved.

Update17:40--17:44UTC: all three current trainers and metric keepers
verified live17:40UTC,GPU23671/22833/22924MiB;configs unchanged.
Scan e11=957/6081,NR e24=2115/3742,SR e9=5889/6486.
No new complete validation:41 rows,best8/23/8 unchanged.
SR about14min train plus validation;NR about23min plus validation;
next useful complete checks remain18:00--18:10UTC and about18:10.
No extra GPU evaluation,no local tests,no active source/config changes.

New target-hit decomposition from existing Scan1--10 subgroup receipts:
selectede8 unique1234/1058 of1419,multiple4188/3447 of8089.
Targets require5705/4945 total hits,current deficits283/440.
Even perfect unique leaves at least98/79 extra multiple hits necessary.
With unique unchanged,multiple needs4471/3887 hits=55.272592/48.052911%.
Full subgroup table and caveats:docs/PVG_SCANREFER_TRAINING_OBSERVATIONS.md.
Do not mix epoch2's higher multiple050 with8's unique050. This establishes
a necessary subgroup improvement,not a module failure or gain.
Prioritize matched oracle/ranking/mask/Span evidence when GPU available.

Update17:31UTC: Scan e10 full9508 semantic Top1 completed17:26:28UTC,
5366/4461 hits=56.436685%/46.918385%. Remote CPU own epoch10/1168
tensors/optimizer/scheduler verified; hardlink preserved. Best remains8.
Full41 rows Scan10/NR23/SR8; selected8/23/8,all targets unmet.
Log precision fix and raw training observations pushed57e552b.
No active trainer/source/config change,no local tests.
SR trainer92354/keeper129032 verified live17:30:13UTC,e9=5421/6486,
GPU22924MiB/41%,task checkpoints null. About24min train plus validation
remain; next useful full check18:00--18:10UTC. NR24 full check about18:10.
Two-hour ACTIVE monitor updated with41-row evidence,new diagnostic staging,
explicit offline local_rank0,remote-only testing and current limitations.

Historical17:25UTC: Scan trainer154585/keeper178546 verified live17:24:45UTC,
e10 validation992/1189,about2min remaining; no completed e10 claimed yet.
Remote log-only extraction preserves cumulative training means at the
same6000/6081 batch point for epochs1--10. Loss7.3920->7.2505,
box1.2628->1.2302,semantic35.6224->35.1608,SP Dice0.1733->0.1669.
Not exact full-epoch means,not proof of generalization/module benefit.
Raw JSON and interpretation: docs/PVG_SCANREFER_TRAINING_OBSERVATIONS.md.
Static trace confirms corrected masks and mixed boxes reach final criterion;
no disconnected-supervision bug established. Real forward diagnostics remain
missing. Native evaluator None also cannot simply be fixed by calling
export_retrain_metrics: that existing exporter requires source-choice
counters and uses position metrics,not the required last/bbf semantic pair.
The independent keeper remains the current mitigation.

Found actual log precision problem: epoch LR formatting .5f makes1e-6
look like0.00000. Repository format changed to .8g; log-only collector
accepts scientific notation. Display only,no live source/schedule change.
No local tests or additional GPU evaluation.

Update2026-11-02 approximately17:12UTC: NR e23 full7899 semantic Top1
validation completed17:09:23UTC,3623/2643 hits=45.866565%/33.459932%.
CPU checkpoint own epoch23,1134 tensors,optimizer/scheduler verified and
hardlinked. New Acc025-first best e23; do not reuse e22's higher050.
Selected Scan8/NR23/SR8; full table40 rows:9/23/8. Targets still unmet.
At17:07UTC all three current trainers and240s keepers verified live,
GPU23671/22833/22924MiB; configs unchanged,NR/SR still no innovations.
Scan e10 about9min training remaining plus full validation; SR e9 about47min
training remaining plus validation. No spare GPU diagnostic launched.

New saved-state audit: Scan e9 optimizer54729 steps=6081*9;
NR e22=20592=936*22; SR e8=25944=3243*8. Scheduler steps agree.
Measured decay milestones correspond to11/31,151,31/41 completed epochs,
respectively. Pinned official scheduler also subtracts warmup_epoch;
default-1 produces this offset. It is inherited,not a demonstrated new
training bug. NR live config explicitly confirms warmup_epoch=-1/start1.
No active scheduler changed. Raw remote receipts and exact interpretation
are in docs/PVG_OPTIMIZER_RECIPE_AUDIT.md. Empty third NR/SR optimizer
group has zero parameters; its1e-5 LR does not affect any task parameter.
Real trained-model forward distributions and matched9508 bypass results
remain unmeasured; no mechanism change or gain justified by this audit.

Update approximately16:56UTC: fixed a concrete offline invocation error.
Current main_utils parser accepts only --local_rank(default1),and ignores
the launcher's --local-rank=0 via parse_known_args. The documented eval
command omitted the explicit underscore option,so it would select device1
on these single-GPU hosts. Added `--local_rank 0` (see exact
command in docs/PVG_FORWARD_AUDIT.md),matching the existing training fix.
Remote parser-only audit confirmed old=1/corrected=0,CUDA uninitialized;
no model/device selection or training source change. Reproducer:
tools/audit_pvground_eval_rank.sh. Raw receipt is stored with current results.

Update16:51UTC: NR current trainer96960 and keeper135075 verified live;
e23 training2873/3742,GPU22833MiB,task checkpoints still null.
Latest complete e22/selectede21 unchanged. Approximately13min training
remaining plus prior validation duration; next useful full check near17:10UTC.

Inspected two additional primary GitHub implementations for the observed
ScanRefer unique/multiple gap,without executing reference code:
ViL3DRef82a95d9e41ecd9920d678823eda1b366947da204,conditional spatial
attention; ConcreteNet7ab4628f19f0442fefaff45550a32e2f30b82189,
radius-masked candidate fusion and scene-restricted contrastive denominator.
Detailed source paths,mechanisms,differences and conditional experiment
rules are in docs/PVG_MULTIPLE_INSTANCE_REFERENCE_NOTES.md.
Our decoder queries already contain context and MCLN has contrastive
supervision; neither reference proves an absent capability or a model cause.
No reference module/loss has been copied or deployed. First obtain real
chosen/oracle/Support/Span/bypass evidence before mechanism changes.

Update approximately16:43UTC: identified an instrumentation gap, not a
proven model failure: prior offline hooks observed Span but omitted Reader
activation size and Support's actual member-mask corrections.
Added Reader query/residual L2, Support per-query mean/max absolute delta
and fused-foreground flips on observed member slots only. Ragged scene
statistics remain separate. Selected-query evidence joins these to existing
semantic rank, IoU and Span gates. Values are explicitly before_bypass;
separate flags identify whether the corresponding result is applied.
No model architecture, weights, training config or running source changed.

Nine focused fixture tests passed on the remote Scan server,CUDA
uninitialized,Torch2.1.2+cu118,pytest9.2.0. Covered all four bypass settings,
unobserved-slot exclusion/ragged scenes,selected-query association and
tracked evaluator/IoU agreement. This is not a real model/dataset forward.
Training env lacked pytest; test-only dependencies installed into the
isolated observations_v2/test_dependencies directory. No local tests run.
New helpers staged in offline_audits/observations_v2; active trainer/evaluator
imports and new helper paths verified remotely. ArchiveSHA
bf7aba8910d14a57c76634fbf805aa9dbd4e00a29ae752e55e8a3d9cd0c4c568.
Raw remote test/import receipts saved under docs/results/mcln_pvground_20261101.
See docs/PVG_FORWARD_AUDIT.md for the updated remote command and limits.
Real gate/activation/correction distributions and full matched bypass
metrics remain missing; do not choose a mechanism change from fixtures.

Update16:31UTC: all three current trainers and independent metric keepers
verified live through the secure wrapper; configurations unchanged.
Scan e10 training3075/6081, GPU23671MiB; NR e23 training1551/3742,
GPU22833MiB; SR e9 training2870/6486, GPU22924MiB.
No new complete validation: latest complete Scan9/NR22/SR8,
selected best Scan8/NR21/SR8. Estimated next full validations:
NR approximately17:10UTC, Scan17:20-17:30UTC, SR approximately18:00UTC.
These are estimates from observed training rates plus previous validation
durations, not completion promises. No concurrent GPU diagnostic launched.

User execution boundary: no tests on the local machine, including Windows
or local WSL CPU/GPU diagnostics, model forward, evaluation or training.
Do not continue local-probe environment preparation or download model/data
for local testing. The prior isolated conda clone completed; this does not
mean a model test ran. Its proposed model probe is cancelled. Future needed
tests and diagnostics run remotely only. Historical CPU test counts below
describe past work and do not authorize rerunning them locally.

The desktop brief MCLN-PV-Ground-training-handoff.md has been merged into
the user's sole desktop handoff:
`C:/Users/gb/Desktop/document/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md`.
Keep the source as history, do not update it again. The canonical file's
Consolidated Current Handoff section distinguishes historical attempts and
snapshots from the current run. This repository file remains technical
documentation published to GitHub, not a second desktop handoff.
The ACTIVE two-hour monitor now preserves both constraints and the current
39-row/latest-best state, with its existing notification policy unchanged.

Update16:12UTC: NR e22 full7899 semantic validation completed16:08:11UTC
(server11/03 00:08:11). Hits3543/2754=44.853779%/34.865173%.
CPU own epoch22,1134 tensors,optimizer/scheduler verified and hardlinked.
Acc025 falls14 hits versus e21 while050 rises112 hits; best stayse21.
Do not combine e21Acc025 with e22Acc050. Full table39 rows:
Scan1-9,NR1-22,SR1-8. BestScan8/NR21/SR8,targets still unmet.

Update16:10UTC: semantic diagnostics pushedc8437a3;15 CPU tests passed.
Helpers staged on Scan server under`offline_audits/c8437a3/tools`,archiveSHA
34a234ad96bbaea6b74a48550d0aeb868154d019118943961c921472f7cbca67.
Remote imports of current trainer,evaluator and IoU functions passed,
CUDA remained uninitialized. Active training source untouched. This is
import preflight only,not model/checkpoint/GPU-forward validation.

Update15:47UTC: ScanRefer e9 completed15:45:32UTC, full9508
last semantic Top1 hits5390/4473=56.689104%/47.044594%.
CPU own epoch9/1168 tensors/optimizer+scheduler verified, fixed hardlink
preserved. Both thresholds fell32 hits versus e8; selected best stays e8.
Full table now38 rows: Scan1-9,NR1-21,SR1-8. Targets remain unmet.
Offline audit hooks/eval entry,CPU evidence and11 passing tests pushed
commit31373aa. No real-sample bypass result or method change yet.

Update15:31UTC: SR e8 full17726 validation completed15:24:27UTC,
11124/8586 semantic Top1 hits=62.755275%/48.437324%. CPU own epoch8,
1134 tensors,optimizer/scheduler confirmed and hardlinks preserved.
All three current trainers and metric keepers verified live at15:29UTC;
NR/SR task checkpoints null, configurations unchanged. Scan e9 training,
NR e22 training, SR e9 training. Keeper successfully advanced SR best.
Publication commitbc70601 pushed; full table now37 rows. Targets unmet.

Update15:23UTC: independent metric keepers are deployed and verified live
on all3 hosts. Screen names are pvg_DATASET_metric_keeper; interval240s.
They CPU-verify a newly selected best and atomically replace
`RUN_DIR/semantic_best.pth` (hard link) and `semantic_best.json`.
Existing retained bests: Scan e8, NR e21, SR e7. Actual Python processes,
first successful loop logs and link counts were checked, not just screens.
Training code/processes and two-hour user notification cadence are unchanged.
The trainer still returns None for ordinary evaluation; this is an external
mitigation, not an in-process evaluator repair. Keepers must remain live.
They exit on a missing best weight or inconsistent metrics; inspect their
logs at each monitor run. They cannot recover historical deleted weights
or guarantee survival through host/process failure.
Tests now7 passing:3 module tests plus4 keeper tests covering trainer
unlink, best replacement, unchanged best, Scan semantic/mask separation,
partial validation and missing-best rejection.

Updated: 2026-11-02, approximately 15:07 UTC / 10:07 America/New_York.
This section supersedes historical September and clean-baseline instructions.
Do not restart stopped clean-baseline or previous failed fine-tunes.

## Acceptance Targets

Require one complete checkpoint to satisfy both ScanRefer thresholds:
Acc025 >= 60.0% and Acc050 >= 52.0%, on all 9508 validation examples.
NR3D requires Acc025 >= 59.8% on 7899 examples.
SR3D requires Acc025 >= 69.4% on 17726 examples.
Use last semantic alignment Top-1; never substitute position or mask metrics.
Select ScanRefer by Acc050 then Acc025, NR/SR by Acc025 then Acc050.
The goal remains active and none of these targets has been demonstrated.

## Verified Current Results

| Dataset | Epoch | Hits025 / Hits050 | Acc025 / Acc050 (%) | Gap To Target (pp) |
| --- | --- | --- | --- | --- |
| ScanRefer | 8 | 5422 / 4505 | 57.025663 / 47.381153 | 2.974337 / 4.618847 |
| NR3D | 21 | 3557 / 2642 | 45.031017 / 33.447272 | 14.768983 at 025 |
| SR3D | 8 | 11124 / 8586 | 62.755275 / 48.437324 | 6.644725 at 025 |

Each listed checkpoint was loaded on CPU and its own epoch checked.
ScanRefer has 1168 model tensors; NR/SR have 1134. All contain optimizer
and scheduler state. Hard links were preserved under
`runs/DATASET/preserved_checkpoints/ckpt_epoch_N.pth`.
NR e21 improves Acc025 over e20 but its Acc050 is lower: report the e21
pair together. Do not replace its 33.447272% with e20's 33.523231%.
All 39 completed epoch rows recovered so far are in the accompanying CSV:
ScanRefer 1-9, NR3D 1-22, SR3D 1-8.
Rows without weights are explicitly marked `missing_retention_deleted`.

Log timestamps are server UTC+08:00. Subtract eight hours for UTC.
Earlier chronological notes contain incorrect conversions for some epochs.
For example, NR e15 completed at server16:33:10, which is08:33:10UTC,
and SR e5 at server14:58:52, which is06:58:52UTC.
The raw evidence JSONL retains server timestamps to prevent ambiguity.

## Runs And Method Attribution

Remote root: `/root/autodl-tmp/mcln_pvground_ft_20261101`.
Only use `work/remote_pvground.ps1`; credentials must never enter documents.

| Dataset | Endpoint | Screen | Current Training Recipe |
| --- | --- | --- | --- |
| ScanRefer | connect.bjb2.seetacloud.com:48073 | pvg_scanrefer_mcln_finetune | run1793579788, batch8, LR1e-5, backbone2e-5, module1e-5, 100epochs, decay10/30 |
| NR3D | connect.nmb1.seetacloud.com:32749 | pvg_nr3d_mcln_scratch | run1793553113, batch12, accumulation4, LR1e-4, backbone1e-3, 240epochs, decay150 |
| SR3D | connect.nmb2.seetacloud.com:45914 | pvg_sr3d_mcln_scratch | run1793552465, batch12, accumulation2, LR1e-4, backbone1e-3, 140epochs, decay30/40 |

All three screens and GPU processes were verified live this turn.
NR e21 completed validation while observing; Scan e9 and SR e8 continue.
NR/SR checkpoint_path and pp_checkpoint remain null. Frozen local RoBERTa
is pretrained; do not describe the text encoder as random initialization.
NR/SR are currently original MCLN scratch runs, without the additional
Reader/Support/Span tensors. Their metrics are not evidence that those
three modules improve NR/SR. Preserve these runs as references; do not
silently relabel them as the proposed three-module method.

ScanRefer loads only the official epoch54 model initialization, with a
fresh optimizer/scheduler and Reader/Support/Span enabled. Checkpoint SHA256:
`a9930065996fce1d0dd5ee9fe00a120bdb3a2c88d158b7a3666717d842ac113d`.
Its epoch9 batch1 witness shows finite loss5.888957 and finite nonzero
gradient/parameter changes in all three modules. Latest logged average
loss was7.2812. This proves optimization runs, not improvement over the
official initialization, which has no matched epoch0 evaluation here.

## Problems, Evidence, And Next Experiments

1. Checkpoint retention is not metric-aware in these runs. The ordinary
   evaluator logs metrics but does not emit the expected JSON receipt;
   retention records/best are empty. Two-hour polling exceeds the roughly
   one-hour NR epoch duration, so several checkpoints disappeared before
   inspection. New collector CPU-audits every surviving numbered checkpoint
   and preserves hard links. It cannot recover deleted weights.
   Mitigation deployed: CPU metric keepers poll every240sec and preserve
   the audited selected best through atomic hardlink replacement. SR e8
   advancement was observed. Native evaluator remains unfixed; inspect
   keeper process/log health during two-hour monitoring.
2. Old status checks truncate logs to a short tail. This hid complete
   ScanRefer epochs3-8 and some NR rows. The new collector scans the entire
   current log and reconstructs integer hits using semantic subgroup counts.
   Scan uses unique1419 + multiple8089. It accepts only the FIRST occurrence
   of subgroup values per epoch, because later unique50/multi50 are masks.
   Reconstructed rates must agree with the rounded overall semantic rate.
3. ScanRefer Acc050 oscillates46.29-47.38% over epochs1-8; e8 exceeds e2 by
   only9 hits (0.094657pp). This is not evidence of a robust method gain.
   Epoch8 unique/multiple semantic025 are86.962650%/51.774014%;
   semantic050 are74.559549%/42.613426%. Multiple-object disambiguation is
   a measured weakness, but subgroup difficulty alone does not prove cause.
   Next: evaluate the SAME retained e8 checkpoint with module bypasses,
   matched preprocessing and per-example predictions; quantify ranking,
   box localization and mask support separately before modifying modules.
4. Span currently uses a hard0.5 mask reference with detached masks and
   clamps raw mixing weights to[0,1]. These are code facts, not a diagnosed
   failure. Collect reference-valid fraction, gate boundary occupancy and
   per-axis errors; test whether a learned correction actually changes
   relevant boxes. Only change its confidence/extent mechanism when those
   measurements identify a failure.

   CPU weight audit at15:38UTC of the retained Scan e8 checkpoint,
   SHA2563572193fef0f32d1d6ca22b41cbbb02932a30de0a51b18deb3b0c40ee37bcf38:
   all34 innovation tensors finite; all six zero-initialized output
   weight/bias tensors have every element nonzero. Reader output weight
   norm0.531269,bias0.026472; Support0.208826/0.005804;
   Span0.183130/0.010051. Optimizer state steps all48648.
   This excludes unchanged zero outputs, but does not establish forward
   activation magnitude, clamp occupancy, gradient health at e8, causal
   contribution or a method gain. Raw evidence:
   `docs/results/mcln_pvground_20261101/scanrefer-e8-module-weight-audit.json`.
   Use sample-based hooks in an isolated evaluation process for raw/clamped
   gates and reference validity; do not launch an additional GPU evaluation
   on the occupied4090 or modify the active trainer to obtain those values.

   Offline instrumentation implemented:
   `tools/eval_pvground_audit.py` accepts normal single-GPU MCLN eval flags
   plus `--audit-jsonl PATH --audit-bypass none|reader|support|span`.
   Run through torch.distributed.launch with a separate port, `--eval`,
   `--use_pvground_innovations`, complete trained checkpoint_path and matched
   preprocessing; omit model_only_initialization and initialization-only
   pp_checkpoint. Pin a preserved epoch path, never a moving best alias.
   `tools/pvground_forward_audit.py` hooks only the offline model and writes
   per-sample candidate gate boundary counts, means, valid reference counts,
   and absolute axis shifts. Support bypass keeps measured geometry but
   returns original member logits; Span bypass restores exact native boxes;
   Reader bypass zeros its residual. No checkpoint keys or weights change.
   Metadata includes checkpoint SHA and invocation. Extended helper
   `tools/pvground_semantic_audit.py` exports selected-query gates, identity/
   pointSHA, semantic chosen/oracle IoU, first-good rank,top10, and selected
   native/reference/final signed center/size errors against root GT.
   Completion requires exactly9508 records and audit hits/counts matching
   evaluator last/bbf025/050. CPU tests run the tracked semantic evaluator
   method and IoU functions in isolation; three seeds match selected boxes,
   IoUs and counters. Altered counter and strict-threshold checks pass.
   Total15 focused tests pass. Full command/limitations:
   `docs/PVG_FORWARD_AUDIT.md`. Remote import preflight passed from isolated
   staging without CUDA initialization; GPU entry point has not run. No real-sample
   bypass result,gate distribution,GT error result or method gain is claimed.

The three intended module responsibilities remain Reader (local visual
evidence), Support (candidate mask membership correction), and Span (axis
extent correction). No novelty claim is established by their names.
Require a matched baseline and module ablations before claiming benefits.

## Open-Source Reference Checked

Official MCLN repository: https://github.com/qzp2018/MCLN
MaskDINO: https://github.com/IDEA-Research/MaskDINO
Inspected MaskDINO commit3831d8514a3728535ace8d4ecc7d28044c42dd14,
`maskdino/modeling/transformer_decoder/maskdino_decoder.py`, via GitHub API.
It exposes mask-to-box initialization and separates proposal initialization
from subsequent decoder refinement. This motivates a diagnostic comparison
of our hard mask extent reference with native box geometry. It does not
justify copying a 2D module unchanged into 3D or claiming a measured gain.
The next experiment must retain Reader/Support/Span and test one diagnosed
mechanism at a time, with source attribution and full validation evidence.

Additional primary-source inspection,2026-11-02 approximately16:05UTC:
Mask3D,https://github.com/JonasSchult/Mask3D,
commit11bd5ff94477ff7194e9a7c52e9fae54d73ac3b5,
`trainer/trainer.py`,function`get_mask_and_scores`,retrieved via GitHub API.
It forms a binary foreground from positive mask logits,averages sigmoid
mask probabilities over that foreground,and combines that quality estimate
with per-query class confidence for ranking. This is instance segmentation,
not language grounding; class confidence is not our decomposed semantic score.
No implementation copied or deployed. It motivates measuring agreement
between semantic candidate ranks,member support and actual box IoUs AFTER
the fixed-checkpoint export. It does not justify adding a scoring module,
claiming a causal weakness,or a gain. If oracle coverage is high but semantic
selection fails,inspect chosen-vs-good candidate support confidence before
considering an internal Support calibration; if oracle coverage is low,
reranking alone cannot recover the missing boxes. These are experiment
decision rules,not observed outcomes.

## Tuning Policy And Validation

When a dataset reaches target, freeze its accepted configuration. If another
is approximately1pp away, first investigate dataset-specific module widths,
depths or internal parameters without changing the architecture/module set.
Within approximately0.3pp, prefer LR/batch/seed trials before a method change.
No current deficit is within0.3pp. No hyperparameters changed this turn.

Local targeted verification: `python -m pytest tests/test_pvground_innovations.py -q`
passed3 tests. Initial collection failed because the sparse checkout omitted
the tracked reader source; materializing that existing file fixed collection.
This is focused CPU coverage, not a full GPU training regression test.
Active ScanRefer's first-update witnesses supply independent runtime evidence.

User's canonical desktop document:
`C:/Users/gb/Desktop/document/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md`.
Keep its historical sections; the current dated section takes precedence.
Publish code and this document to https://github.com/666666666666gao/MCLN .
## Current Override: 2026-11-04 04:25 UTC

Current-only checks04:25UTC: Reader run1793763075=3541/6081 training
batches,finite first-update witness remains verified;roughly34min of
training plus full validation remains. NR run1793553113 epoch58=1941/3742;
SR run1793552465 epoch22=3585/6486. All current trainers/keepers/
retainers live with finite recent losses and unchanged scratch configs.
No new complete semantic validation since NR57/SR21; current best/latest
metrics remain those recorded in the previous overrides.

## Current Override: 2026-11-04 04:10 UTC

Live checks04:10UTC: Reader competitor run1793763075=2415/6081 augmented
batches;finite first update remains verified,worker254229 active,~50min
roughly remain before full validation. NR run1793553113 epoch58=841/3742;
SR run1793552465 epoch22=2873/6486. All trainers/keepers/retainers live,
recent losses finite,no errors,original scratch configs unchanged.
No new full validation since NR57 and SR21. Reader experiment code/docs
and NR57 receipt are pushed through commits55424c0,9ea748a,fd274c2.

## Current Override: 2026-11-04 04:03 UTC

NR3D run1793553113 epoch57 full7899 validation=4090/3305 hits,
51.778706%/41.840739%. Best49=4123/3325; latest57 is -33/-20 and634
Acc025 hits below target. CPU audit passed best49/latest57 own epochs,
1134 tensors,nonempty optimizer/scheduler,SHA and preserved paths.
Epoch57 SHA da6ab7a97bb4cd63d23479e50701251852526cf27695c2bb9d48a378ba3555ff.
The first collector overlapped checkpoint promotion and asserted;retry
after epoch58 began passed,with no training/config changes. Receipt:
`docs/results/mcln_pvground_20261101/nr3d-current-full-history-20261104T0401.json`.

Reader competitor run1793763075 remains active;at03:57 it was1487/6081
with verified finite first update. NR58=334/3742 at04:03;SR22=2289/6486
at03:57. Original scratch recipes unchanged. Full Reader validation and
checkpoint audit pending.

## Current Override: 2026-11-04 03:43 UTC

Current-only remote check03:42:35UTC: Scan Reader run1793763075 remains
live at371/6081 augmented training batches,finite first update verified
at batch6;GPU worker254229 uses22052MiB. At the observed1.2batches/s,
rough full-validation ETA is around05:00UTC,not a completion claim.
Current source/recipe unchanged since launch. No validation metric yet.
NR run1793553113 epoch57 advanced to3115/3742,trainer96960 live,latest
losses finite;epoch57 full evaluation is approaching. SR run1793552465
epoch22 is1588/6486,trainer92354 live,latest losses finite. Both retain
their original null-init scratch configurations and budgets.

## Current Override: 2026-11-04 03:39 UTC

Authorized new ScanRefer Reader experiment is running in independent
run1793763075,screen `pvg_scanrefer_reader_competitor`; it does not resume,
overwrite or inspect old/failed runs. B2 epoch2 model-only initialization
SHA46ecfb6a6831fca0704c55fe87fa96c9c92e749824f13173d67d5697da1b92f8,
fresh optimizer/scheduler,frozen text encoder,seed0,batch8/accum6,
native2e-6/backbone4e-6/module2e-5,one full48655-entry epoch and
full9508 validation.

Reader adds a zero-initialized learned residual contrasting each candidate
local summary with the mean summary of its four nearest predicted candidate
centers. This uses predicted boxes and Reader features only,not GT anchors.
Support/Span roles are unchanged. Remote CPU identity/scale-gradient test
and six-batch GPU preflight passed. At03:38:47UTC the first full-run
optimizer update had finite loss7.05528736114502 and actual finite nonzero
updates Reader11/11,Support10/10,Span14/14 tensors. Full validation is
pending; no accuracy benefit is claimed.

Reader selection motivation: immutable B2 has 5483/4656 over9508,with
6021 strict-IoU0.50 top-ten candidate coverage and 4656 semantic Top1 hits.
This is not recoverable accuracy. Reader bypass changes349 queries and
nets17 strict-IoU0.50 hits,so the new experiment tests candidate-relative
local evidence rather than duplicating the existing language-conditioned
spatial attention. Full method, evidence limits and result gates are in
`docs/PVG_READER_COMPETITOR_EXPERIMENT.md`.

At03:13UTC NR trainer96960 was live at1198/3742 in epoch57; SR trainer92354
was live at229/6486 in epoch22. Both retain original scratch recipes.
Receipt for the Scan trial will be added after complete validation/audit.

## Current Override: 2026-11-04 03:10 UTC

