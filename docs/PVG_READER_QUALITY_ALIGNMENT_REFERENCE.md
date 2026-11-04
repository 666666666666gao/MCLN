# Reader Quality Alignment Source Review

## Measured Question

The completed Reader competitor trial scores5471/4608 over9508,below
matched trained control5477/4620 and immutable B25483/4656. Aggregate
scores do not distinguish failed box coverage from candidate ranking.
The full paired diagnosis registered in PVG_READER_COMPETITOR_EXPERIMENT.md
must finish before selecting another trained change.

The current Reader source selects four neighbors solely by predicted
center distance. It does not establish object identity or exclude duplicate
predictions of one object. That is a code property and an untested
hypothesis,not proof of why the trial lost accuracy. Existing audit JSONLs
record top10 query indices and GT IoUs,not all candidate pairwise boxes;
they cannot establish the four-neighbor duplicate rate. Do not claim that
this rate has been measured.

## Pinned Open Source

Source text reviewed2026-11-04;repository heads resolved by git ls-remote.
No external implementation was executed and no package was installed.

TOOD,commit1bda27c6f290c5b828c566eb95ea50fed75a53cb:
https://github.com/fcjian/TOOD/blob/1bda27c6f290c5b828c566eb95ea50fed75a53cb/mmdet/core/bbox/assigners/task_aligned_assigner.py

Its assigner detaches class scores and predicted-box overlaps,combines
them as score^alpha times IoU^beta,and selects top-k candidates per GT.
Positive centers must also lie inside GT. Conflicts are resolved by IoU;
unassigned entries retain background assignment. This is a dense detector
assignment policy,not an MCLN query/token loss.

VarifocalNet,commitf459696240b4442b75bcebbbe4f0910ee4e0f66b:
https://github.com/hyz-xmaster/VarifocalNet/blob/f459696240b4442b75bcebbbe4f0910ee4e0f66b/mmdet/models/dense_heads/vfnet_head.py
https://github.com/hyz-xmaster/VarifocalNet/blob/f459696240b4442b75bcebbbe4f0910ee4e0f66b/mmdet/models/losses/varifocal_loss.py

The head initializes classification-quality targets to zero and writes
detached refined-box IoUs only at assigned positive locations/classes.
The loss applies sigmoid BCE with quality-dependent positive weighting
and a different negative weighting. Its targets follow its detector's
assignment. This is not an instruction to mark every query overlapping
the phrase target as a semantic positive.

## MCLN Transfer Boundary

MCLN already assigns semantic maps by Hungarian matches and supervises
unmatched queries as not mentioned. Its BBF uses token softmax followed
by role-map aggregation. Adding a candidate constant before token softmax
cancels;changing only BBS would miss the evaluated branch. See
PVG_READER_BBF_INTERFACE_AUDIT.md.

These references motivate making eligibility,quality targets,and gradient
destinations explicit. Neither repository proves the current MCLN loss is
wrong or that adding an IoU predictor will improve ScanRefer. A new quality
head would also change the current three-module implementation and cannot
be treated as a harmless parameter adjustment.

First classify full paired gains/losses by valid-candidate coverage and
ranking,including unique/multiple groups. If loss is concentrated in
ranking despite valid candidates,measure train-only eligibility against
the existing matcher before specifying any Reader supervision. If coverage
dominates,inspect predicted geometry and existing Support/Span evidence
instead of assuming a ranking loss can create missing boxes. These
diagnostic categories may coexist;they do not establish isolated causes.

Any later change must remain inside the Reader/Support/Span roles,retain
prediction-only inference,register its matched control,and compare both
thresholds to B2. Do not select validation examples as training data.
NR/SR scratch runs and budgets remain unchanged. No new training or
module-source change follows from this source review alone.

## Completed Paired Evidence: 2026-11-04 09:24 UTC

Reader full re-evaluation9508 completedexit0 with5471/4609;its050 result
is1hit above training-time4608. Both remain separate records. Exact
ordered input/point/GT pairing to B2 passed. At050,105 gained and152 lost
net-47. Lost samples have3/21/128 cases with no valid candidate/valid
candidate outside Top10/valid candidate inside Top10. Overall oracle
coverage changes8260to8252 and Top10 coverage6021to5917. Coverage-only
explanations do not describe most paired losses.

Among152 losses,52 selected reference boxes and11 native boxes are valid
before final fusion;their union is62. Thus the categories cannot be reduced
to semantic ranking alone. The reference count4713vs4710 is computed at
each model's own selected query,not fixed-query geometry or causal Support
benefit. Record both selection and fusion failure hypotheses. The existing
Span supervision pilot remains insufficient to claim improvement over B2.
No nearest-neighbor duplicate rate has been established.

## Separating The Three Module Questions

| Module | Existing evidence | Missing evidence before a change |
| --- | --- | --- |
| Reader | Fixed-B2 bypass nets+17 strict050 hits; competitor trial is-12 vs matched trained control and-48 vs B2 | Paired trial failure categories; train-only coverage/identity of the four nearest candidate neighbors |
| Support | Bypass loses40 strict050 hits; seven selected nonpositive-size cases coincide with extreme correction magnitudes | Per-cohort loss evidence showing how much regression is associated with these cases; seven cases do not explain289 missing target hits |
| Span | Fixed-B2 final geometry nets394 strict050 hits over native at the selected query; gate pilot is+34 vs trained control but-2 vs B2 | Whether the new trial's lost examples retain a valid native/reference box before fusion,then matched training evidence for a specific supervision change |

These results come from different experiments and cannot be added as
independent contributions. Preserve the positive B2 geometry gains while
testing any proposed change. Do not replace all three modules because a
single exploratory Reader variant regressed.

The original NR/SR scratch experiments do not use these modules. Current
all-epoch CSV shows NR completed62epochs,best59=4161/7899 and SR
completed23epochs,best15=11743/17726. Their recent ten025 hit counts are
NR4107,4004,4108,4090,4090,4103,4161,4114,4121,4141 and
SR11561,11743,11534,11536,11563,11426,11445,11579,11352,11388.
The fluctuations do not diagnose module transfer failure or justify
interrupting their240/140epoch budgets. Neither run has reached its
configured first LR decay (NR150,SR30).
