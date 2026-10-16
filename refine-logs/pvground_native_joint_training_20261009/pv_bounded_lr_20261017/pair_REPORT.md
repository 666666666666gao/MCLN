# Completed matched-endpoint normal-training comparison

Both complete arms used seed2027 and 9166 native train-loop iterations / 73328 row visits: two complete train epochs and full9508 last/bbs evaluation at E0,E1,E2. Both stopped after two adjacent strict declines including E0. Actual terminal controllers and children were absent with exit0.

|Arm|E0|E1|E2|
|---|---|---|---|
|Final duplicate|5677/4920|5651/4735|5631/4697|
|Coarse endpoint|5677/4920|5678/4742|5657/4722|

Same-epoch coarse-minus-control differences are +27/+7 at E1 and +26/+25 at E2. Coarse best trained E1 is +1/-178 versus the retained E0; no new model or effective third contribution is promoted. The auxiliary target location can affect training, but this recipe did not preserve the strong starting accuracy and does not prove a sole degradation cause.

New target: one complete model >=60.2%/52.0%, at least5724/4945 hits; retained5677/4920 is short47/25. The frozen paired-source protocol contains the historical5705/4945 target; this did not change its shared recipe or stop rule. Normal-pair fresh result integrity review and independent saved-row coordinate recount remain pending. No cold full-checkpoint recovery is claimed. The earlier real two-step sanity audit is not substituted for this normal-result audit.
