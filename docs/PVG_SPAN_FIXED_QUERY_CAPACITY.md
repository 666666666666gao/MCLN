# Span Fixed-Query Capacity Audit

## Result: 2026-11-04 10:27 UTC

Remote stdlib-only postprocessing of the two existing full9508-record
evaluation archives computed the continuous-gate optimum at each model's
recorded selected query. No model import,forward,training or new validation
was performed. The running Span-only control/treatment is unchanged.

| Archive | Recorded025 / 050 | Continuous gate oracle025 / 050 | Oracle percentages025 / 050 |
| --- | ---: | ---: | ---: |
| B2 | 5483 / 4656 | 5633 / 5158 | 59.244846 / 54.249053 |
| Reader re-evaluation | 5471 / 4609 | 5629 / 5150 | 59.202777 / 54.164914 |

The earlier125-point gate grid found B25631/5152 and Reader5628/5136.
The new method enumerates all piecewise endpoint combinations,so its
search covers the continuous gate family rather than that sampled grid.
Each threshold uses the same selected queries and boxes in its archive.
The two thresholds' oracle-optimal gates can differ;this table does not
establish that one gate setting or learned model attains both counts.

On the B2 archived fixed-query box family,the025 optimum5633 remains
72hits below the5705-hit target. This is a capacity limitation of changing
only the existing three gates while keeping selected queries,native boxes
and reference boxes fixed. It is not a global ceiling for MCLN,Reader,
Support,another input point sample,or another box family.

The050 oracle5158 exceeds the4945-hit target,but its gap over recorded
4656 is not attainable gain or evidence the current loss can learn it.
GT chooses a different gate for every diagnostic sample;GT is never an
inference input. The currently registered experiment tests learnability
and remains useful even though gate-only changes cannot meet the025
target on this fixed archived family.

## Derivation

Source boundary:ExtremalSpanMixer.mix_one in models/pvground_innovations.py.
For each axis,the same clamped gate g in[0,1] mixes center and size;
both input sizes are clamped to at least1e-6. Thus predicted lower face,
upper face and size are affine functions of g. Axes have independent gates.

Split each axis's interval at0,1 and every in-range crossing of either
predicted face with either GT face. There are at most six breakpoints.
Between two adjacent breakpoints,the intersection length is affine or
identically zero,and the predicted size is affine and positive.

Within any Cartesian product of three such intervals,hold two gates
fixed. Intersection volume and predicted volume are affine in the
remaining gate. IoU is then a ratio of two affine functions with positive
denominator. Its derivative has constant sign or is zero,so moving this
gate to one endpoint cannot reduce IoU. Repeating for all three gates
reaches a cell vertex without reducing IoU. Searching all combinations
of axis breakpoints therefore contains a continuous global maximizer.

The denominator is positive because union volume is at least the
positive GT volume. The implementation also requires GT volume>1e-8,
matching the existing diagnostic denominator convention without its
epsilon clamp becoming active. All archived samples passed.

This argument is derived for this box family;it is not a borrowed
network method or an independent external mathematical review.

## Numerical Evidence And Limits

Helper:tools/audit_scanrefer_span_continuous_oracle.sh.
Receipt:results/mcln_pvground_20261101/scanrefer-span-continuous-oracle-summary-20261104.json.

Remote numerical witnesses passed:constant identical boxes,an interior
exact match at a gate absent from the earlier grid,and64seeded random
box triples compared against1331grid combinations each. Largest observed
grid excess was2.08e-17,within the1e-12 floating tolerance. Every archived
selected gate's reconstructed IoU was at most its computed optimum
within1e-12. Both archives passed9508/evaluator-footer/current-hit checks
and match the prior JSONL SHA256 values.

No computed optimum was within1e-9 of either threshold. The closest
distances wereB2 .000511909 at025/.0000249586 at050 and Reader
.00116737/.000177482. These are double-precision numerical results
supported by the piecewise proof and numerical witnesses,not interval
arithmetic certificates or a replacement for actual full validation.

## Next Experimental Decision

Finish the registered weight0/0.1 comparison and its full checkpoint
freeze audit before changing its recipe. Interpret any gain as measured
Span learnability within a fixed box family. For the joint60/52goal,
later work also needs evidence-backed improvement to selected candidates
or upstream native/reference boxes while preserving the three module
roles. Do not respond to a gate-only gain by repeatedly tuning gates
under an assumption that this fixed family can meet both goals.

The paired Reader diagnosis already shows most strict050 regressions
retain a valid candidate inTop10;that motivates selection investigation,
but does not identify a successful Reader modification. Keep the
Hungarian/not-mentioned supervision and BBF interface constraints in
PVG_READER_QUALITY_ALIGNMENT_REFERENCE.md. No architecture change,new
quality head or duplicate-neighbor mechanism is justified by this oracle
alone. NR/SR original scratch runs and full budgets remain unchanged.
