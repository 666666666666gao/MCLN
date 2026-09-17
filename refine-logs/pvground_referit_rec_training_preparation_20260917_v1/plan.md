# ReferIt F continuation, fixed before Scan F endpoint

Only after the current Scan F formal9508 integrity and REC-only pass, and the
dataset's real F backward probe restores all state, may full training start.
This preparation does not queue full training. Preserve current Scan execution.

Use the same task-observation architecture and native-bbs competition as Scan F.
Initialize each dataset from its own published full pretrained parent (Nr epoch25,
Sr epoch31), not a failed endpoint or Scan checkpoint. Native butd_cls supplies
instance boxes plus predicted classes. All native losses remain; Mask is diagnostic.
Only actual referring rows receive the auxiliary competition; detection rows
contribute zero to it, with the full batch denominator retained.

Single seed2027, batch8, one fixed fit pass, LR/backbone LR1e-5, native AdamW
weight decay0.0005, clip0.1, no intermediate selection or added epochs.
Nr:36747 fit /6172 holdout,4594 updates. Sr:65558 /10328,8195 updates.
Physical-room detection exclusions and annotation order are bound by saved hashes.
Module holdout was seen by upstream pretraining; it is not formal scene generalization.

Record full initial and terminal holdout bbs/bbf boxes/scores/rows. bbs is primary;
only both bbs thresholds nonregressing versus its own initial allow formal evaluation.
No Mask gate. Formal Nr7899 and Sr17726 use the same architecture and bbs rule.
Target Nr at least4726/4059 hits (exceeds59.82/51.38), Sr at least12139/10335
(preserves protected68.4813/58.3042, exceeding68.43/57.30).
These are intended acceptance rules; the formal execution/audit must still be prepared.

Use serial GPU lock, disk capacity check, initial/terminal independent audit and
exact-row coverage before deleting superseded latest. Do not launch until complete
controller/evaluation continuation and disk provision are ready.
