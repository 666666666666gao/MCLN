# L1 endpoint loading and zero-start reconciliation

The active training run and its scientific contract are unchanged. The new
load_nr3d_l1_terminal.py helper reads only the registered position-key endpoint
for an isolated evaluation. It checks artifact bytes against the terminal
receipt's SHA256, saved mode=position, steps=6687 and parent checkpoint identity,
then strictly loads its state into a frozen CPU TextPositionKey(288,8,position).
The caller must move it to the evaluated model's device before attaching it.

Both saved L1 arms contain the same-shaped288x288 weight, but their input
meanings differ. A shape-only load could accept a text-control weight and
interpret it as a position weight. This is why the existing saved mode and
parent metadata must be checked. The loader does not choose a model, verify
accuracy, or authorize promotion. The paired terminal screen must separately
pass against both controls before any new formal evaluation.

Five CPU tests passed in the actual Py3.7/Torch1.10.2 environment:
serialized position-bias output round-trip identity, frozen loaded parameters,
and rejection of a text-arm artifact, incomplete endpoint, different parent or
unrecorded digest. Fixtures are synthetic; no live training state was read or
changed, no GPU forward was made, and no optimizer update occurred.
Evidence: refine-logs/l1_endpoint_reader_cpu_20260905_v1/.

The complete6172-row L1 zero-start output was also compared with M5's native
start. All row IDs and scene IDs match; every sampled point-cloud SHA256,
selected REC Query/Box IoU and selected Mask Query/Mask IoU is exactly equal.
The mismatch counts for all six compared fields are zero. This establishes
the current paired start for the saved fields, not equality of every unrecorded
Query tensor or token input. No trained endpoint was examined.

L1 baseline SHA256:
035a9f3c90cfe4ff5faaed1beeabc3093fd6c7941a2adbee30bf7b9e2b1faec6.
M5 baseline SHA256:
ba9968e48a3b39cac92802a94053eba90f8d11db70ebbfc0b1168273ccbd6f81.
Detailed reconciliation:
refine-logs/text_position_l1_20260905_v2/train/zero_start_m5_reconciliation.json.

This does not establish identity with the older P2/R1 cache, whose protected
strict REC count differs by6. All L1 scientific comparisons continue to use
its own fixed start and simultaneous text-key control. These checks provide
no new formal benchmark metric or evidence of an accuracy gain from training.
