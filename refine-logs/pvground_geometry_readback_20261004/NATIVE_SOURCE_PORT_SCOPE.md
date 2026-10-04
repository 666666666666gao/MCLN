# Delayed semantic source port: prepared, not deployed

The separate source_preview contains two files generated from the exact pinned
real-model and head sources in READBACK_INTERFACE_WITNESS. Only the final semantic
subhead is deferred when the explicit readback installation is present. Geometry,
objectness, heading, intermediate semantic predictions, contrast and both native
Mask paths retain their existing order. The same semantic subhead runs once after
the current final box refinement. Parameter names of existing native heads stay
unchanged. No active face source or job was modified.

Installation requires an explicit existing geometry provider and final semantic
head. The provider is not chosen or loaded here. Parent checkpoint loading must
precede readback installation, and a real factory must witness the new state and
optimizer parameter set. There is no fallback, silent missing-state loading or
second deployed score. All256 candidates remain.

native_root_bbs.py spells out the actual current ScanRefer root formula: softmax
over tokens, binary positive-token sum, modifier/pronoun/relation map evidence,
minus other_entity evidence. This is not the .6/.2/.2/.1 native CE target and is
not a generic category probability suitable for BCE. The helper is not currently
called by a training or deployed evaluator. Equality to the real evaluator and
gradient response still require the future real-batch witness.

SOURCE_PORT_DRAFT_AST_ONLY means Python3.7 syntax and SHA-bound generation only.
It is not an actual native factory, source-review PASS, GPU execution, optimizer
update or accuracy result. Zero-residual equality, native semantic call count,
BN/Dropout state, fixed geometry/Mask outputs, actual score response, standalone
semantic gradients and exact save/restore are required before any deployment.

The parent PV model has29 inherited lines with trailing whitespace. They are
preserved in the source snapshot rather than reformatted as unrelated changes;
the explicit publication attribute applies only to this source_preview subtree.
Current training and4506 metric-best retention remain unchanged. The eventual
structural evidence visibility control and quality-supervision experiment stay
separate; Nr3D/Sr3D native maps/coefficients have not been adapted here.
