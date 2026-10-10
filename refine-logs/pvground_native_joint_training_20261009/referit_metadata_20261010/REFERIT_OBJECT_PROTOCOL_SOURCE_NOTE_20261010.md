# Nr3D / Sr3D object-input protocol: source findings only

The actual author-core CPU constructors are closed and published in section20.376.133. They prove strict core loading, not compatibility of real training inputs. Keep the current ScanRefer training and the sealed R3 source unchanged.

The retained author checkpoint configurations for both datasets have `butd=False`, `butd_cls=True`, `butd_gt=False`, `joint_det=True`, `detect_intermediate=True`, and `augment_det=False`. The CPU constructor used the current ScanRefer object flags instead. Both combinations construct the object branch because `TrainTester.get_model` uses `butd or butd_gt or butd_cls`; matching state shapes does not make these protocols equivalent.

The exact native dataset file is `../runtime_binding/dataset_source/src/joint_det_dataset.py`, SHA256 `3afebfc69232f9547e6810563ba6dd4f34a0f770f929df2f4203b21493de0e3d`, equal to `NATIVE_SOURCE_PORT.json`:

- Lines1197 onward: `butd` and `butd_cls` both enter `_get_detected_objects`; this loads the GroupFree file before later overrides.
- Lines1367 onward: `butd_cls` then replaces boxes/masks with the scene object boxes and uses saved classifier predictions for the object classes. These are ground-truth scene proposals plus predicted classes, not GroupFree proposal boxes and not ground-truth target selection.
- Lines1094 onward: `detect_intermediate` can add the first anchor as an additional target when the annotation has an auxiliary entity and anchor IDs. Root-only synthetic checks do not prove this real multi-GT interface.
- `TrainTester.get_datasets` adds ScanNet detection annotations with multiplicity10 when `joint_det=True`. Dataset lines239-255 use the mixed list for training but load only the chosen benchmark for validation. Training exposure and steps therefore must be computed from the actual constructed dataset, not copied from the ScanRefer36665 count.
- The current sealed R3 entrypoint at line410 explicitly requires `butd=True`, `butd_cls=False`, `joint_det=False`. It cannot launch the author protocol without a separate, explicitly reviewed change. This is an intentional current protocol restriction, not a passed author-protocol test.

Future baseline and final-method runs must use and report the same chosen object/detection protocol and corresponding pretrained core. If the author protocol is chosen, change only the isolated future entrypoint flags and launch configuration, then check actual referring, anchor, and joint-detection rows. Current G/C protects already matched queries and excludes detection rows, but real multi-GT behavior still needs verification. No new target GT selector, GT anchor input, fallback, or dataset-specific ranking is proposed.

Do this only after the final ScanRefer architecture is selected. The R3 model interface is a preparation snapshot, not evidence that the eventual three effective mechanisms have been established. No model edit, loader execution, GPU admission, training query, or training launch was performed for this source note.
