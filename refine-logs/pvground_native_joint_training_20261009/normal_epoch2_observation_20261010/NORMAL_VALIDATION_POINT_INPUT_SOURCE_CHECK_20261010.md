# Validation point input source check — 2026-10-10 09:10 CST

Scope: read existing local source only. No SSH, dataset execution, model evaluation, or active training changes.

The dataset loads cached `val_v3scans.pkl` into `self.scans` (`joint_det_dataset.py`, lines 209–215). Each expression restores `scan.pc` from `scan.orig_pc` (lines 1272–1274). `_get_pc` reads these stored points directly; point/color augmentation runs only for `split == 'train' and self.augment` (lines 871–918). GT box jitter and detected-box augmentation are also guarded by the training split. Cached scan creation uses seed 1184 for the 50,000-point choice (`visual_data_handlers.py`, lines 112–125).

The train and validation DataLoaders share a generator, and its seed changes per training epoch. This source fact does not imply different validation point samples: the examined validation path does not resample points in `__getitem__`.

Conclusion: this source check found no evidence that epoch-to-epoch validation point resampling explains the observed E1 decline. It does not establish bitwise model determinism or independently verify cached arrays. Preserve the active experiment and its original scheduled observation at 12:01:45.656476 CST.
