from types import SimpleNamespace

import train_dist_mod


def test_joint_detection_debug_holdout_uses_scanrefer_only(monkeypatch):
    calls = []

    class FakeDataset:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            partition = kwargs["scanrefer_debug_scene_partition"]
            if partition == "train":
                scene_ids = ["train_{:03d}".format(index) for index in range(128)]
            else:
                scene_ids = [
                    "holdout_{:03d}".format(min(index, 119))
                    for index in range(128)
                ]
            self.annos = [{"scan_id": scene_id} for scene_id in scene_ids]
            self.augment = True

        def __len__(self):
            return len(self.annos)

    monkeypatch.setattr(train_dist_mod, "Joint3DDataset", FakeDataset)
    args = SimpleNamespace(
        dataset=["scanrefer"],
        test_dataset="scanrefer",
        joint_det=True,
        debug=True,
        debug_train_holdout=True,
        use_color=True,
        use_height=False,
        use_multiview=False,
        data_root="/data",
        detect_intermediate=True,
        butd=False,
        butd_gt=False,
        butd_cls=False,
        augment_det=True,
        skip_missing_superpoints=True,
        use_sacr_source=False,
        use_sacr_score_refiner=False,
        eval=False,
    )

    train_dataset, test_dataset = train_dist_mod.TrainTester.get_datasets(args)

    assert len(train_dataset) == 128
    assert len(test_dataset) == 128
    assert [call["dataset_dict"] for call in calls] == [
        {"scanrefer": 1},
        {"scanrefer": 1},
    ]
    assert calls[0]["augment_det"] is True
    assert calls[1]["augment_det"] is False
    assert test_dataset.augment is False
