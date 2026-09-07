def build_probe_dataset(dataset_class, args, annotation, rows, dset):
    language = [row for row in rows if row['dataset'] == dset]
    detection = [row for row in rows if row['dataset'] == 'scannet']
    assert len(language) == 12 and len(detection) == 4
    assert len({r['scan_id'] for r in language}) == 12

    class ProbeDataset(dataset_class):
        def _scene_graph_parse(self, annos):
            assert len(annos) == annotation['protocols'][dset]['train']['language_rows']
            chosen = []
            for item in language:
                row = annos[item['raw_language_row_id']]
                for key in ['dataset', 'scan_id', 'target_id', 'target', 'utterance', 'anchor_ids']:
                    assert row[key] == item[key], (dset, key)
                chosen.append(row)
            annos[:] = chosen
            super()._scene_graph_parse(annos)

        def load_scannet_annos(self):
            annos = super().load_scannet_annos()
            assert len(annos) == 1199
            chosen = [annos[item['raw_detection_row_id']] for item in detection]
            assert [r['scan_id'] for r in chosen] == [r['scan_id'] for r in detection]
            return chosen

    dataset = ProbeDataset(dataset_dict={dset: 1, 'scannet': 10}, test_dataset=dset,
        split='train', data_path=args.data_root, use_color=args.use_color,
        use_height=args.use_height, use_multiview=args.use_multiview,
        detect_intermediate=args.detect_intermediate, butd=args.butd, butd_gt=args.butd_gt,
        butd_cls=args.butd_cls, augment_det=args.augment_det,
        skip_missing_superpoints=args.skip_missing_superpoints)
    assert len(dataset) == 52  # twelve language rows plus four detection rows repeated ten times
    dataset.annos = dataset.annos[:16]
    assert [r['scan_id'] for r in dataset.annos] == [r['scan_id'] for r in rows]
    assert dataset.joint_det and dataset.augment and not dataset.use_sacr_source
    return dataset
