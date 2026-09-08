def _get_detected_objects(self, split, scan_id, augmentations):
    # Initialize
    all_detected_bboxes = np.zeros((MAX_NUM_OBJ, 6))
    all_detected_bbox_label_mask = np.array([False] * MAX_NUM_OBJ)
    detected_class_ids = np.zeros((MAX_NUM_OBJ,))
    detected_logits = np.zeros((MAX_NUM_OBJ, NUM_CLASSES))

    # note single stage method
    if not self.butd and not self.butd_cls:
        return (
            all_detected_bboxes, all_detected_bbox_label_mask,
            detected_class_ids, detected_logits
        )

    # Load: class, box, pc, logits
    detected_dict = np.load(
        f'{self.data_path}/group_free_pred_bboxes/group_free_pred_bboxes_{split}/{scan_id}.npy',
        allow_pickle=True
    ).item()

    all_bboxes_ = np.array(detected_dict['box'])
    classes = detected_dict['class']
    cid = np.array([DC.nyu40id2class[
        self.label_map[c]] for c in detected_dict['class']
    ])
    all_bboxes_ = np.concatenate((
        (all_bboxes_[:, :3] + all_bboxes_[:, 3:]) * 0.5,
        all_bboxes_[:, 3:] - all_bboxes_[:, :3]
    ), 1)

    assert len(classes) < MAX_NUM_OBJ
    assert len(classes) == all_bboxes_.shape[0]

    num_objs = len(classes)
    all_detected_bboxes[:num_objs] = all_bboxes_
    all_detected_bbox_label_mask[:num_objs] = np.array([True] * num_objs)
    detected_class_ids[:num_objs] = cid
    detected_logits[:num_objs] = detected_dict['logits']    # logits
    # Match current augmentations
    if self.augment and self.split == 'train':
        all_det_pts = box2points(all_detected_bboxes).reshape(-1, 3)
        if augmentations.get('yz_flip', False):
            all_det_pts[:, 0] = -all_det_pts[:, 0]
        if augmentations.get('xz_flip', False):
            all_det_pts[:, 1] = -all_det_pts[:, 1]
        all_det_pts = rot_z(all_det_pts, augmentations['theta_z'])
        all_det_pts = rot_x(all_det_pts, augmentations['theta_x'])
        all_det_pts = rot_y(all_det_pts, augmentations['theta_y'])
        all_det_pts += augmentations['shift']
        all_det_pts *= augmentations['scale']
        all_detected_bboxes = points2box(all_det_pts.reshape(-1, 8, 3))

    if self.augment_det and self.split == 'train':
        min_ = all_detected_bboxes.min(0)
        max_ = all_detected_bboxes.max(0)
        rand_box = (
            (max_ - min_)[None]
            * np.random.random(all_detected_bboxes.shape)
            + min_
        )
        corrupt = np.random.random(len(all_detected_bboxes)) > 0.7
        all_detected_bboxes[corrupt] = rand_box[corrupt]
        detected_class_ids[corrupt] = np.random.randint(
            0, len(DC.nyu40ids), (len(detected_class_ids))
        )[corrupt]
    return (
        all_detected_bboxes, all_detected_bbox_label_mask,
        detected_class_ids, detected_logits
    )
