import torch

from scripts.native_teacher_box_transfer import native_teacher_box_loss


def example():
    boxes = torch.tensor([[[4., 0., 0., 2., 2., 2.], [.7, 0., 0., 2., 2., 2.]]], requires_grad=True)
    root = torch.tensor([[0., 0., 0., 2., 2., 2.]])
    teacher = torch.tensor([[.1, 0., 0., 2., 2., 2.]], requires_grad=True)
    matches = [(torch.tensor([1]), torch.tensor([0]))]
    return boxes, teacher, root, matches


def test_better_teacher_updates_only_current_matched_box_and_stays_frozen():
    boxes, teacher, root, matches = example()
    loss, stats = native_teacher_box_loss(boxes, teacher, root, matches)
    assert stats['eligible'].tolist() == [True]
    loss.backward()
    assert teacher.grad is None
    assert boxes.grad[0, 0].abs().sum() == 0
    assert boxes.grad[0, 1, 0] > 0


def test_student_query_permutation_preserves_loss_with_current_assignment():
    boxes, teacher, root, matches = example()
    before, _ = native_teacher_box_loss(boxes, teacher, root, matches)
    permuted, stats = native_teacher_box_loss(boxes[:, [1, 0]], teacher, root, [(torch.tensor([0]), torch.tensor([0]))])
    assert torch.equal(before, permuted)
    assert stats['student_query_indices'].tolist() == [0]


def test_worse_teacher_has_zero_auxiliary_gradient():
    boxes, teacher, root, matches = example()
    teacher = torch.tensor([[1.2, 0., 0., 2., 2., 2.]])
    loss, stats = native_teacher_box_loss(boxes, teacher, root, matches)
    assert not stats['eligible'].any() and loss.item() == 0.
    loss.backward()
    assert boxes.grad.abs().sum() == 0


def test_teacher_below_target_threshold_cannot_supply_positive_box_label():
    boxes, teacher, root, matches = example()
    boxes = boxes.detach().clone().requires_grad_()
    with torch.no_grad(): boxes[0, 1, 0] = 4.
    teacher = torch.tensor([[1.5, 0., 0., 2., 2., 2.]])
    loss, stats = native_teacher_box_loss(boxes, teacher, root, matches)
    assert stats['teacher_root_ious'][0] > stats['student_root_ious'][0]
    assert not stats['eligible'].any() and loss.item() == 0.


def test_nonroot_assignment_is_not_used_as_referring_target():
    boxes, teacher, root, _ = example()
    loss, stats = native_teacher_box_loss(boxes, teacher, root, [(torch.tensor([0, 1]), torch.tensor([3, 0]))])
    assert stats['student_query_indices'].tolist() == [1]
    assert loss.item() > 0
