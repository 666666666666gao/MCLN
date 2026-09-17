"""Use the native REC competition only for referring rows in joint training."""
import torch

from pvground_rec_competition import competition_loss


def referit_competition_loss(predictions, batch, last_indices, dataset):
    assert dataset in ('nr3d', 'sr3d')
    sources = batch['sample_dataset']
    assert len(sources) == len(last_indices)
    assert all(source in (dataset, 'scannet') for source in sources)
    rows = [i for i, source in enumerate(sources) if source == dataset]
    if not rows:
        return predictions['last_sem_cls_scores'].sum() * 0, dict(
            eligible25=0, eligible50=0, active25=0, active50=0)
    ids = torch.tensor(rows, device=predictions['last_sem_cls_scores'].device)
    selected_predictions = {key: predictions[key][ids] for key in (
        'last_sem_cls_scores', 'last_center', 'last_pred_size')}
    selected_batch = {key: batch[key][ids] for key in (
        'positive_map', 'modify_positive_map', 'pron_positive_map',
        'rel_positive_map', 'other_entity_map', 'box_label_mask',
        'center_label', 'size_gts')}
    loss, counts = competition_loss(selected_predictions, selected_batch,
                                    [last_indices[i] for i in rows])
    # Preserve the 2*full-batch denominator; detection rows contribute zero.
    return loss * (len(rows) / len(sources)), counts
