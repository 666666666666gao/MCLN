"""Joint-detection rows must not receive the referring-expression auxiliary loss."""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'models'))
from pvground_rec_competition import competition_loss
from pvground_referit_rec_competition import referit_competition_loss
from test_pvground_rec_competition import fixture


def main():
    for dataset in ('nr3d', 'sr3d'):
        prediction, batch, matching = fixture()
        reference, _ = competition_loss(prediction, batch, matching)
        mixed_prediction = {key: value.detach().repeat(2, 1, 1).requires_grad_()
                            for key, value in prediction.items()}
        mixed_batch = {key: value.repeat((2,) + (1,) * (value.ndim - 1))
                       for key, value in batch.items()}
        mixed_batch['sample_dataset'] = [dataset, 'scannet']
        # The real joint batch reports the language dataset even on detection rows.
        mixed_batch['language_dataset'] = [dataset, dataset]
        loss, counts = referit_competition_loss(mixed_prediction, mixed_batch,
                                               matching * 2, dataset)
        assert torch.equal(loss, reference / 2)
        assert counts['eligible25'] == counts['eligible50'] == 1
        loss.backward()
        gradient = mixed_prediction['last_sem_cls_scores'].grad
        assert bool((gradient[0] != 0).any()) and bool((gradient[1] == 0).all())
        mixed_batch['sample_dataset'] = ['scannet', 'scannet']
        loss, counts = referit_competition_loss(mixed_prediction, mixed_batch,
                                               matching * 2, dataset)
        assert float(loss) == 0 and not any(counts.values())
    print('REFERIT_REC_COMPETITION_CPU_PASS mixed rows and detection-only rows for nr3d/sr3d')


if __name__ == '__main__':
    main()
