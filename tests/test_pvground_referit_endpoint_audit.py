"""Reject malformed mixed-row logs independently of the trainer."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_pvground_referit_endpoint import audit_training_rows


def fixture():
    partition=dict(fit_ids=list(range(10)),holdout_ids=[10],original_language_rows=6,
                   language_fit_ids=list(range(6)),detection_kept_base_ids=[0,1],detection_repeats=2)
    spec=dict(batch_size=8,total_steps=2,dataset='nr3d')
    rows=[]
    for step,ids in enumerate([[0,6,1,7,2,8,3,9],[4,5]],1):
        rows.append(dict(step=step,total_steps=2,rows=ids,
            sample_dataset=['nr3d' if i<6 else 'scannet' for i in ids],
            loss=1.2,loss_native=1.,loss_rec_competition=.2,grad_norm=.1,seconds=1.,
            loss_bbox=.1,loss_giou=.1,loss_ce=.1,loss_sem_align=.1,
            eligible25=2,active25=1,eligible50=1,active50=1))
    return rows,partition,spec


class AuditTests(unittest.TestCase):
    def test_complete_mixed_log(self):
        rows,part,spec=fixture()
        result=audit_training_rows(rows,part,spec)
        self.assertEqual(result['language_rows'],6)
        self.assertEqual(result['detection_rows'],4)

    def test_duplicate_row_and_missing_row_rejected(self):
        rows,part,spec=fixture();rows[0]['rows'][3]=rows[0]['rows'][1]
        with self.assertRaises(AssertionError):audit_training_rows(rows,part,spec)

    def test_language_dataset_cannot_label_detection_rows(self):
        rows,part,spec=fixture();rows[0]['sample_dataset']=['nr3d']*8
        with self.assertRaises(AssertionError):audit_training_rows(rows,part,spec)

    def test_competition_pairs_cannot_include_detection(self):
        rows,part,spec=fixture();rows[0]['eligible25']=5
        with self.assertRaises(AssertionError):audit_training_rows(rows,part,spec)

    def test_wrong_loss_sum_rejected(self):
        rows,part,spec=fixture();rows[0]['loss']=1.5
        with self.assertRaises(AssertionError):audit_training_rows(rows,part,spec)


if __name__=='__main__':unittest.main()
