"""CPU checks for frozen-state drift and checkpoint-epoch attribution."""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from pvground_checkpoint_update_audit import audit


class CheckpointUpdateAuditTests(unittest.TestCase):
    def test_frozen_buffer_drift_and_unchanged_module_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefixes = ("decoder.5.source_query_read.", "candidate_support_corrector.",
                        "candidate_span_mixer.")
            names = [prefix + "output.bias" for prefix in prefixes]
            initial = {"module." + name: torch.zeros(1) for name in names}
            initial["module.core.running_mean"] = torch.zeros(2)
            initial["module.core.num_batches_tracked"] = torch.tensor(0)
            trained = {name: value.clone() for name, value in initial.items()}
            trained["module." + names[0]].fill_(.25)
            trained["module." + names[1]].fill_(.5)
            trained["module.core.running_mean"][0] = 1
            architecture = dict(dataset="scanrefer")
            torch.save(dict(epoch=0, architecture=architecture, model=initial), root / "init.pth")
            retained = dict(epoch=0, rows=9508, hits025=5606, hits050=4843,
                            primary_score="last/bbf", same_complete_model=True)
            current = dict(retained, epoch=17, hits025=5635, hits050=4793)
            torch.save(dict(epoch=17, architecture=architecture, model=trained,
                            retained_metrics=retained), root / "trained.pth")
            (root / "trainable.json").write_text(json.dumps(dict.fromkeys(names, 1)))
            (root / "metrics.jsonl").write_text(
                json.dumps(retained) + "\n" + json.dumps(current) + "\n")
            result = audit(root / "init.pth", root / "trained.pth",
                           root / "trainable.json", root / "metrics.jsonl")
            self.assertFalse(result["frozen_state_unchanged"])
            self.assertEqual(result["changed_frozen_state"], ["core.running_mean"])
            self.assertEqual(result["frozen_state_tensors"], 2)
            self.assertEqual([row["changed_tensors"] for row in result["modules"]], [1, 1, 0])
            self.assertEqual(result["own_epoch_metrics"], current)
            self.assertEqual(result["parameter_deltas"][0]["delta_l2"], .25)


if __name__ == "__main__":
    unittest.main()
