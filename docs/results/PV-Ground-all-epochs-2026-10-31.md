# PV-Ground Complete Epoch Metrics

Only full native last/bbf semantic-alignment evaluations are included.
Epoch0 is untrained initialization. Historical runs are separate experiments;
interrupted epochs have no result. Same-row accuracies use the same model.
ScanRefer selects trained Acc050 first; NR3D/SR3D select trained Acc025 first.
The other threshold breaks ties. Selection here does not prove weight retention.
Author-baseline reproduction results are documented separately in
PV_GROUND_MODULE_FT_2026-10-30.md.

## Current Trained Selection

| Dataset | Run | Epoch | Acc025 | Acc050 |
| --- | --- | ---: | ---: | ---: |
| scanrefer | 1793345477 | 23 | 59.339504% | 50.588978% |
| nr3d | 1793419199 | 5 | 60.653247% | 49.297379% |
| sr3d | 1793419200 | 3 | 67.454586% | 57.463613% |

## All Complete Epochs

| Dataset | Run | Scope | Epoch | Rows | Hits025 | Hits050 | Acc025 | Acc050 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scanrefer | 1793345477 | current | 0 (init) | 9508 | 5606 | 4843 | 58.960875% | 50.936054% |
| scanrefer | 1793345477 | current | 1 | 9508 | 5559 | 4542 | 58.466554% | 47.770299% |
| scanrefer | 1793345477 | current | 2 | 9508 | 5552 | 4538 | 58.392932% | 47.728229% |
| scanrefer | 1793345477 | current | 3 | 9508 | 5619 | 4641 | 59.097602% | 48.811527% |
| scanrefer | 1793345477 | current | 4 | 9508 | 5580 | 4581 | 58.687421% | 48.180480% |
| scanrefer | 1793345477 | current | 5 | 9508 | 5598 | 4600 | 58.876735% | 48.380311% |
| scanrefer | 1793345477 | current | 6 | 9508 | 5597 | 4629 | 58.866218% | 48.685318% |
| scanrefer | 1793345477 | current | 7 | 9508 | 5623 | 4716 | 59.139672% | 49.600337% |
| scanrefer | 1793345477 | current | 8 | 9508 | 5613 | 4720 | 59.034497% | 49.642406% |
| scanrefer | 1793345477 | current | 9 | 9508 | 5608 | 4739 | 58.981910% | 49.842238% |
| scanrefer | 1793345477 | current | 10 | 9508 | 5617 | 4776 | 59.076567% | 50.231384% |
| scanrefer | 1793345477 | current | 11 | 9508 | 5644 | 4779 | 59.360538% | 50.262936% |
| scanrefer | 1793345477 | current | 12 | 9508 | 5632 | 4781 | 59.234329% | 50.283971% |
| scanrefer | 1793345477 | current | 13 | 9508 | 5626 | 4773 | 59.171224% | 50.199832% |
| scanrefer | 1793345477 | current | 14 | 9508 | 5601 | 4769 | 58.908288% | 50.157762% |
| scanrefer | 1793345477 | current | 15 | 9508 | 5645 | 4778 | 59.371056% | 50.252419% |
| scanrefer | 1793345477 | current | 16 | 9508 | 5612 | 4756 | 59.023980% | 50.021035% |
| scanrefer | 1793345477 | current | 17 | 9508 | 5635 | 4793 | 59.265881% | 50.410181% |
| scanrefer | 1793345477 | current | 18 | 9508 | 5641 | 4800 | 59.328986% | 50.483803% |
| scanrefer | 1793345477 | current | 19 | 9508 | 5620 | 4767 | 59.108119% | 50.136727% |
| scanrefer | 1793345477 | current | 20 | 9508 | 5606 | 4776 | 58.960875% | 50.231384% |
| scanrefer | 1793345477 | current | 21 | 9508 | 5643 | 4791 | 59.350021% | 50.389146% |
| scanrefer | 1793345477 | current | 22 | 9508 | 5612 | 4763 | 59.023980% | 50.094657% |
| scanrefer | 1793345477 | current | 23 | 9508 | 5642 | 4810 | 59.339504% | 50.588978% |
| nr3d | 1793382384 | historical | 0 (init) | 7899 | 4739 | 4132 | 59.994936% | 52.310419% |
| nr3d | 1793382384 | historical | 1 | 7899 | 4786 | 3918 | 60.589948% | 49.601215% |
| nr3d | 1793382384 | historical | 2 | 7899 | 4780 | 3909 | 60.513989% | 49.487277% |
| nr3d | 1793382384 | historical | 3 | 7899 | 4787 | 3902 | 60.602608% | 49.398658% |
| nr3d | 1793382384 | historical | 4 | 7899 | 4774 | 3918 | 60.438030% | 49.601215% |
| nr3d | 1793382384 | historical | 5 | 7899 | 4787 | 3902 | 60.602608% | 49.398658% |
| nr3d | 1793382384 | historical | 6 | 7899 | 4784 | 3910 | 60.564628% | 49.499937% |
| nr3d | 1793382384 | historical | 7 | 7899 | 4772 | 3900 | 60.412710% | 49.373338% |
| nr3d | 1793382384 | historical | 8 | 7899 | 4768 | 3925 | 60.362071% | 49.689834% |
| nr3d | 1793417914 | historical | 0 (init) | 7899 | 4739 | 4132 | 59.994936% | 52.310419% |
| nr3d | 1793419199 | current | 0 (init) | 7899 | 4739 | 4132 | 59.994936% | 52.310419% |
| nr3d | 1793419199 | current | 1 | 7899 | 4783 | 3918 | 60.551969% | 49.601215% |
| nr3d | 1793419199 | current | 2 | 7899 | 4783 | 3916 | 60.551969% | 49.575896% |
| nr3d | 1793419199 | current | 3 | 7899 | 4781 | 3915 | 60.526649% | 49.563236% |
| nr3d | 1793419199 | current | 4 | 7899 | 4781 | 3918 | 60.526649% | 49.601215% |
| nr3d | 1793419199 | current | 5 | 7899 | 4791 | 3894 | 60.653247% | 49.297379% |
| nr3d | 1793419199 | current | 6 | 7899 | 4783 | 3893 | 60.551969% | 49.284720% |
| nr3d | 1793419199 | current | 7 | 7899 | 4767 | 3889 | 60.349411% | 49.234080% |
| sr3d | 1793380711 | historical | 0 (init) | 17726 | 11620 | 10023 | 65.553424% | 56.544060% |
| sr3d | 1793380711 | historical | 1 | 17726 | 11944 | 10166 | 67.381248% | 57.350784% |
| sr3d | 1793380711 | historical | 2 | 17726 | 11934 | 10206 | 67.324834% | 57.576441% |
| sr3d | 1793380711 | historical | 3 | 17726 | 11947 | 10167 | 67.398172% | 57.356426% |
| sr3d | 1793419200 | current | 0 (init) | 17726 | 11620 | 10023 | 65.553424% | 56.544060% |
| sr3d | 1793419200 | current | 1 | 17726 | 11944 | 10164 | 67.381248% | 57.339501% |
| sr3d | 1793419200 | current | 2 | 17726 | 11944 | 10187 | 67.381248% | 57.469254% |
| sr3d | 1793419200 | current | 3 | 17726 | 11957 | 10186 | 67.454586% | 57.463613% |

## Source Snapshots

| File | Observed UTC | SHA256 |
| --- | --- | --- |
| ScanRefer-full-epoch-history-2026-10-31.json | 2026-10-31T12:29:17.850734+00:00 | 65d5147abc289042838de874b75b14696dbb8d2c06ba9b54c87555e57f08f8a9 |
| NR3D-full-epoch-history-2026-10-31.json | 2026-10-31T13:15:47.097097+00:00 | 13ed18854465467f63960347637eb9c9b5bddbdf3c21c8c90a2b168af22102b6 |
| SR3D-full-epoch-history-2026-10-31.json | 2026-10-31T12:55:51.446269+00:00 | fb32464a33d7fa863a04c829ff8411eacb333bf9a35e431e586fd630a297cbc4 |

sr3d/1793418888: no complete evaluation.
