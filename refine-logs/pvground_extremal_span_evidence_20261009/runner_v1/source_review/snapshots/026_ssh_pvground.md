# Existing AutoDL PV-Ground runtime

Warm reuse; no environment rebuild or package installation for this port.

- Runtime: `/root/autodl-tmp/mcln_pvground_runtime_20260908_v1`.
- Existing env spec SHA256:
  `966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c`.
- Python 3.7, Torch 1.10.2+cu111; existing isolated spconv/cumm and compiled
  PV/OpenPCDet operators. Single A100 40GB.
- Runtime/source/official ScanRefer checkpoint re-read on 2026-10-02; see
  `refine-logs/cs_pvground_20261002/resource_receipt.json`.
- The prior kernel witness exists in the runtime. This implementation must also
  pass real model forward/backward, requested-batch capacity, and state restore
  before formal training. CPU import/load alone is insufficient.
- No credentials stored here. Existing authorized SSH runner handles connection.
