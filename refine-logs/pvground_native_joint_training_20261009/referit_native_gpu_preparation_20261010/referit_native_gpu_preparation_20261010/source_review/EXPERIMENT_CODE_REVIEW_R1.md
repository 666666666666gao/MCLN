**Round 1: FAIL — SOURCE_ONLY.**

The initial preparation recorded `d3788754842c305e16ae462651482536ae4858d2d89a527126b532852dc51132` as its preflight SHA256. The actual written file was `4ba83e46e22080e79e20bcc5345c631ea5794a15a0db7d02bf9ecbd9b7364017`. All 325 line endings are CRLF; normalizing them to LF exactly reproduces the recorded digest. `prepare_referit_native_gpu_check.py:138` hashed `content.encode()` instead of the written file. No normalized-text hash convention was intended.

B-01 blocks accepting the initial source-identity receipt. The minimal remedy is to hash the written script with `read_bytes()` and refresh the existing metadata. This requires no project execution, training, or new digest mechanism. Original exact snapshots and deterministic findings are retained in `RAW_INPUT_SNAPSHOTS.json` and `SOURCE_DETAIL_CHECKS.json`.

The copied 15 Python files match the prior same-query sources; the two init manifests change only C from true to false. The model/data/protection/recovery paths were reviewed statically. No GPU execution or admission is implied. Requested Astra/max; actual model/effort UNATTESTED, same-family/provisional.
