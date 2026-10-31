set -euo pipefail
root=/root/autodl-tmp/pvground_clean_baseline_20261031
test ! -e "$root"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
mkdir -p "$root"
cd /tmp/PV-Ground
tar --exclude=.git --exclude=__pycache__ --exclude='*/__pycache__' --exclude=data --exclude=OpenPCDet --exclude=env --exclude=logs --exclude=tensorboard_output -cf - . | tar -xf - -C "$root"
ln -s /tmp/PV-Ground/data "$root/data"
ln -s /tmp/PV-Ground/OpenPCDet "$root/OpenPCDet"
tar -xzf /tmp/pvg_clean_overlay.tgz -C "$root"
cd "$root"
/root/autodl-tmp/PV-Ground/env/bin/python build_pvground_clean_baseline.py "$root" | tee architecture_patch.json
/root/autodl-tmp/PV-Ground/env/bin/python -m py_compile models/pv_ground.py models/losses.py train_pvground_clean_baseline.py
sha256sum models/pv_ground.py models/losses.py train_pvground_clean_baseline.py
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
