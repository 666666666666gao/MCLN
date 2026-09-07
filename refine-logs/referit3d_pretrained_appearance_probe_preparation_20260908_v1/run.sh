cd /root/autodl-tmp/mcln_referit3d_pretrained_appearance_probe_preparation_20260908_v1
/root/miniconda3/envs/bdetr/bin/python -u scripts/queue_referit3d_pretrained_appearance_probe.py --plan /root/autodl-tmp/mcln_referit3d_pretrained_appearance_probe_preparation_20260908_v1/plan.json > queue.log 2>&1
code=$?
printf "%s\n" "$code" > queue.exit
exit "$code"
