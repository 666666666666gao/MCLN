import hashlib,json
from pathlib import Path
r=Path(__file__).resolve().parent
assert (r/'launch_queue.exit').read_text().strip()=='1'
assert not any((r/p).exists() for p in ['spec.json','launch.json','preflight','controller.log'])
for name in ['prepare_evaluation.py','prepare_evaluation.log']:
 old=r/name;new=r/(name+'.before_scene_count')
 assert not new.exists();old.rename(new)
rec={'failure':'Data preparation assumed ScanRefer expression scenes equal all cached ScanNet validation scenes.','actual_expression_scenes':141,'scene_cache_scenes':312,'annotation_rows':9508,'model_forwards_before_fix':0,'checkpoint_and_evaluation_rows_unchanged':True}
(r/'scene_count_repair.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec))
