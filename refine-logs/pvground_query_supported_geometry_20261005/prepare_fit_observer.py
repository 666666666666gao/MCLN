"""Reuse the existing single-observer lifecycle for the full sequential pair."""
from pathlib import Path

root=Path(__file__).resolve().parent
source=(root.parent/'pvground_final_quality_20261005/observe_quality_authorized.py').read_text(encoding='utf-8')
source=source.replace("stage = json.loads((local/'OBSERVER_STAGE.json').read_bytes())['stage']", "stage = 'fit'")
source=source.replace("local/'quality_fit_spec.json'", "local/'control_spec.json'")
source=source.replace("/controller.py --stage ", "/controller.py --phase ")
source=source.replace('QUALITY_OBSERVATION','QUERY_GEOMETRY_OBSERVATION').replace('QUALITY_OBSERVER_CLOSED','QUERY_GEOMETRY_OBSERVER_CLOSED')
path=root/'observe_fit_authorized.py'
assert not path.exists()
path.write_text(source,encoding='utf-8')
print(str(path))
