"""Isolate observation-state plumbing; retain the completed B source unchanged."""
from pathlib import Path
import datetime,hashlib,json,shutil

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def replace_once(text, old, new):
    assert text.count(old)==1,old
    return text.replace(old,new)

parent=Path('/root/autodl-tmp/mcln_pvground_source_query_source_20260908_v1/PV-Ground')
root=Path('/root/autodl-tmp/mcln_pvground_observation_source_20260909_v1')
expected={
    'models/pv_utils.py':'28a386005e748ea700d7b589530bd78a36b8532d408632d134924a9dff1ecd44',
    'models/pv_ground.py':'47e670fcd2f73f3640553ce1c8dae5752bf3e1aa8fb1641dca7b6fb6f3a69d51',
    'models/encoder_decoder_layers.py':'87ed3893169165effef6c9213aae6c3f15a68d10d894c5f191408813fecf191e'}
assert not root.exists()
texts={}
for name,digest in expected.items():
    raw=(parent/name).read_bytes();assert sha(raw)==digest,name
    texts[name]=raw.decode()
name='models/pv_utils.py'
texts[name]='from pvground_observation_query import bev_observation, spatial_observation\n'+texts[name]
texts[name]=replace_once(texts[name],'        point_features = torch.cat(point_features_list, dim=-1)',
    "        observations = [bev_observation(keypoints, batch_dict, self.voxel_size, self.point_cloud_range)]\n"
    "        observations += [spatial_observation(aggregate, keypoints, self.point_cloud_range)\n"
    "                         for aggregate in [self.SA_rawpoints] + list(self.SA_layers)]\n"
    "        batch_dict['source_observations'] = tuple(state.view(batch_size, self.n_keypoints, -1) for state in observations)\n"
    '        point_features = torch.cat(point_features_list, dim=-1)')
name='models/pv_ground.py'
texts[name]=replace_once(texts[name],"        end_points['source_features'] = voxel_batch['point_features_before_fusion']",
    "        end_points['source_features'] = voxel_batch['point_features_before_fusion']\n        end_points['source_observations'] = voxel_batch['source_observations']")
texts[name]=replace_once(texts[name],'                source_position=point_position\n',
    "                source_position=point_position,\n                source_observations=end_points['source_observations']\n")
name='models/encoder_decoder_layers.py'
texts[name]=replace_once(texts[name],'                source_features=None, source_position=None):',
    '                source_features=None, source_position=None, source_observations=None):')
texts[name]=replace_once(texts[name],'                query + query_pos, source_features, source_position)',
    '                query + query_pos, source_features, source_position, source_observations)')
for name,text in texts.items():compile(text,name,'exec')
root.mkdir();shutil.copytree(parent,root/'PV-Ground',ignore=shutil.ignore_patterns('__pycache__'))
for name,text in texts.items():(root/'PV-Ground'/name).write_text(text)
files={}
for path in parent.rglob('*'):
    if path.is_file() and '__pycache__' not in path.parts:
        name=path.relative_to(parent).as_posix()
        assert name in texts or path.read_bytes()==(root/'PV-Ground'/name).read_bytes(),name
        files[name]=sha((root/'PV-Ground'/name).read_bytes())
for name,digest in expected.items():assert sha((parent/name).read_bytes())==digest
record=dict(status='prepared',time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    root=str(root),model_source=str(root/'PV-Ground'),parent_source=str(parent),parent_sha256=expected,
    changed_sha256={name:files[name] for name in texts},files=files,parent_unchanged=True,runtime_unchanged=True,
    model_forwards=0,optimizer_steps=0,formal_rows=0,disk_free=shutil.disk_usage(root).free)
(root/'source_port.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='files'}))
