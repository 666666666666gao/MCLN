from pathlib import Path
import hashlib,json,shutil,datetime
runtime=Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1')
root=Path('/root/autodl-tmp/mcln_pvground_vsa_order_source_20260908_v1')
source=runtime/'PV-Ground'
assert not root.exists()
old=(source/'models/pv_utils.py').read_bytes()
assert hashlib.sha256(old).hexdigest()=='190c0f70e075482e0c1ea136ca46e840d6c4159a426a7f096f05eb8a39e731fd'
before=b"""            cur_coords = batch_dict['multi_scale_3d_features'][src_name].indices
            cur_features = batch_dict['multi_scale_3d_features'][src_name].features.contiguous()
"""
after=b"""            sparse = batch_dict['multi_scale_3d_features'][src_name]
            cur_coords = sparse.indices
            # Stacked ball query requires contiguous batches. Canonical voxel
            # order also keeps its first-nsample selection independent of spconv row order.
            linear = cur_coords[:, 0].long()
            for axis, extent in enumerate(sparse.spatial_shape, start=1):
                linear = linear * int(extent) + cur_coords[:, axis].long()
            order = torch.argsort(linear)
            cur_coords = cur_coords[order]
            cur_features = sparse.features[order].contiguous()
"""
assert old.count(before)==1
new=old.replace(before,after)
compile(new,'pv_utils.py','exec')
root.mkdir();shutil.copytree(source,root/'PV-Ground',ignore=shutil.ignore_patterns('__pycache__'))
(root/'PV-Ground/models/pv_utils.py').write_bytes(new)
port=json.loads((runtime/'source_port.json').read_bytes())
port.update(previous_runtime_sha256=hashlib.sha256(old).hexdigest(),after_sha256=hashlib.sha256(new).hexdigest(),
    added_fix='canonical batch/z/y/x rows before stacked VSA aggregation; matching coordinate and feature permutation')
(root/'source_port.json').write_text(json.dumps(port,indent=2)+'\n')
changed=[]
for p in source.rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts:
        q=root/'PV-Ground'/p.relative_to(source)
        if p.read_bytes()!=q.read_bytes():changed.append(p.relative_to(source).as_posix())
assert changed==['models/pv_utils.py']
record=dict(status='prepared',time_cst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    root=str(root),model_source=str(root/'PV-Ground'),source_port=str(root/'source_port.json'),changed_files=changed,
    before_sha256=hashlib.sha256(old).hexdigest(),after_sha256=hashlib.sha256(new).hexdigest(),
    original_runtime_unchanged=(source/'models/pv_utils.py').read_bytes()==old,
    optimizer_steps=0,formal_rows=0,disk_free=shutil.disk_usage(root).free)
(root/'preparation.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
