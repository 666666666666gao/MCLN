import hashlib,json,os,subprocess
from pathlib import Path
directory=Path('/root/autodl-tmp/mcln_weight_cleanup_20260906_v1')
inventory=json.loads((directory/'inventory_before.json').read_text())
groups={}
for item in inventory['weights']:
    path=Path(item['path']); stat=path.stat()
    group=groups.setdefault((stat.st_dev,stat.st_ino),{'bytes':stat.st_size,'allocated_bytes':stat.st_blocks*512,'nlink':stat.st_nlink,'paths':[]})
    group['paths'].append(str(path))
for group in groups.values():
    if group['bytes']>=400000000:
        h=hashlib.sha256()
        with Path(group['paths'][0]).open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        group['sha256']=h.hexdigest()
result={'physical_weight_inode_count':len(groups),'physical_weight_allocated_bytes':sum(v['allocated_bytes'] for v in groups.values()),'groups':sorted(groups.values(),key=lambda v:v['bytes'],reverse=True)}
with (directory/'physical_inventory.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({'physical_weight_inode_count':result['physical_weight_inode_count'],'physical_weight_allocated_bytes':result['physical_weight_allocated_bytes'],'largest_unique_files':result['groups'][:22]}))
