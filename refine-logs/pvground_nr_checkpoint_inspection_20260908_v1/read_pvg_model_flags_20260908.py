from pathlib import Path
import json
base=Path('/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/PV-Ground')
output={}
for p in base.glob('*.py'):
    lines=p.read_text().splitlines();hits=[]
    for i,line in enumerate(lines):
        if 'butd=' in line or 'butd =' in line:
            hits.extend(str(j+1)+': '+lines[j] for j in range(max(0,i-2),min(len(lines),i+3)))
    if hits:output[p.name]=hits
print(json.dumps(output))
