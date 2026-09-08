from pathlib import Path
import hashlib,json
root=Path('/root/autodl-tmp/mcln_pvground_scanrefer_finetune_20260908_vsaorder_v1')
assert (root/'controller.exit').read_text().strip()=='0'
results={}
for stage in ['initial','terminal']:
    receipt=json.loads((root/stage/'receipt.json').read_bytes())
    data=(root/stage/'rows.jsonl').read_bytes()
    assert hashlib.sha256(data).hexdigest()==receipt['rows_sha256']
    rows=[json.loads(line) for line in data.splitlines()]
    assert len(rows)==6887
    results[stage]={mode:{threshold:{str(k):sum(row[mode][threshold][i] for row in rows) for i,k in enumerate([16,32,64,256])}
                         for threshold in ['oracle25','oracle50']} for mode in ['bbs','bbf']}
    for threshold in ['oracle25','oracle50']:
        assert results[stage]['bbs'][threshold]['256']==results[stage]['bbf'][threshold]['256']
print(json.dumps(dict(rows=6887,formal_rows=0,scope='GT-assisted upper bounds from fixed exported rows, not deployable accuracy',results=results)))
