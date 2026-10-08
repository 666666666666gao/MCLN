"""Compare legacy span preconditions at calls made by the actual CPU parser."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

root = Path(__file__).resolve().parent
source = Path('/root/autodl-tmp/mcln_scanrefer_detection_aligned_source_20260908_v1/model_source')
os.chdir(str(source))
sys.path.insert(0, str(source))
import sng_parser
from sng_parser.backends import spacy_parser
import spacy

assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
assert not (root / 'SPAN_TRACE_RECEIPT.json').exists()
records = json.loads((root / 'historical_parsed_captions.json').read_bytes())
assert len(records) == 9508
assert hashlib.sha256((root / 'historical_parsed_captions.json').read_bytes()).hexdigest() == json.loads((root / 'CAPTION_SOURCE.json').read_bytes())['captions_sha256']
current = spacy_parser.find_char_span_by_token_idx
rows = []
events = []
active_row = None
calls = 0


def shadow(index, doc):
    global calls
    calls += 1
    actual = current(index, doc)
    doc_text = doc.text + ' ABCDEF'
    words = doc_text.split()
    token = doc[index].text
    # These statuses describe the old function's actual precondition/return;
    # they do not repair it or claim a complete legacy parser execution.
    if index >= len(words) or token not in words[index]:
        status, old = 'would_raise', None
    elif token != words[index]:
        status, old = 'empty_span', []
    else:
        begin = doc_text.find(' '.join(words[index:]))
        status, old = 'span', [begin, begin + len(token)]
    if old != actual:
        events.append(dict(row_id=active_row, token_index=index, token=token,
                           legacy_status=status, legacy_span=old, current_span=actual))
    return actual


spacy_parser.find_char_span_by_token_idx = shadow
started = time.monotonic()
with (root / 'SPAN_TRACE_ROWS.jsonl').open('x') as stream:
    for record in records:
        active_row = record['row_id']
        before, call_before = len(events), calls
        nodes, edges = sng_parser.parse(record['caption'], target_selection='first_object')
        row = dict(row_id=active_row, scan_id=record['scan_id'], target_id=record['target_id'],
                   span_calls=calls-call_before, span_differences=events[before:],
                   graph_nodes=len(nodes), graph_edges=len(edges))
        rows.append(row)
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        if len(rows) % 512 == 0:
            stream.flush()
            print(json.dumps(dict(event='SPAN_TRACE_PROGRESS', rows=len(rows), total=9508,
                                  seconds=time.monotonic()-started)), flush=True)
receipt = dict(status='ACTUAL_CURRENT_PARSER_CALL_SPAN_COMPARISON_COMPLETED',
               time_cst=datetime.datetime.now().astimezone().isoformat(), rows=len(rows), span_calls=calls,
               expressions_with_any_difference=sum(bool(row['span_differences']) for row in rows),
               different_calls=len(events), legacy_would_raise_calls=sum(e['legacy_status']=='would_raise' for e in events),
               legacy_empty_span_calls=sum(e['legacy_status']=='empty_span' for e in events),
               different_valid_span_calls=sum(e['legacy_status']=='span' for e in events),
               spacy_version=spacy.__version__, parser_path=str(Path(spacy_parser.__file__).resolve()),
               parser_sha256=hashlib.sha256(Path(spacy_parser.__file__).read_bytes()).hexdigest(),
               scope='legacy preconditions/returns shadowed at actual current-parser calls; not a full legacy replay or REC evaluation',
               pv_model_forwards=0, gpu_forwards=0, optimizer_updates=0, new_weights=0,
               seconds=time.monotonic()-started)
(root / 'SPAN_TRACE_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
