"""Retire the already archived inferior E3 without waiting for an unrelated cache copy."""
import ast
from pathlib import Path

root = Path(__file__).resolve().parent
text = (root / 'retire_verified_terminal_and_cache_authorized.py').read_text()
text = text.replace('Delete only the archived inferior E3 latest and closed EG-Sr3D candidate cache.',
                    'Delete only the archived inferior E3 latest; the separate cache copy remains owned.')
text = text.replace("cache = json.loads((root/'RETIRED_SR_ARRAY_ARCHIVE_COMPLETE.json').read_bytes())\nassert cache['status'] == 'ONE_RETIRED_SR_CANDIDATE_ARRAY_FULLY_ARCHIVED_SHA_VERIFIED'\nrows = [latest, cache]", 'rows = [latest]')
cache = "'/root/autodl-tmp/mcln_eg3dvg_sr3d_transfer_20260920_v1/formal/candidates.npy'"
text = text.replace(',\n    ' + cache, '').replace(',\n ' + cache, '')
text = text.replace('1150251168', '841676832').replace('deleted_count=2', 'deleted_count=1')
text = text.replace('EXACT_TWO_ARTIFACT_RETIREMENT.json', 'EXACT_E3_WEIGHT_RETIREMENT.json')
text = text.replace('RETIREMENT_STDOUT.json', 'E3_RETIREMENT_STDOUT.json')
text = text.replace('RETIREMENT_STDERR.txt', 'E3_RETIREMENT_STDERR.txt')
text = text.replace('RETIREMENT_EXIT.json', 'E3_RETIREMENT_EXIT.json')
text = text.replace('ONLY_ARCHIVED_INFERIOR_E3_AND_CLOSED_SR_CACHE_RETIRED',
                    'ONLY_ARCHIVED_INFERIOR_E3_WEIGHT_RETIRED')
text = text.replace("/cleanup_receipt.json')", "/e3_cleanup_receipt.json')")
assert cache not in text and 'RETIRED_SR_ARRAY_ARCHIVE_COMPLETE' not in text
module = ast.parse(text, feature_version=(3, 7))
embedded = next(node.value.value for node in module.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == 'code' for target in node.targets))
ast.parse(embedded, feature_version=(3, 7))
(root / 'retire_verified_E3_authorized.py').write_text(text)
print('One archived E3 weight retirement prepared; no deletion executed')
