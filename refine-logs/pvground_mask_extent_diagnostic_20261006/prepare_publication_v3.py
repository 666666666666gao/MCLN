"""Use the witnessed data-disk evidence directory in the publication path guard."""
import ast
from pathlib import Path

root=Path(__file__).resolve().parent
source=(root/'publish_formal_launch_v2.py').read_text(encoding='utf-8')
changes=[
    ('此发布时尚无完整9508终态或新精度。','本次发布尚未收取完整9508终态与新指标。'),
    ("publication_review=json.loads((local/'PUBLISH_LAUNCH_V2_REVIEW.json').read_bytes())",
     "publication_review=json.loads((local/'PUBLISH_LAUNCH_V3_REVIEW.json').read_bytes())"),
    ("assert not (project/bundle['prefix']).exists()",
     "evidence=(project/bundle['prefix']).resolve()\nassert str(evidence)=='/root/autodl-tmp/mcln_published_evidence_20261005/pvground_mask_extent_diagnostic_20261006'\nassert not evidence.exists()"),
    ("path=project/name;assert project.resolve() in path.resolve().parents",
     "path=project/name;assert evidence in path.resolve().parents"),
    ("'PUBLISH_LAUNCH_V2_REVIEW.json','PUBLISH_LAUNCH_V2_REVIEW.md','PUBLICATION_FAILURE_INSPECTION.json',",
     "'PUBLISH_LAUNCH_V2_REVIEW.json','PUBLISH_LAUNCH_V2_REVIEW.md','PUBLICATION_FAILURE_INSPECTION.json',\n          'PUBLISH_LAUNCH_V3_REVIEW.json','PUBLISH_LAUNCH_V3_REVIEW.md','PUBLICATION_PATH_WITNESS.json',")]
for before,after in changes:
    assert source.count(before)==1,before
    source=source.replace(before,after)
ast.parse(source)
target=root/'publish_formal_launch_v3.py';assert not target.exists()
target.write_text(source,encoding='utf-8')
print('ACTUAL_RELOCATED_EVIDENCE_ROOT_GUARD_V3_PREPARED')
