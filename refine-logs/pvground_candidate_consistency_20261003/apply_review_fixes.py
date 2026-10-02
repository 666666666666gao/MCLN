"""Preserve the actual first review, then fix only its concrete blockers."""
import datetime
import json
from pathlib import Path
import shutil

local = Path(__file__).parent
review = json.loads((local / 'EXPERIMENT_CODE_REVIEW.json').read_bytes())
assert review['verdict'] == 'FAIL' and review['blocking_findings']
archive = local / 'review_round1'
archive.mkdir()
names = ('EXPERIMENT_CODE_REVIEW.md', 'EXPERIMENT_CODE_REVIEW.json',
         'pvground_candidate_consistency.py', 'run.py', 'build_runner.py',
         'cpu_test.py', 'pair.py', 'prepare_pair.py', 'run_preflight_authorized.py',
         'preflight_spec.json', 'g_control_spec.json', 'g_consistent_spec.json')
for name in names:
    shutil.copyfile(local / name, archive / name)
trace = local / '.aris/traces/experiment-bridge/2026-10-03_candidate_consistency_run01'
shutil.copyfile(archive / 'EXPERIMENT_CODE_REVIEW.md', trace / '001-code-review.response.md')
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
meta = dict(call_number=1, purpose='candidate-consistency-code-review', timestamp=now,
            agent_id='/root/pv_candidate_consistency_code_review', model='gpt-6-astra',
            reasoning_effort='max', reviewer_family='openai', review_independence='same-family',
            acceptance_status='provisional', status='ok', verdict=review['verdict'])
(trace / '001-code-review.meta.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
events = local / '.aris/meta/events.jsonl'
events.parent.mkdir(parents=True, exist_ok=True)
with events.open('a', encoding='utf-8') as stream:
    stream.write(json.dumps(dict(event='review_trace', skill='experiment-bridge',
        purpose=meta['purpose'], agent_id=meta['agent_id'], trace_path=str(trace), status='ok')) + '\n')
path = local / 'pvground_candidate_consistency.py'
text = path.read_text(encoding='utf-8')
before = 'additional = selected[bid].nonzero(as_tuple=False).flatten()'
assert text.count(before) == 1
path.write_text(text.replace(before, before + '.to(queries.device)'), encoding='utf-8')
path = local / 'build_runner.py'
text = path.read_text(encoding='utf-8')
before = '        "    assert sha(output/\'pvground_candidate_consistency.py\') == spec[\'consistency_module_sha256\']\\n"\n'
assert text.count(before) == 1
text = text.replace(before, '')
needle = "(local / 'run.py').write_text(text, encoding='utf-8')"
assert text.count(needle) == 1
patch = '''# Bind existing pins to the module bytes actually imported by the common runner.
replace("    assert sha(output/'pvground_semantic_assignment.py')==spec['assignment_module_sha256']\\n", "")
replace("    from pvground_candidate_consistency import candidate_consistency_correction, verify_consistency_replacement\\n",
        "    from pvground_candidate_consistency import candidate_consistency_correction, verify_consistency_replacement\\n"
        "    assert sha(sys.modules['pvground_semantic_assignment'].__file__) == spec['assignment_module_sha256']\\n"
        "    assert sha(sys.modules['pvground_candidate_consistency'].__file__) == spec['consistency_module_sha256']\\n"
        "    import pvground_task_observation_query\\n")
for module_name in ('pvground_observation_query', 'pvground_task_observation_query', 'pvground_source_query'):
    before = "sha(output/'" + module_name + ".py')"
    assert text.count(before) == 1
    text = text.replace(before, "sha(sys.modules['" + module_name + "'].__file__)")
replace("    port = json.loads(Path(spec['source_port']).read_bytes())\\n",
        "    assert sha(spec['source_port']) == spec['source_port_sha256']\\n"
        "    port = json.loads(Path(spec['source_port']).read_bytes())\\n")
replace("    imported['evaluator']=str(evaluator_path)\\n",
        "    imported['evaluator']=str(evaluator_path)\\n"
        "    for name in ('pvground_semantic_assignment', 'pvground_candidate_consistency',\\n"
        "                 'pvground_task_observation_query', 'pvground_observation_query', 'pvground_source_query'):\\n"
        "        imported[name] = str(Path(sys.modules[name].__file__).resolve())\\n")
'''
path.write_text(text.replace(needle, patch + '\n' + needle), encoding='utf-8')
print(json.dumps(dict(initial_review_preserved=True, fixed_cpu_cuda_indices=True,
                      corrected_existing_import_and_source_port_checks=True)))
