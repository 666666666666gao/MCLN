"""Record actual reviewed launch/publication and the single live read-only waiter."""
import datetime
import json
from pathlib import Path

root = Path(__file__).parent
launch = json.loads((root/'launch.json').read_bytes())
publication = json.loads((root/'launch_publication.json').read_bytes())
wait = json.loads((root/'wait.json').read_bytes())
assert publication['actual_launch_cst'] == launch['time_cst']
assert wait['status'] == 'waiting' and wait['observations'] == []
now = datetime.datetime.now().astimezone().isoformat()
state = dict(time_cst=now, goal_turn_result='PROGRESS', goal_achieved=False,
    best_retained='original_g_5615_4495', original_g_primary_hits=[5615,4495],
    launch=launch, launch_publication=publication, publisher_native_session=57692,
    publisher_session_status='CLOSED_EXIT0', source_review_verdict='PASS',
    source_review_independence='same-family', source_review_acceptance='provisional',
    sole_waiter_native_session=5205, waiter_status='LIVE', first_observation_cst=wait['next_scheduled_cst'],
    preflight_waiter_73739='CLOSED_EXIT0', prior_publishers_88907_97217='CLOSED_EXIT0',
    formal_training_started=True, formal_result_available=False, new_method_promoted=False,
    prospective_next_structure=str(root/'FUTURE_METHOD_PLAN.md'),
    retention='verified formal9508/rows/CPU GT box thresholds/checkpoint SHA -> remove exact owned nonbest terminal; no failed local archives',
    next_steps=['Continue the same waiter5205 near planned21:54:57; do not launch a second observer/training',
        'Use actual full-loop progress to revise ETA; near stage ending, recheck only240s if incomplete',
        'Keep current source/spec/loss/budget unchanged while pair runs',
        'After actual closed terminal collect text/rows/retention receipts, analyze same-checkpoint native bbs and scope-aware audit',
        'Choose six-face decoding, final geometric readback or adaptation only from source-pair result; no repeated normalization'])
(root/'active_formal_continuation_state.json').write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')
note = ('\nPV '+now+': F formalrange control actually launched'+launch['time_cst']+
    ' controller403326, native launch903d01 exit0. SameoriginalG,local_range->whole_range serial, freshAdamW/native+G/seed2027B8/LR1e-5/fixed3723 each; '
    'only109rangevisibility, all256 retained. FreshAstra/max/forknone sourcePASS0/81files samefamilyprovisional, '
    'nativebackendSKUunverified; no duplicate successfulengineeringpreflight. Actualfree2631118848B/reserve962457382B. '
    'Single readonly waiter5205 LIVE first21:54:57.103967,40min thenactualETA/240s nearend; no secondobservers or jobs. '
    'Publisher57692 actualCLOSED0 §'+publication['section']+' main'+publication['github_main']+
    ' doc'+str(publication['handoff_bytes'])+'B/SHA'+publication['handoff_sha256']+
    ',4local+remote/Gitpayloadexact. Newuser3partfuture architecture preserved FUTURE_METHOD_PLAN.md; '
    'first109sourcecontrol keptunchanged, futuredirectiontokens/sixface/distribution/finalsingleheadgeometricreadback/optionaltrainteacher '
    'afteractualresult, no ordinaryattention or denominatorreruns. Persistentcleanup implementedcontroller: '
    'completeformal/CPUselectedGTboxthreshold/rowsSHA/terminalSHA+protectedG checks before exactownednonbestunlink, '
    'tiesincumbent, oneactiveatomiclatest, nofailedlocalarchives; necessaryparents/V99/logsprotected. '
    'No newformalaccuracyorpromotionyet; originalG5615/4495 best, 5615/4754 Scan target andNr/Sr UNMET. '
    'PriorI73739/61175/97217 andN88907 allCLOSED0 neverrerun. Nextsamewaiter/collectclosedactualrecords, auditanalyze/retention, conditionallynextmodule. GoalACTIVE_UNMET.\n')
for path in (Path(r'C:\Users\gb\memory\2026-10-03.md'), Path(r'C:\Users\gb\MEMORY.md')):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(note)
print(json.dumps(dict(time_cst=now, publication=publication['section'], waiter=5205,
    next_check_cst=wait['next_scheduled_cst'], goal_achieved=False)))
