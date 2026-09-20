"""Exercise the actual queued controller's decision prefix on isolated CPU fixtures."""
import ast,json,hashlib
from pathlib import Path
r=Path('/root/autodl-tmp/mcln_eg3dvg_sr3d_adapt_20260920_v1')
source=(r/'controller.py').read_text()
tree=ast.parse(source)
cut=next(i for i,node in enumerate(tree.body) if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='lock' for t in node.targets))
prefix=ast.Module(body=tree.body[:cut])
cases=[('nr25_below',4725,4059,None,None,'deferred_for_nr_priority'),
       ('nr50_below',4726,4058,None,None,'deferred_for_nr_priority'),
       ('sr_exact_protected',4726,4059,12139,10335,'skipped_zero_update_already_meets_protected'),
       ('sr25_below',4726,4059,12138,10335,'run_fixed_adaptation'),
       ('sr50_below',4726,4059,12139,10334,'run_fixed_adaptation')]
base=r/'entry_policy_checks';base.mkdir()
records=[]
for name,n25,n50,s25,s50,expected in cases:
    fixture=base/name;fixture.mkdir()
    nr=fixture/'nr';(nr/'evaluation/formal').mkdir(parents=True)
    (nr/'controller.exit').write_text('0\n')
    (nr/'evaluation/formal/audit.json').write_text(json.dumps({'integrity_pass':True,'metrics':{'bbs':{'rec_hits25':n25,'rec_hits50':n50}}}))
    sr=fixture/'sr'
    if s25 is not None:
        (sr/'formal').mkdir(parents=True)
        (sr/'controller.exit').write_text('0\n')
        (sr/'formal/audit.json').write_text(json.dumps({'integrity_pass':True,'metrics':{'bbs':{'rec_hits25':s25,'rec_hits50':s50}}}))
    spec={'nr_adaptation_root':str(nr),'transfer_root':str(sr),'required_nr_hits25':4726,'required_nr_hits50':4059,'protected_sr_hits25':12139,'protected_sr_hits50':10335}
    (fixture/'spec.json').write_text(json.dumps(spec))
    ns={'__file__':str(fixture/'controller.py'),'__name__':'__policy_fixture__'}
    code=None
    try:
        exec(compile(prefix,str(r/'controller.py'),'exec'),ns)
    except SystemExit as exit_signal:
        code=exit_signal.code
    decision=json.loads((fixture/'decision.json').read_text())
    assert decision['action']==expected
    if expected=='run_fixed_adaptation':
        assert code is None and not (fixture/'controller.exit').exists() and decision['proceed_to_gpu_preflight']
    else:
        assert code==0 and (fixture/'controller.exit').read_text().strip()=='0' and not decision['proceed_to_gpu_preflight']
    records.append({'case':name,'expected':expected,'actual':decision['action'],'pass':True})
report={'status':'pass','controller_sha256':hashlib.sha256(source.encode()).hexdigest(),'cases':records,
        'production_files_changed':False,'model_forwards':0,'optimizer_steps':0,
        'scope':'Actual controller prefix before GPU lock; isolated fixture artifacts; not GPU or training validation.'}
(r/'entry_policy_checks.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
