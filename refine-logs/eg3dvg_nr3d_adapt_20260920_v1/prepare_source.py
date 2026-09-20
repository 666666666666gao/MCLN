import ast,datetime,hashlib,json,os,pickle,re,shutil,sys
from pathlib import Path
old=Path('/root/autodl-tmp/mcln_eg3dvg_nr3d_transfer_20260920_v1');r=Path('/root/autodl-tmp/mcln_eg3dvg_nr3d_adapt_20260920_v1');r.mkdir();source=r/'source'
assert json.loads((old/'train_dataset_preflight.json').read_text())['status']=='complete'
shutil.copytree(str(old/'train_input_source'),str(source),ignore=shutil.ignore_patterns('__pycache__'))
p=source/'src/joint_det_dataset.py';before=p.read_text();tree=ast.parse(before);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Joint3DDataset')
methods={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['_is_view_dep','_augment_nr3d']};assert len(methods)==2
standalone=ast.Module(body=[ast.ClassDef(name='Old',bases=[],keywords=[],body=[methods['_is_view_dep'],methods['_augment_nr3d']],decorator_list=[])],type_ignores=[]);ns={};exec(compile(ast.fix_missing_locations(standalone),'original_methods','exec'),ns)
lines=before.splitlines(keepends=True)
replacements={'_is_view_dep':'''    def _is_view_dep(utterance):
        """Match the original view words independent of case and punctuation."""
        rels = {'front', 'behind', 'back', 'left', 'right', 'facing',
                'leftmost', 'rightmost', 'looking', 'across'}
        return bool(set(re.findall(r'[a-z]+', utterance.lower())) & rels)
''','_augment_nr3d':'''    def _augment_nr3d(utterance):
        return not Joint3DDataset._is_view_dep(utterance)
'''}
for name,n in sorted(methods.items(),key=lambda item:item[1].lineno,reverse=True):lines[n.lineno-1:max(child.lineno for child in ast.walk(n) if hasattr(child, "lineno"))]=[replacements[name]]
after=''.join(lines);assert after.count('import random\n')==1;after=after.replace('import random\n','import random\nimport re\n');compile(after,str(p),'exec');p.write_text(after)
tree=ast.parse(after);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Joint3DDataset');funcs=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in replacements];m=ast.Module(body=[ast.ClassDef(name='Joint3DDataset',bases=[],keywords=[],body=funcs,decorator_list=[])],type_ignores=[]);fixed={'re':re};exec(compile(ast.fix_missing_locations(m),'fixed_methods','exec'),fixed)
with (old/'train_input_cache/nr3d_annotations.pkl').open('rb') as f:annos=pickle.load(f)
oldcls=ns['Old'];newcls=fixed['Joint3DDataset'];changes=[];reverse=0;flag_changes=0
for i,a in enumerate(annos):
 text=a['utterance'];was=oldcls._augment_nr3d(text);now=newcls._augment_nr3d(text)
 assert now == (not newcls._is_view_dep(text))
 if was!=now:
  changes.append({'row':i,'scan_id':a['scan_id'],'utterance':text,'old_allow':was,'new_allow':now});reverse+=int(not was and now)
 flag_changes+=int(oldcls._is_view_dep(text)!=newcls._is_view_dep(text))
assert len(annos)==32919 and len(changes)==325 and reverse==0
assert all(not newcls._augment_nr3d(s) for s in ['Facing the windows, choose the desk.','facing the window, pick the box','The chair on the LEFT.'])
assert newcls._augment_nr3d('The red chair next to the table.')
report={'status':'pass','time_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'training_rows':len(annos),'rotate_allowed_to_blocked':len(changes),'blocked_to_allowed':reverse,'view_flag_changes':flag_changes,'before_sha256':hashlib.sha256(before.encode()).hexdigest(),'after_sha256':hashlib.sha256(after.encode()).hexdigest(),'file':'src/joint_det_dataset.py','source':str(source),'model_forwards':0,'optimizer_steps':0,'active_eval_source_unchanged':True,'examples':changes[:8]}
(r/'view_fix.json').write_text(json.dumps(report,indent=2));(r/'view_fix_rows.json').write_text(json.dumps(changes,indent=2));print(json.dumps(report),flush=True)

