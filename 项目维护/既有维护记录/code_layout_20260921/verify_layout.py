"""Static computational-body and reference checks for the approved migration."""
from pathlib import Path
import ast
import hashlib
import json
import py_compile

ROOT=Path(__file__).resolve().parents[3]
BASE=Path(__file__).resolve().parent
mapping=json.loads((BASE/'mapping.json').read_text(encoding='utf-8'))

class ComputationalBody(ast.NodeTransformer):
    def visit_Import(self,node):return None
    def visit_ImportFrom(self,node):return None
    def visit_Expr(self,node):
        if '_path_sys.path.insert' in ast.unparse(node):return None
        return self.generic_visit(node)
    def visit_Assign(self,node):
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ['ROOT','PROJECT_ROOT','PACKAGE_ROOT','workspace']:
            node.value=ast.Constant('ROOT_PATH')
            return node
        return self.generic_visit(node)
    def visit_BinOp(self,node):
        if isinstance(node.op,ast.Div) and ('.py' in ast.unparse(node) or '__file__' in ast.unparse(node)):
            return ast.Constant('SOURCE_PATH')
        return self.generic_visit(node)
    def visit_Constant(self,node):
        if isinstance(node.value,str) and ('.py' in node.value or 'G5手动运行说明.md' in node.value):
            return ast.Constant('SOURCE_PATH_TEXT')
        return node

rows=[]
for row in mapping['sources']:
    if not row['new'].endswith('.py'):continue
    old=(BASE/'before'/row['old']).read_text(encoding='utf-8-sig')
    new=(ROOT/row['new']).read_text(encoding='utf-8-sig')
    a=ast.dump(ComputationalBody().visit(ast.parse(old)),include_attributes=False)
    b=ast.dump(ComputationalBody().visit(ast.parse(new)),include_attributes=False)
    py_compile.compile(str(ROOT/row['new']),cfile=str(BASE/'compiled'/Path(row['new']).with_suffix('.pyc')),doraise=True)
    rows.append({'path':row['new'],'computational_ast_equal':a==b})
    if a!=b:print('AST DIFFERENCE',row['new'])

json_changes=[]
def differences(a,b,loc=''):
    if type(a)!=type(b):return [(loc,a,b)]
    if isinstance(a,dict):
        if a.keys()!=b.keys():return [(loc+'/<keys>',list(a),list(b))]
        return [x for k in a for x in differences(a[k],b[k],loc+'/'+k)]
    if isinstance(a,list):
        if len(a)!=len(b):return [(loc+'/<length>',len(a),len(b))]
        return [x for i,(v,w) in enumerate(zip(a,b)) for x in differences(v,w,loc+'/'+str(i))]
    return [] if a==b else [(loc,a,b)]

unexpected=[]
for old in (BASE/'before/outputs_e2e').rglob('*.json'):
    new=ROOT/old.relative_to(BASE/'before')
    a=json.loads(old.read_text(encoding='utf-8-sig'));b=json.loads(new.read_text(encoding='utf-8-sig'))
    diffs=differences(a,b)
    for loc,x,y in diffs:
        allowed=(loc.endswith('/sha256') or loc.endswith('/size_bytes') or
                 isinstance(x,str) and isinstance(y,str) and ('.py' in x or 'G5手动运行说明.md' in x))
        if not allowed:unexpected.append({'file':str(new.relative_to(ROOT)),'field':loc,'before':x,'after':y})
    json_changes.append({'file':str(new.relative_to(ROOT)),'changed_fields':len(diffs)})

bad_contract=[]
for rel in ['outputs_e2e/unified/e2e_g5/20260919_approved/engineering_v4/contract.json',
            'outputs_e2e/unified/e2e_g5_r1/20260920_approved/contract.json']:
    value=json.loads((ROOT/rel).read_text(encoding='utf-8'));rows2=value['files']+value.get('historical_contract',[])
    for r in rows2:
        p=Path(r['path']);data=p.read_bytes()
        if hashlib.sha256(data).hexdigest()!=r['sha256'] or len(data)!=r['size_bytes']:bad_contract.append(str(p))

report={'source_ast':rows,'json_updates':json_changes,'unexpected_json_changes':unexpected,'active_contract_mismatches':bad_contract,
        'moved_paths_absent':all(not (ROOT/a).exists() for a in mapping['mapping']),
        'new_paths_present':all((ROOT/b).is_file() for b in mapping['mapping'].values())}
(BASE/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'sources':len(rows),'ast_differences':sum(not r['computational_ast_equal'] for r in rows),'unexpected_json_changes':len(unexpected),'active_contract_mismatches':len(bad_contract),'paths_ok':report['moved_paths_absent'] and report['new_paths_present']}))
