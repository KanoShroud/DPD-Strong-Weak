"""Approved path-only layout migration; original bytes retained for review."""
from pathlib import Path
import ast
import hashlib
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
mapping = {}

def put(old, folder):
    mapping[old] = folder + '/' + Path(old).name

U = '统一模型代码'
for name in ['runtime_paths','reference_artifacts','preflight_reference','materialize_reference_snapshot','classify_data_integrity','verified_feature_loader','g5_verified_io']:
    put(f'{U}/{name}.py', f'{U}/common')
for name in ['g5_runtime_v2','g5_sample_range','g5_live_log']:
    put(name+'.py', f'{U}/common')
for p in (ROOT/U).glob('e2e_g*.py'):
    gate = re.match(r'e2e_g(\d+)', p.stem)[1]
    folder = f'{U}/gates/g{gate}'
    if 'recovery' in p.stem or 'recover_' in p.stem or 'resume_' in p.stem:
        folder += '/recovery'
    if p.stem == 'e2e_g5_r1': folder += '/r1'
    put(p.relative_to(ROOT).as_posix(), folder)
for name in ['g5_r1_decode','g5_r1_report']:
    put(f'{U}/{name}.py',f'{U}/gates/g5/r1')
for name in ['resume_g1_training','recover_g1_finalize']:
    put(f'{U}/{name}.py',f'{U}/gates/g1/recovery')
put(f'{U}/diagnose_g1_hard_count_decode.py',f'{U}/gates/g1/diagnostics')
put(f'{U}/backfill_e2e_rmse.py',f'{U}/gates/g3')
for name in ['verify_e2e_g4_cache','verify_e2e_g4_final']:
    put(f'{U}/{name}.py',f'{U}/gates/g4/audits')
for p in (ROOT/U/'audits').glob('*.py'):
    if p.name == '__init__.py': continue
    n=p.stem
    if n in ['cache_read_path_probe','cache_training_path_probe','optimize_feature_loader']:
        folder=f'{U}/common/diagnostics'
    elif n.startswith('test_g5'):
        folder=f'{U}/gates/g5/' + ('r1/tests' if n=='test_g5_r1' else 'tests')
    elif n.startswith('g5_slot'):
        folder=f'{U}/gates/g5/diagnostics'
    elif 'g5' in n:
        folder=f'{U}/gates/g5/audits'
    elif n=='e2e_g2_preflight': folder=f'{U}/gates/g2'
    else: folder=f'{U}/gates/g0'
    put(p.relative_to(ROOT).as_posix(),folder)
for name,folder in {
    'G5一键继续':'运行入口/E2E/G5','G5_R1一键运行':'运行入口/E2E/G5_R1',
    'G5资源采样':'运行入口/E2E/G5','G5停止回读':f'{U}/gates/g5/recovery',
    'G5工程接入验证':f'{U}/gates/g5/diagnostics','G5样本读取验证':f'{U}/gates/g5/diagnostics',
    'g5_prefetch_probe':f'{U}/gates/g5/diagnostics',
}.items(): put(name+'.py',folder)
put('G5手动运行说明.md','运行入口/E2E/G5')

def digest(data): return hashlib.sha256(data).hexdigest()
def backup(p):
    q=BASE/'before'/p.relative_to(ROOT)
    if not q.exists():
        q.parent.mkdir(parents=True,exist_ok=True)
        q.write_bytes(p.read_bytes())

module_map={a[:-3].replace('/','.'):b[:-3].replace('/','.') for a,b in mapping.items() if a.endswith('.py')}
bare_map={Path(a).stem:b for a,b in module_map.items()}
# Path.stem treats dotted module names as suffixes; construct basename aliases explicitly.
bare_map={a.rsplit('.',1)[-1]:b for a,b in module_map.items()}
allmods={**bare_map,**module_map}

def mapped_string(s):
    if 'source_snapshot' in s or 'snapshot/' in s: return s
    replacements=dict(mapping)
    replacements.update({a.replace('/','\\'):b.replace('/','\\') for a,b in mapping.items() if '/' in a})
    pattern='|'.join(re.escape(a) for a in sorted(replacements,key=len,reverse=True))
    return re.sub(pattern,lambda match:replacements[match.group()],s)

def transform(old,new,raw):
    text=raw.decode('utf-8-sig'); newline='\r\n' if '\r\n' in text else '\n'
    text=text.replace('\r\n','\n');tree=ast.parse(text)
    lines=text.splitlines(keepends=True);starts=[0]
    for line in lines:starts.append(starts[-1]+len(line))
    def pos(line,col):return starts[line-1]+len(lines[line-1].encode('utf-8')[:col].decode('utf-8'))
    edits=[];occupied=[]
    def edit(node,value):
        lo=pos(node.lineno,node.col_offset);hi=pos(node.end_lineno,node.end_col_offset)
        if any(lo<y and hi>x for x,y in occupied):return
        edits.append((lo,hi,value));occupied.append((lo,hi))
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom):
            if node.module in allmods:
                original=ast.get_source_segment(text,node)
                edit(node,original.replace('from '+node.module+' import','from '+allmods[node.module]+' import',1))
                continue
            items=[];changed=False
            for a in node.names:
                full=(node.module+'.' if node.module else '')+a.name
                if full in module_map:
                    dest=module_map[full];parent,leaf=dest.rsplit('.',1)
                    items.append(f'from {parent} import {leaf}'+(f' as {a.asname}' if a.asname else ''));changed=True
                elif node.module in allmods:
                    items.append(f'from {allmods[node.module]} import {a.name}'+(f' as {a.asname}' if a.asname else ''));changed=True
                else:items.append(f'from {node.module} import {a.name}'+(f' as {a.asname}' if a.asname else ''))
            if changed:edit(node,('\n'+' '*node.col_offset).join(items))
        elif isinstance(node,ast.Import):
            if any(a.name in allmods for a in node.names):
                items=[]
                for a in node.names:
                    if a.name in allmods:
                        dest=allmods[a.name];parent,leaf=dest.rsplit('.',1)
                        items.append(f'from {parent} import {leaf} as {a.asname or a.name}')
                    else:items.append('import '+a.name+(f' as {a.asname}' if a.asname else ''))
                edit(node,('\n'+' '*node.col_offset).join(items))
    # Existing root variables keep their original semantic roots after relocation.
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name in ['ROOT','PROJECT_ROOT','PACKAGE_ROOT','workspace'] and '__file__' in ast.unparse(node.value):
                depth=len(Path(new).parts)-1-(1 if name=='PACKAGE_ROOT' else 0)
                edit(node.value,f'Path(__file__).resolve().parents[{depth}]')
    # Direct file-relative paths outside root assignments.
    for node in ast.walk(tree):
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Div) and '__file__' in ast.unparse(node):
            try:
                value=eval(compile(ast.Expression(node),'','eval'),{'__builtins__':{},'Path':Path,'__file__':str(ROOT/old)})
                rel=value.resolve().relative_to(ROOT).as_posix()
                if rel in mapping:rel=mapping[rel]
                depth=len(Path(new).parts)-1
                edit(node,f'Path(__file__).resolve().parents[{depth}] / {rel!r}')
            except (NameError,AttributeError,TypeError,ValueError):pass
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and isinstance(node.value,str):
            s=node.value
            # Keep filename-only process identity checks and historical snapshot suffixes.
            value=mapped_string(s)
            if s in module_map:value=module_map[s]
            if value!=s:edit(node,repr(value))
    for lo,hi,value in sorted(edits,reverse=True):text=text[:lo]+value+text[hi:]
    # Direct PyCharm execution must find the root before importing project packages.
    if "__main__" in text and 'sys.path.insert' not in text:
        t=ast.parse(text);index=0
        for n in t.body:
            if (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)) or (isinstance(n,ast.ImportFrom) and n.module=='__future__'):index=n.end_lineno
            else:break
        parts=text.splitlines(keepends=True);depth=len(Path(new).parts)-1
        parts.insert(index,f'\nimport sys as _path_sys\nfrom pathlib import Path as _PathRoot\n_path_sys.path.insert(0, str(_PathRoot(__file__).resolve().parents[{depth}]))\n')
        text=''.join(parts)
    ast.parse(text)
    return text.replace('\n',newline).encode('utf-8'),len(edits)

def main():
    if (BASE/'mapping.json').exists():raise RuntimeError('Migration already started')
    for old,new in mapping.items():
        assert (ROOT/old).is_file(),old
        assert not (ROOT/new).exists(),new
        assert (ROOT/new).resolve().is_relative_to(ROOT)
    source_rows=[]
    for old,new in mapping.items():
        p=ROOT/old;raw=p.read_bytes();backup(p)
        data,count=transform(old,new,raw) if old.endswith('.py') else (raw,0)
        q=ROOT/new;q.parent.mkdir(parents=True,exist_ok=True)
        # Both source and destination were resolved inside the explicitly approved worktree.
        p.rename(q)
        if data!=raw:q.write_bytes(data)
        source_rows.append(dict(old=old,new=new,old_sha256=digest(raw),new_sha256=digest(data),path_edits=count))
    # Unmoved model/physics modules may import relocated helpers.
    for folder in ['models','physics']:
        for p in (ROOT/U/folder).glob('*.py'):
            rel=p.relative_to(ROOT).as_posix();raw=p.read_bytes();data,count=transform(rel,rel,raw)
            if data!=raw:backup(p);p.write_bytes(data);source_rows.append(dict(old=rel,new=rel,old_sha256=digest(raw),new_sha256=digest(data),path_edits=count))
    for new in mapping.values():
        p=(ROOT/new).parent
        if new.endswith('.py'):
            while p!=ROOT:
                init=p/'__init__.py'
                if not init.exists():init.write_text('"""项目脚本分类包。"""\n',encoding='utf-8')
                p=p.parent
    (ROOT/'运行入口/S2').mkdir(parents=True,exist_ok=True)
    (BASE/'mapping.json').write_text(json.dumps({'mapping':mapping,'sources':source_rows},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'moved':len(mapping),'source_records':len(source_rows)},ensure_ascii=False))

if __name__=='__main__':main()
