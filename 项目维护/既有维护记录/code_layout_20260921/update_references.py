"""Update only paths and associated identities, retaining original metadata."""
from pathlib import Path
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[3]
BASE=Path(__file__).resolve().parent
j=json.loads((BASE/'mapping.json').read_text(encoding='utf-8'))
mapping=j['mapping']
replacements={}
for a,b in mapping.items():
    replacements[a]=b
    if '/' in a:replacements[a.replace('/','\\')]=b.replace('/','\\')
pattern=re.compile('|'.join(re.escape(a) for a in sorted(set(replacements)|set(replacements.values()),key=len,reverse=True)))

def change_text(value):
    # Source snapshots are historical original bytes, not live entry points.
    if 'source_snapshot' in value or '/before/' in value or '\\before\\' in value:return value
    return pattern.sub(lambda match:replacements.get(match.group(),match.group()),value)

def backup(p):
    dest=BASE/'before'/p.relative_to(ROOT)
    if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())

def sha(data):return hashlib.sha256(data).hexdigest()

source_dest={str((ROOT/r['new']).resolve()).lower() for r in j['sources'] if r['new'].endswith('.py')}
changed=set()
old_hashes={}
payloads={}
originals={}
for p in (ROOT/'outputs_e2e').rglob('*.json'):
    if 'maintenance' in p.parts or any('source_snapshot' in part for part in p.parts):continue
    original=BASE/'before'/p.relative_to(ROOT)
    raw=original.read_bytes() if original.exists() else p.read_bytes()
    try:payloads[p]=json.loads(raw.decode('utf-8-sig'));originals[p]=raw
    except (UnicodeError,json.JSONDecodeError):continue

def rewrite(value):
    if isinstance(value,list):return [rewrite(v) for v in value]
    if isinstance(value,dict):
        result={change_text(k):rewrite(v) for k,v in value.items()}
        path=result.get('path')
        if isinstance(path,str) and 'sha256' in result:
            p=Path(path)
            if str(p.resolve()).lower() in source_dest or p in changed:
                raw=p.read_bytes();result['sha256']=sha(raw)
                if 'size_bytes' in result:result['size_bytes']=len(raw)
        return result
    if isinstance(value,str):return change_text(value)
    return value

for iteration in range(25):
    count=0
    for p,value in list(payloads.items()):
        new=rewrite(value)
        if new!=value or (iteration==0 and p.read_bytes()!=originals[p]):
            backup(p)
            raw=originals[p];newline='\r\n' if b'\r\n' in raw else '\n'
            text=json.dumps(new,ensure_ascii=False,indent=2).replace('\n',newline)+newline
            p.write_bytes(text.encode('utf-8'));payloads[p]=new;changed.add(p);count+=1
    if not count:break
else:raise RuntimeError('Metadata identity graph did not converge')

docs=[]
for p in [*ROOT.glob('*.md'),ROOT/'运行入口/E2E/G5/G5手动运行说明.md']:
    if p.name=='AGENTS.md':continue
    raw=p.read_bytes();text=raw.decode('utf-8-sig');new=change_text(text)
    if text!=new:backup(p);p.write_bytes(new.encode('utf-8'));docs.append(str(p.relative_to(ROOT)))

# Regenerated source hash mapping records path-only final code, not old execution hashes.
for row in j['sources']:
    row['new_sha256']=sha((ROOT/row['new']).read_bytes())
(BASE/'mapping.json').write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf-8')
report={'status':'PASS','updated_json':[str(p.relative_to(ROOT)) for p in sorted(changed)],'documents':docs,
        'identity_graph_iterations':iteration+1,'historical_originals':'before/','scope':'User-approved path and corresponding identity migration; no metric/configuration/data changes'}
(BASE/'references.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'json':len(changed),'docs':len(docs),'iterations':iteration+1}))
