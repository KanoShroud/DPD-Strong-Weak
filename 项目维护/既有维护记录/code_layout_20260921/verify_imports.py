from pathlib import Path
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
BASE=Path(__file__).resolve().parent
mapping=json.loads((BASE/'mapping.json').read_text(encoding='utf-8'))['mapping']
rows=[]
for new in mapping.values():
    if not new.endswith('.py'):continue
    module=new[:-3].replace('/','.')
    code='import importlib,pathlib; m=importlib.import_module('+repr(module)+'); assert pathlib.Path(m.__file__).resolve()==pathlib.Path('+repr(str(ROOT/new))+').resolve()'
    result=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=40,
                          env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8:replace','PYTHONDONTWRITEBYTECODE':'1'})
    rows.append({'path':new,'returncode':result.returncode,'stderr':result.stderr[-2500:]})
    if result.returncode: print('FAIL',new,result.stderr[-1500:],flush=True)
    if len(rows)%15==0:print('import checked',len(rows),flush=True)
(BASE/'imports.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
print('COMPLETE',len(rows),'failures',sum(r['returncode']!=0 for r in rows),flush=True)
