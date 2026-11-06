"""Repair moved navigation and executable paths; leave frozen results intact."""
import ast
import io
import json
import os
import re
import tokenize
from pathlib import Path
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[2]
HOME=Path(__file__).resolve().parent
MP='研究方向/多径DPD'
WEAK='研究方向/强弱源混叠DPD'
FINITE='研究方向/有限记录DPD'
LOW='研究方向/低空与传播几何'


def main():
    registry=HOME/'映射.json'
    info=json.loads(registry.read_text(encoding='utf-8'))
    rows=[json.loads(line) for line in (HOME/'files.jsonl').read_text(encoding='utf-8').splitlines()]
    origins={r['new']:ROOT/r['old'] for r in rows}
    prefixes=sorted(info['rules'].items(),key=lambda p:len(p[0]),reverse=True)
    original_map=info.setdefault('modified_originals',{})
    changed=[]

    def relocate(p):
        p=p.resolve()
        if not p.is_relative_to(ROOT):return p
        rel=p.relative_to(ROOT).as_posix()
        for a,b in prefixes:
            if rel==a or rel.startswith(a+'/'):
                rel=b+rel[len(a):];break
        return ROOT/info.get('code_files',{}).get(rel,rel)

    replacements={}
    for a,b in prefixes:
        replacements[a]=b
        replacements[a.replace('/','\\')]=b.replace('/','\\')
        replacements[a.replace('/','\\\\')]=b.replace('/','\\\\')
    pattern=re.compile('(?:'+'|'.join(re.escape(s) for s in sorted(replacements,key=len,reverse=True))+r''')(?=[/\\'"`)\s<>]|$)''')

    def commit(file,raw,text):
        key=file.relative_to(ROOT).as_posix()
        if key not in original_map:
            original=HOME/'变更前文件'/file.relative_to(ROOT)
            original.parent.mkdir(parents=True,exist_ok=True)
            original.write_bytes(raw)
            original_map[key]=original.relative_to(ROOT).as_posix()
        tmp=file.with_name(file.name+'.layout_pending')
        tmp.write_bytes((b'\xef\xbb\xbf' if raw.startswith(b'\xef\xbb\xbf') else b'')+text.encode('utf-8'))
        os.replace(tmp,file)
        changed.append(key)

    files=[]
    for topic in (MP,WEAK,FINITE,LOW):
        for file in (ROOT/topic).rglob('*'):
            if not file.is_file():continue
            if any(x in file.parts for x in ('source','sources','source_snapshot','snapshots','build','__pycache__','runtime')):continue
            rel=file.relative_to(ROOT).as_posix()
            prose=file.suffix in ('.md','.html')
            source=file.suffix=='.py' and any('/'+s+'/' in rel for s in ('代码','运行入口','验证'))
            paper_script=file.suffix=='.py' and '/论文/' in rel and '/scripts/' in rel
            if prose or source or paper_script:files.append(file)
    for file in files:
        raw=file.read_bytes()
        try:text=raw.decode('utf-8-sig')
        except UnicodeDecodeError:continue
        initial=text
        key=file.relative_to(ROOT).as_posix()
        origin=origins.get(key,file)
        if '/研究记录/' in key and file.name in ('对话上下文.md','实验结果记录.md','研究总结.md','修改记录.md','任务与决定沿革.md'):
            origin=ROOT/('当前任务.md' if file.name=='任务与决定沿革.md' else file.name)
        if file.suffix in ('.md','.html'):
            def link(match):
                rawvalue=match.group(2)
                value=unquote(rawvalue).strip('<>')
                if value.startswith(('#','http:','https:','mailto:','data:','javascript:')):return match.group(0)
                path,sep,anchor=value.partition('#')
                path=path.removeprefix('file:///')
                old=Path(path)
                if not old.is_absolute():old=origin.parent/old
                old=old.resolve()
                if old.parent==ROOT and old.name in ('实验结果记录.md','研究总结.md','修改记录.md','对话上下文.md','当前任务.md'):
                    topic='/'.join(key.split('/')[:2])
                    if re.search('g(?:8|9|10)|supplement-formal',anchor,re.I):topic=WEAK
                    elif re.search('g11|slepian|window|precision',anchor,re.I):topic=FINITE
                    name='任务与决定沿革.md' if old.name=='当前任务.md' else old.name
                    dest=ROOT/topic/'研究记录'/name
                else:dest=relocate(old)
                if not dest.exists():return match.group(0)
                # Absolute targets make extracted sections independent of depth.
                target=dest.as_uri() if file.suffix=='.html' else dest.as_posix()
                return match.group(1)+target+(sep+anchor if sep else '')+match.group(3)
            text=re.sub(r'''((?:href|src)=["'])([^"']+)(["'])''',link,text)
            text=re.sub(r'(\]\()([^\)]+)(\))',link,text)
        if file.suffix=='.py':
            offsets=[0]
            for line in text.splitlines(keepends=True):offsets.append(offsets[-1]+len(line))
            patches=[]
            for token in tokenize.generate_tokens(io.StringIO(text).readline):
                if token.type==tokenize.STRING:
                    replacement=pattern.sub(lambda m:replacements[m.group(0)],token.string)
                    if replacement!=token.string:
                        patches.append((offsets[token.start[0]-1]+token.start[1],offsets[token.end[0]-1]+token.end[1],replacement))
            for a,b,replacement in reversed(patches):text=text[:a]+replacement+text[b:]
        else:
            text=pattern.sub(lambda m:replacements[m.group(0)],text)
        if file.suffix=='.py' and '/代码/mp_dpd/' not in key:
            for old,new in [
                ('统一模型代码.gates.g11','研究方向.有限记录DPD.代码.gates.g11'),
                ('统一模型代码.gates.research_discovery','研究方向.有限记录DPD.代码.gates.research_discovery'),
                ('统一模型代码.gates','研究方向.强弱源混叠DPD.代码.gates')]:text=text.replace(old,new)
            # Gate and entry ROOT used to depend on a now-changed directory depth.
            if '/代码/gates/' in key or '/运行入口/' in key:
                text=re.sub(r'(?m)^(\s*)(ROOT|PROJECT|root)\s*=\s*Path\(__file__\)\.resolve\(\)\.parents\[\d+\]',
                    r"\1\2 = next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').is_file())",text)
            text=text.replace("ROOT/'outputs_e2e'","ROOT/'研究方向'").replace("ROOT / 'outputs_e2e'","ROOT / '研究方向'")
            try:ast.parse(text)
            except SyntaxError as exc:raise RuntimeError(f'Invalid rewrite of {file}') from exc
        if text!=initial:commit(file,raw,text)
    info['navigation_and_source_edits']=sorted(set(info.get('navigation_and_source_edits',[])+changed))
    registry.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print('REPAIRED',len(changed))


if __name__=='__main__':main()
