"""Layout inventory, active runtime isolation, and navigation acceptance."""
import json
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HOME=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from project_paths import relocate  # noqa: E402


class Links(HTMLParser):
    def __init__(self):
        super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        self.links.extend(v for k,v in attrs if k in ('href','src') and v)


def main():
    reg=HOME/'映射.json'; info=json.loads(reg.read_text(encoding='utf-8'))
    for name in ('当前任务.md','对话上下文.md','实验结果记录.md','研究总结.md','修改记录.md'):
        info['rules'][name]='项目维护/布局迁移_20261106/原始混合文档/'+name
    info['runtime_links_preserved_in']='项目维护/工具运行时引用'
    reg.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[json.loads(s) for s in (HOME/'files.jsonl').read_text(encoding='utf-8').splitlines()]
    deliberate={
        '研究方向/多径DPD/代码/mp_dpd/api.py',
        '研究方向/多径DPD/代码/mp_dpd/paths.py',
        '研究方向/多径DPD/运行入口/定位已保存观测.py',
        '研究方向/多径DPD/运行入口/迁移运行检查.py',
        '研究方向/多径DPD/验证/migration_probe.py'}
    missing=[];size_changes=[];preserved=0
    for row in rows:
        current=ROOT/row['new']
        original=ROOT/info.get('modified_originals',{}).get(row['new'],row['new'])
        if not original.is_file():missing.append(row['new']);continue
        if original.stat().st_size!=row['bytes']:
            size_changes.append(row['new'])
            assert row['new'] in deliberate, row['new']
        else:preserved+=1
        if not current.is_file():assert row['new'] in info['code_files'],row['new']
    assert not missing,missing[:10]
    before=np.load(ROOT/'项目维护/运行闭包迁移_20261106/final.npz')
    after=np.load(HOME/'final_functional.npz')
    assert before.files==after.files
    assert all(np.array_equal(before[k],after[k],equal_nan=True) for k in before.files)
    sources=json.loads((HOME/'final_functional.json').read_text(encoding='utf-8'))['sources']
    assert all('/研究方向/多径DPD/' in p.replace('\\','/') for p in sources),sources
    assert not (ROOT/'研究方向/多径DPD/代码/mp_dpd/runtime').exists()
    # Check all moved reports, not only the currently opened HTML.
    local=0;broken=[];pages=0
    for html in (ROOT/'研究方向').rglob('*.html'):
        if any(x in html.parts for x in ('source','sources','source_snapshot','snapshots','build')):continue
        parser=Links()
        try:parser.feed(html.read_text(encoding='utf-8-sig'))
        except UnicodeError:continue
        pages+=1
        for value in parser.links:
            if value.startswith(('#','http:','https:','mailto:','data:','javascript:','zotero:')):continue
            parsed=urlsplit(value)
            part=unquote(parsed.path)
            if parsed.scheme=='file' and len(part)>2 and part[0]=='/' and part[2]==':':part=part[1:]
            target=Path(part)
            if not target.is_absolute():target=html.parent/target
            local+=1
            if not target.exists():broken.append(dict(page=html.relative_to(ROOT).as_posix(),href=value,target=str(target.resolve())))
    result=dict(file_rows=len(rows),byte_size_preserved=preserved,intentional_wrapper_edits=size_changes,
        missing_files=missing,removed_cache_and_intermediate=len(info['cleanup']),
        removed_bytes=sum(r['bytes'] for r in info['cleanup']),equal_numeric_arrays=len(before.files),
        no_other_direction_imports=True,no_runtime_round_directory=True,
        html_pages=pages,local_links=local,broken_links=broken,
        boundary='Engineering migration only; no formal performance batch or novelty claim.')
    (HOME/'验收.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='broken_links'},ensure_ascii=False,indent=2))
    print('BROKEN LINKS',len(broken));print(json.dumps(broken[:12],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
