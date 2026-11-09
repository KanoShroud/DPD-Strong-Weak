"""Read-only Zotero acquisition for the explicitly approved reading batch."""
import json
import argparse
import hashlib
import re
import urllib.request
import urllib.parse
from pathlib import Path
import fitz

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / '近邻重审_20261105'
CACHE = Path('C:/Users/Administrator/AppData/Local/Temp/dpd_reading_20261106')
FIRST = {'N37','N38','N41','N49','N50','N51','N52','N54','N58','N59','N61','N65'}

def get(route):
    req = urllib.request.Request('http://127.0.0.1:23119/api/users/0/'+route,
                                 headers={'Zotero-API-Version':'3'})
    with urllib.request.urlopen(req, timeout=30) as response:
        text = response.read().decode('utf-8')
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

def norm(text):
    return re.sub(r'[^a-z0-9]', '', text.lower())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--remaining', action='store_true')
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    items=[]
    for start in range(0,10000,100):
        batch=get(f'items/top?limit=100&start={start}')
        items.extend(batch)
        if len(batch)<100:
            break
    original=json.loads((OLD/'文献矩阵.json').read_text(encoding='utf-8'))
    approved=[r for r in original if r['id'] in FIRST or r.get('local_pdf')]
    approved.append({'id':'N69','title':'Passive sensing for unknown waveforms: parameter identifiability and optimization','grade':'D','doi':'','zotero_keys':['VBLFJMVL']})
    previous = json.loads((HERE/'sources.json').read_text(encoding='utf-8')) if args.remaining else []
    if args.remaining:
        done = {r['id'] for r in previous}
        approved = [r for r in original if r['id'] not in done]
    out=[]
    for row in approved:
        matches=[x for x in items if x['key'] in row.get('zotero_keys',[]) or
                 (row.get('doi') and x['data'].get('DOI','').lower()==row['doi'].lower()) or
                 norm(x['data'].get('title',''))==norm(row['title'])]
        scope = ('第三批' if row['id'] in {'N29','N30','N32','N33','N34','N35','N46','N66','N67'} else '第二批') if args.remaining else ('第一批' if row['id'] in FIRST else ('用户新增' if row['id']=='N69' else '原已入库全文'))
        result={'id':row['id'],'original_grade':row['grade'],'scope':scope, 'title':row['title'],'matches':matches,'pdfs':[]}
        for item in matches:
            for child in get('items/'+item['key']+'/children'):
                d=child['data']
                if d.get('contentType')!='application/pdf':
                    continue
                url=get('items/'+child['key']+'/file/view/url')
                if isinstance(url,dict):url=url.get('url','')
                path=Path(urllib.parse.unquote(urllib.parse.urlparse(url).path).lstrip('/'))
                if not path.is_file():continue
                doc=fitz.open(path)
                pages=[p.get_text(sort=True) for p in doc]
                dest=CACHE/(row['id']+'_'+child['key']+'.txt')
                dest.write_text('\n\n'.join(f'===== PDF PAGE {i+1} =====\n{t}' for i,t in enumerate(pages)),encoding='utf-8')
                result['pdfs'].append({'item_key':item['key'],'attachment_key':child['key'],'path':str(path),'pages':len(doc),'text_path':str(dest),'page_chars':[len(t) for t in pages], 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
                doc.close()
        out.append(result)
        print(row['id'],len(matches),[(p['pages'],p['attachment_key']) for p in result['pdfs']],flush=True)
    if args.remaining:
        (HERE/'sources_batch23.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
        out = previous + out
    (HERE/'sources.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Approved corpus:',len(out),'with PDFs:',sum(bool(x['pdfs']) for x in out))

if __name__=='__main__':main()
