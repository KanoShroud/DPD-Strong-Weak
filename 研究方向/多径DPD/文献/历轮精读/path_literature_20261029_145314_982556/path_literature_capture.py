"""Read-only Zotero/public-paper capture for a focused theoretical review."""
from pathlib import Path
from datetime import datetime
import json
import urllib.request
import urllib.parse
import ssl
import hashlib
import certifi
import fitz

ROOT = Path(__file__).resolve().parent
KEYS = ['A9X2TH3D', 'UEFFJ7PQ', 'W7YUTHCP', '6YQ38NQH', 'LH3D9LYE', 'J2ZDTX8M']
PUBLIC = {
    'EDM2026': 'https://zenodo.org/records/22095151/files/TASLPRO.2026.3717238.pdf?download=1',
    'EDM2022': 'https://arxiv.org/pdf/2205.08960',
    'Graph2010': 'https://www.iss.uni-stuttgart.de/forschung/publikationen/kreissig_itg10.pdf',
    'Graphs2011': 'https://eurasip.org/Proceedings/Ext/SPAMEC2011/papers/SPAMEC_2011_IV_1.pdf',
}
DOIS = ['10.1109/TASL.2008.2004533', '10.1109/TSP.2016.2593690', '10.1109/TASLPRO.2026.3717238']


def extract(path, dest):
    doc = fitz.open(path)
    dest.write_text('\n'.join(f'\n=== PDF PAGE {i+1} ===\n'+p.get_text() for i, p in enumerate(doc)), encoding='utf-8')
    return len(doc)


def main():
    out = ROOT/('path_literature_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    local = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base = 'http://127.0.0.1:23119/api/users/0/items/'
    ctx = ssl.create_default_context(cafile=certifi.where())
    records = []
    for key in KEYS:
        row = dict(key=key)
        try:
            item = json.load(local.open(base+key, timeout=20))
            children = json.load(local.open(base+key+'/children', timeout=20))
            row['metadata'] = item['data']
            row['pdfs'] = []
            for child in children:
                if child['data'].get('contentType') != 'application/pdf':
                    continue
                url = local.open(base+child['key']+'/file/view/url', timeout=20).read().decode('utf-8')
                try:
                    url = json.loads(url)
                except json.JSONDecodeError:
                    pass
                path = Path(urllib.request.url2pathname(urllib.parse.urlparse(url).path))
                entry = dict(attachment=child['key'], path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                entry['pages'] = extract(path, out/(key+'.txt'))
                row['pdfs'].append(entry)
                break
        except Exception as exc:
            row['error'] = repr(exc)
        records.append(row)
        print(key, row.get('metadata', {}).get('title'), row.get('error', ''), flush=True)
    for tag, url in PUBLIC.items():
        row = dict(tag=tag, url=url)
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Academic-model-review/1.0'})
            data = urllib.request.urlopen(req, context=ctx, timeout=25).read()
            if not data.startswith(b'%PDF'):
                raise ValueError('Not a PDF')
            path = out/(tag+'.pdf')
            path.write_bytes(data)
            row.update(path=str(path), sha256=hashlib.sha256(data).hexdigest(), pages=extract(path, out/(tag+'.txt')))
        except Exception as exc:
            row['error'] = repr(exc)
        records.append(row)
        print(tag, row.get('pages'), row.get('error', ''), flush=True)
    for doi in DOIS:
        row = dict(doi=doi)
        try:
            row['crossref'] = json.load(urllib.request.urlopen('https://api.crossref.org/works/'+doi, context=ctx, timeout=25))['message']
        except Exception as exc:
            row['error'] = repr(exc)
        records.append(row)
    (out/'sources.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'path_literature_capture.py').write_bytes(Path(__file__).read_bytes())
    print('OUTPUT', out, flush=True)


if __name__ == '__main__':
    main()
